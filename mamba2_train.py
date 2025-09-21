import datetime

import math
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.mamba2 import MambaLM
from lib.byte_tokenizer import ByteTokenizer


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    tokenizer = ByteTokenizer(add_eos=True)
    tok_noeos = ByteTokenizer(add_eos=False)
    MAX_LEN = 512
    NUM_TRAIN_EXAMPLES = 600_000

    stories_ds = load_dataset('roneneldan/TinyStories', split='train')
    stories_ds = stories_ds.select(range(NUM_TRAIN_EXAMPLES))

    class PackedLM(Dataset):
        """
        Concatenate all tokenized examples (each ends with EOS) and slice
        fixed windows of length (max_len + 1) to form (input, target) pairs.
        """

        def __init__(self, hf_split, tokenizer: ByteTokenizer, max_len: int, stride: int | None = None):
            self.max_len = max_len
            window_len = max_len + 1  # input+target length
            self.stride = window_len if stride is None else stride  # non-overlapping by default

            buf: list[int] = []
            for rec in hf_split:
                text = (rec["text"] or "").strip()
                ids = tokenizer.encode(text)  # already appends EOS (by construction)
                if not ids:
                    ids = [tokenizer.EOS]
                buf.extend(ids)

            self.seqs: list[list[int]] = []
            # Build windows [i : i+window_len] with given stride
            for i in range(0, max(0, len(buf) - window_len + 1), self.stride):
                self.seqs.append(buf[i:i + window_len])

            if not self.seqs:
                raise RuntimeError("PackedLM: buffer too small for the chosen MAX_LEN.")

        def __len__(self) -> int:
            return len(self.seqs)

        def __getitem__(self, idx):
            seq = torch.tensor(self.seqs[idx], dtype=torch.long)
            # Next-token prediction: shift by 1
            return seq[:-1], seq[1:]

    stories_train_ds = PackedLM(stories_ds, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        stories_train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    print("Loader length:", len(train_loader))
    print("Total tokens:", len(stories_train_ds) * MAX_LEN)
    sample_inputs, sample_targets = train_loader.__iter__().__next__()
    print(f"Example input:\n\n{tokenizer.decode(sample_inputs[0])}\n")
    print(f"Example target:\n\n{tokenizer.decode(sample_targets[0])}\n")

    # Model hyperparameters
    vocab_size = len(tokenizer)
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_state = d_model // n_heads
    dropout = 0.1

    # TODO: Add ema model
    model = MambaLM(vocab_size, d_model, n_layers, n_heads, d_state, dropout)
    model = model.to(device)

    # Training hyperparameters
    train_epochs = 1
    train_log_interval = 10
    sample_interval = 100
    sample_length = 256
    accum_steps = 2
    base_lr = 3e-4
    min_lr = 3e-5
    total_steps = len(train_loader) * train_epochs // accum_steps
    warmup_steps = int(total_steps * 0.01)

    print(f'Total training steps: {total_steps}, Warmup steps: {warmup_steps}')

    def get_lr(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'runs/{timestamp}_mamba2'
    writer = SummaryWriter(log_dir=save_dir)

    # Fixed prompt for sampling
    fixed_prompt = (
        "Once upon a time, in a nice little town, there lived a big dragon."
    )
    fixed_prompt_ids = torch.tensor(
        tok_noeos.encode(fixed_prompt)[:MAX_LEN],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    best_loss = float('inf')

    for epoch in range(train_epochs):
        epoch_total_loss = 0.0
        epoch_batches = 0
        model.train()

        # micro-batch counter for accumulation
        micro_i = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # reset after EOS inside window (you already handle EOS correctly)
            eos_mask = (inputs == tokenizer.EOS)
            reset_mask = torch.zeros_like(inputs, dtype=torch.bool)
            reset_mask[:, 1:] = eos_mask[:, :-1]

            # autocast (bfloat16 on CUDA); safe to leave on CPU too (noop)
            use_cuda = (device.type == 'cuda')
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                logits, _ = model(inputs, states=None, reset_mask=reset_mask)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # average the loss across micro-batches so each update sees the mean
            loss = loss / accum_steps
            loss.backward()

            micro_i += 1
            epoch_total_loss += loss.item() * accum_steps  # undo the division for reporting
            epoch_batches += 1

            if micro_i % accum_steps == 0:
                # (optional) update LR schedule just before the step
                lr = get_lr(step)
                for g in optimizer.param_groups:
                    g['lr'] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # this is a *real* optimizer step
                step += 1

                # ---- logging & checkpointing on real steps ----
                if step % train_log_interval == 0:
                    avg_loss = epoch_total_loss / epoch_batches
                    writer.add_scalar('train/loss', avg_loss, step)
                    writer.add_scalar('train/bpb', avg_loss / math.log(2), step)
                    writer.add_scalar('train/ppl', math.exp(avg_loss), step)

                    # save checkpoints
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": {
                            "vocab_size": model.emb.num_embeddings,
                            "d_model": model.emb.embedding_dim,
                            "n_layers": model.n_layers,
                            "n_heads": model.n_heads,
                            "d_state": model.d_state,
                            "dropout": model.dropout,
                        },
                    }
                    torch.save(ckpt, f'{save_dir}/last.pt')
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(ckpt, f'{save_dir}/best.pt')

                    # reset running stats
                    epoch_total_loss = 0.0
                    epoch_batches = 0
                    print(f'Step {step}, Loss: {avg_loss:.4f}, LR: {lr:.2e}')

                # ---- sampling on real steps ----
                if step % sample_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        gen_ids = model.generate(
                            fixed_prompt_ids,
                            max_new_tokens=sample_length,
                            temperature=0.6,
                            top_p=0.92,
                            top_k=None,
                            eos_id=tokenizer.EOS,
                        )
                    new_tokens = gen_ids[0][fixed_prompt_ids.size(1):]
                    sample_text = tokenizer.decode(new_tokens.tolist())
                    print('\n--- Sample Generation ---')
                    print(f'{fixed_prompt}{sample_text}')
                    print('-------------------------\n')
                    writer.add_text('samples', f'{fixed_prompt}{sample_text}', step)
                    model.train()

        print(f'Epoch {epoch + 1} complete')


if __name__ == '__main__':
    main()
