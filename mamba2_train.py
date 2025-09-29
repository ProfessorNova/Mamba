import datetime
import math
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset, tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from lib.mamba2 import MambaLM


# Pretty-print large numbers with K/M/B suffixes
def _fmt(n: int) -> str:
    for suf, div in (('B', 1_000_000_000), ('M', 1_000_000), ('K', 1_000)):
        if n >= div:
            val = n / div
            return f"{val:.2f}{suf}" if val < 10 else f"{val:.0f}{suf}"
    return str(n)


def seed_everything(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PackedLM(Dataset):
    def __init__(self, hf_split, tokenizer: GPT2TokenizerWrapper, max_len: int, stride: Optional[int] = None):
        self.max_len = max_len
        window_len = max_len + 1  # input+target length
        self.stride = window_len if stride is None else stride  # non-overlapping by default

        buf: list[int] = []
        for rec in tqdm(hf_split, desc="Tokenizing dataset"):
            text = (rec["text"] or "").strip()
            ids = tokenizer.encode(text)  # appends EOS when add_eos=True
            if not ids:
                ids = [tokenizer.EOS]
            buf.extend(ids)

        self.seqs: list[list[int]] = []
        # Build windows [i : i+window_len] with given stride
        for i in tqdm(range(0, max(0, len(buf) - window_len + 1), self.stride), desc="Packing dataset"):
            self.seqs.append(buf[i:i + window_len])

        if not self.seqs:
            raise RuntimeError("PackedLM: buffer too small for the chosen MAX_LEN.")

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        # Next-token prediction: shift by 1
        return seq[:-1], seq[1:]


def main():
    seed_everything(1337)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # ---- GPT-2 tokenizers ----
    tokenizer = GPT2TokenizerWrapper(add_eos=True)
    tok_no_eos = GPT2TokenizerWrapper(add_eos=False)

    MAX_LEN = 512
    BATCH_SIZE = 8

    stories_ds = load_dataset('roneneldan/TinyStories', split='train')

    stories_train_ds = PackedLM(stories_ds, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(
        stories_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print("Total tokens:", _fmt(len(stories_train_ds) * MAX_LEN))
    sample_inputs, sample_targets = next(iter(train_loader))
    print(f"Example input:\n\n{tokenizer.decode(sample_inputs[0].tolist())}\n")
    print(f"Example target:\n\n{tokenizer.decode(sample_targets[0].tolist())}\n")

    # Model hyperparameters
    vocab_size = len(tokenizer)
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_state = 294

    model = MambaLM(vocab_size, d_model, n_layers, n_heads, d_state).to(device)
    print("Model parameters:", _fmt(sum(p.numel() for p in model.parameters())))

    # Training hyperparameters
    TRAIN_EPOCHS = 1
    TRAIN_LOG_INTERVAL = 10
    SAMPLE_INTERVAL = 100
    SAMPLE_LENGTH = 256
    ACCUM_STEPS = 32
    total_steps = int(len(train_loader) * TRAIN_EPOCHS / ACCUM_STEPS)
    print(f'Total training steps: {_fmt(total_steps)}')
    print(f'Tokens per step: {_fmt(BATCH_SIZE * MAX_LEN * ACCUM_STEPS)}')

    # Optimizer, scheduler, criterion
    BASE_LR = 6e-4
    MIN_LR = 1e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1,
    )
    criterion = nn.CrossEntropyLoss()

    # Cosine w/ warmup scheduler (step-wise)
    warmup_steps = max(100, int(0.03 * total_steps))
    print(f'Warmup steps: {warmup_steps}')

    def lr_lambda(lr_step):
        if lr_step < warmup_steps:
            return max(1e-8, lr_step / max(1, warmup_steps))
        progress = min(1.0, lr_step / max(1, total_steps))
        decay_to = MIN_LR / BASE_LR
        return decay_to + (1 - decay_to) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'runs/{timestamp}_mamba2'
    writer = SummaryWriter(log_dir=save_dir)

    # Fixed prompt for sampling
    fixed_prompt = "Once upon a time, in a nice little town, there lived a big dragon."
    fixed_prompt_ids = torch.tensor(
        tok_no_eos.encode(fixed_prompt)[:MAX_LEN], dtype=torch.long
    ).unsqueeze(0).to(device)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    best_loss = float('inf')

    for epoch in range(TRAIN_EPOCHS):
        epoch_total_loss = 0.0
        epoch_batches = 0
        model.train()

        micro_i = 0
        for inputs, targets in train_loader:
            inputs: torch.Tensor = inputs.to(device, non_blocking=True)
            targets: torch.Tensor = targets.to(device, non_blocking=True)

            # reset after EOS inside window
            eos_mask = (inputs == tokenizer.EOS)
            reset_mask = torch.zeros_like(inputs, dtype=torch.bool)
            reset_mask[:, 1:] = eos_mask[:, :-1]

            # autocast (bfloat16 on CUDA)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                logits, _ = model(inputs, states=None, reset_mask=reset_mask)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

            # grad accumulation
            (loss / ACCUM_STEPS).backward()
            micro_i += 1
            epoch_total_loss += loss.item()
            epoch_batches += 1

            if micro_i % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # ---- Scheduler step ----
                scheduler.step()

                # ---- bookkeeping ----
                step += 1

                # ---- logging & checkpointing ----
                if step % TRAIN_LOG_INTERVAL == 0:
                    avg_loss = epoch_total_loss / max(1, epoch_batches)
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
                    lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('train/loss', avg_loss, step)
                    writer.add_scalar('train/ppl', ppl, step)
                    writer.add_scalar('train/lr', lr, step)
                    print(f'Step {step}, Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, LR: {lr:.2e}')

                    # save checkpoints
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": {
                            "vocab_size": vocab_size,
                            "d_model": d_model,
                            "n_layers": n_layers,
                            "n_heads": n_heads,
                            "d_state": d_state,
                        },
                        "step": step,
                        "epoch": epoch,
                    }
                    torch.save(ckpt, f'{save_dir}/last.pt')

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(ckpt, f'{save_dir}/best.pt')

                    # reset running stats
                    epoch_total_loss = 0.0
                    epoch_batches = 0

                if step % SAMPLE_INTERVAL == 0:
                    model.eval()
                    with torch.no_grad():
                        gen_ids = model.generate(
                            fixed_prompt_ids,
                            max_new_tokens=SAMPLE_LENGTH,
                            temperature=0.6,
                            top_k=None,
                            top_p=0.92,
                            eos_id=tokenizer.EOS,
                        )
                    new_tokens = gen_ids[0][fixed_prompt_ids.size(1):]
                    sample_text = tokenizer.decode(new_tokens.tolist())
                    print('\n--- Sample Generation ---')
                    print(f'{fixed_prompt}{sample_text}')
                    print('-------------------------------\n')
                    writer.add_text('samples', f'{fixed_prompt}{sample_text}', step)
                    model.train()

        print(f'Epoch {epoch + 1} complete')

    # Save final copies
    torch.save(model.state_dict(), f'{save_dir}/final.pt')
    print("Training complete.")


if __name__ == '__main__':
    main()
