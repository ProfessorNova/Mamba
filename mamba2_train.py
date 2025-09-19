import datetime

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

    tokenizer = ByteTokenizer(add_bos=False, add_eos=True)
    tok_noeos = ByteTokenizer(add_bos=False, add_eos=False)
    MAX_LEN = 2048
    NUM_TRAIN_EXAMPLES = 250_000

    stories_ds = load_dataset('roneneldan/TinyStories', split='train')
    stories_ds = stories_ds.select(range(NUM_TRAIN_EXAMPLES))

    # Packed, fixed-length LM dataset (no padding, no masks).
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
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    print("Loader length:", len(train_loader))
    sample_inputs, sample_targets = train_loader.__iter__().__next__()
    print(f"Example input:\n\n{tokenizer.decode(sample_inputs[0])[:512]}\n")
    print(f"Example target:\n\n{tokenizer.decode(sample_targets[0])[:512]}\n")

    # Model hyperparameters
    vocab_size = len(tokenizer)
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_state = d_model // n_heads
    dropout = 0.1

    model = MambaLM(vocab_size, d_model, n_layers, n_heads, d_state, dropout)
    model = model.to(device)

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Training hyperparameters
    train_epochs = 5
    train_log_interval = 10
    sample_interval = 100
    sample_length = 256

    # TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = f'runs/{timestamp}_mamba2'
    writer = SummaryWriter(log_dir=save_dir)

    # Training loop
    step = 0
    best_loss = float('inf')

    # Fixed prompt for sampling
    fixed_prompt = (
        "Once upon a time, in a nice little town, there lived a big dragon."
    )
    fixed_prompt_ids = torch.tensor(
        tok_noeos.encode(fixed_prompt)[:MAX_LEN],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    for epoch in range(train_epochs):
        epoch_total_loss = 0.0
        epoch_batches = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_value = loss.item()
            epoch_total_loss += loss_value
            epoch_batches += 1
            writer.add_scalar('train/loss', loss_value, step)
            step += 1
            if step % train_log_interval == 0:
                avg_loss = epoch_total_loss / epoch_batches
                writer.add_scalar('train/loss_avg', avg_loss, step)
                # Save model checkpoint
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
                # reset running averages
                epoch_total_loss = 0.0
                epoch_batches = 0
                print(f'Step {step}, Loss: {avg_loss:.4f}')
            if step % sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    gen_ids = model.generate(fixed_prompt_ids, max_new_tokens=sample_length, temperature=0.7, top_k=50)
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
