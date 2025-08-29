import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset, tqdm
from torch.utils.data import DataLoader, Dataset

from lib.mamba import MambaConfig, MambaModel
from lib.tokenizer import SimpleTokenizer


def format_alpaca(example: dict) -> str:
    """Format one Alpaca record as prompt+response text."""
    inst = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()
    if inp:
        return (
                "### Instruction:\n" + inst + "\n\n" + "### Input:\n" + inp + "\n\n" + "### Response:\n" + out + "\n"
        )
    else:
        return (
                "### Instruction:\n" + inst + "\n\n" + "### Response:\n" + out + "\n"
        )


def build_prompt_for_inference(example: dict) -> str:
    """Prompt that stops right before the expected response."""
    inst = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    if inp:
        return (
                "### Instruction:\n" + inst + "\n\n" + "### Input:\n" + inp + "\n\n" + "### Response:\n"
        )
    else:
        return (
                "### Instruction:\n" + inst + "\n\n" + "### Response:\n"
        )


class CharDataset(Dataset):
    """Returns (context, target) windows for next-char prediction over one long id stream."""

    def __init__(self, ids: List[int], block_size: int) -> None:
        self.ids = ids
        self.block_size = block_size
        self.length = max(0, len(ids) - block_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx: idx + self.block_size]
        y = self.ids[idx + 1: idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def generate_greedy(model: MambaModel, tokenizer: SimpleTokenizer,
                    prompt: str, max_new_tokens: int, device: torch.device) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = int(torch.argmax(logits[0, -1]).item())
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(ids[0].tolist())


def train(args: argparse.Namespace) -> None:
    # Load HF dataset and make a split (Alpaca only has 'train')
    ds = load_dataset(args.dataset_name)
    base = ds["train"]
    split = base.train_test_split(test_size=0.0001, seed=args.seed)
    train_ds_hf, val_ds_hf = split["train"], split["test"]

    # Format each record as prompt+response text
    train_texts = [format_alpaca(r) for r in train_ds_hf]
    val_texts = [format_alpaca(r) for r in val_ds_hf]

    # Build tokenizer from training text only (to avoid leakage)
    full_train_text = "\n\n".join(train_texts)
    tokenizer = SimpleTokenizer(full_train_text)

    train_ids = tokenizer.encode(full_train_text)
    full_val_text = "\n\n".join(val_texts)
    val_ids = tokenizer.encode(full_val_text)

    # Model config
    config = MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=tokenizer.vocab_size,
        d_state=args.d_state,
        expand=args.expand,
        dt_rank=args.dt_rank,
        d_conv=args.d_conv,
    )

    # Datasets / loaders
    train_data = CharDataset(train_ids, block_size=args.block_size)
    val_data = CharDataset(val_ids, block_size=args.block_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model / loss / opt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MambaModel(config).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # For log-interval inference: pick a random validation example prompt
    val_example_prompt = build_prompt_for_inference(random.choice(val_ds_hf))

    # Train
    global_step = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epoch"):
        model.train()
        running = 0.0
        for step, (x, y) in tqdm(enumerate(train_loader, start=1), desc="Train", total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            if step % args.log_interval == 0:
                avg = running / args.log_interval
                print(f"[epoch {epoch} step {step}] train_loss={avg:.4f}")

                # quick validation loss
                model.eval()
                with torch.no_grad():
                    vloss_sum, vcount = 0.0, 0
                    for vx, vy in tqdm(val_loader, desc="Val"):
                        vx, vy = vx.to(device), vy.to(device)
                        vlogits = model(vx)
                        vloss = criterion(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                        vloss_sum += vloss.item()
                        vcount += 1
                vavg = vloss_sum / max(1, vcount)
                print(f"[epoch {epoch} step {step}] val_loss={vavg:.4f}")

                # short greedy sample for qualitative check
                sample = generate_greedy(model, tokenizer, val_example_prompt,
                                         max_new_tokens=args.sample_chars, device=device)
                print("----- sample (greedy) -----")
                print(sample[: args.sample_print_chars])
                print("---------------------------")

                running = 0.0
                model.train()

        # end-of-epoch validation as well (optional / redundant but nice to have)
        model.eval()
        with torch.no_grad():
            vloss_sum, vcount = 0.0, 0
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vlogits = model(vx)
                vloss = criterion(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                vloss_sum += vloss.item()
                vcount += 1
        vavg = vloss_sum / max(1, vcount)
        print(f"[epoch {epoch}] end_val_loss={vavg:.4f}")

    # Save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'mamba.pt')
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    tokenizer.save(str(out_dir / 'tokenizer.json'))
    print(f"Saved model + tokenizer to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mamba on a Hugging Face dataset (e.g., tatsu-lab/alpaca)")
    p.add_argument('--dataset_name', type=str, default='tatsu-lab/alpaca',
                   help='HF dataset name (e.g., tatsu-lab/alpaca)')
    p.add_argument('--out_dir', type=str, default='runs/mamba_alpaca', help='Directory to save outputs')
    # model
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layer', type=int, default=2)
    p.add_argument('--d_state', type=int, default=16)
    p.add_argument('--expand', type=int, default=2)
    p.add_argument('--dt_rank', type=int, default=32)
    p.add_argument('--d_conv', type=int, default=4)
    # training
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--log_interval', type=int, default=200)
    p.add_argument('--sample_chars', type=int, default=256, help='Chars to sample each log interval')
    p.add_argument('--sample_print_chars', type=int, default=400, help='Trim printed sample for readability')
    p.add_argument('--seed', type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)
