import argparse

import torch
from transformers import AutoTokenizer

from lib.mamba2 import MambaLM


def build_prompt(instruction: str, inp: str = "") -> str:
    fixed_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    if inp.strip():
        return (
            f"{fixed_prompt}\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"{fixed_prompt}\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Response:\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate text with a fine-tuned Mamba2 model")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Path to checkpoint file")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path")
    parser.add_argument(
        "--instruction", type=str,
        default="Tell me a story about a blacksmith who saves a village from a black dragon.",
        help="Instruction string"
    )
    parser.add_argument(
        "--input", type=str,
        default="",
        help="Optional input string"
    )
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]

    # Rebuild model and load weights
    model = MambaLM(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build prompt
    prompt = build_prompt(args.instruction, args.input)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # Strip the prompt part for readability
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    output = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
    print(f"{prompt}{output}", end="\n\n")


if __name__ == "__main__":
    main()
