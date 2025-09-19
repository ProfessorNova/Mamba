import argparse

import torch

from lib.byte_tokenizer import ByteTokenizer
from lib.mamba2 import MambaLM


def main():
    parser = argparse.ArgumentParser(description="Generate text with a Mamba2 model")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Path to checkpoint file")
    parser.add_argument(
        "--input", type=str,
        default="Once upon a time, in a nice little town, there lived a big dragon.",
        help="Input text"
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]

    # Rebuild model and load weights
    model = MambaLM(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = ByteTokenizer(add_bos=False, add_eos=False)

    # Build prompt
    prompt = args.input
    input_ids = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long
    ).unsqueeze(0).to(device)

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
