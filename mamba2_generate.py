import argparse

import torch

from lib.gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from lib.mamba2 import MambaLM


def main():
    parser = argparse.ArgumentParser(description="Generate text with a Mamba2 model")
    parser.add_argument("--ckpt", type=str, default="last.pt", help="Path to checkpoint file")
    parser.add_argument(
        "--input", type=str,
        default="Once upon a time, in a nice little town, there lived a big dragon.",
        help="Input text"
    )
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]

    # Rebuild model and load weights
    model = MambaLM(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_state=config["d_state"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = GPT2TokenizerWrapper(add_eos=False)

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
            top_p=args.top_p,
            eos_id=tokenizer.eos_token_id,
        )

    # Strip the prompt part for readability
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    output = tokenizer.decode(new_tokens.tolist())
    print(f"{prompt}{output}", end="\n\n")


if __name__ == "__main__":
    main()
