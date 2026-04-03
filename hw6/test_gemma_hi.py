import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--prompt", default="hi")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_devices)

    print(f"Loading model: {args.model_name}", flush=True)
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    try:
        import torch._dynamo as dynamo
        dynamo.disable()
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=text,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=False).strip()

    print("=== Prompt ===", flush=True)
    print(args.prompt, flush=True)
    print("=== Output ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
