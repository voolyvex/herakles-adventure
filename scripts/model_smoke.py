"""Manual smoke check that the causal LM loads and generates.

This is a script, not a test: it asserts nothing, and it downloads a
multi-gigabyte model. It lives under ``scripts/`` and its entry point is named
``main`` rather than ``test_*`` so that pytest cannot collect it — collecting it
would turn a CI run into a multi-gigabyte download that checks nothing.

Run it directly:

    python scripts/model_smoke.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Testing model loading...")
    model_name = "microsoft/phi-2"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )

    print("Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    # Exercise a simple generation
    print("\nTesting generation...")
    inputs = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Generated:", tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
