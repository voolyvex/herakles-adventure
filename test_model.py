import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_loading():
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
    
    # Test a simple generation
    print("\nTesting generation...")
    inputs = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    print("Generated:", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    test_model_loading()
