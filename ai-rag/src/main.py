import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    model_name = "gpt2"  # Example model name, replace with your desired model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Example usage
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Model and tokenizer loaded successfully.")