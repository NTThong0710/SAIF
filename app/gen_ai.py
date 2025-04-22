from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_response(prompt: str):
    result = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]
