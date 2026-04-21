import random

def generate_synthetic(n=200):
    topics=["ML","systems","python","optimization"]
    return [f"Explain {random.choice(topics)} clearly." for _ in range(n)]

def model_in_the_loop_generate(model, tokenizer, prompts):
    return [p + " (stub output)" for p in prompts]
