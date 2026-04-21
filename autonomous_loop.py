from synthetic_data import generate_synthetic, model_in_the_loop_generate
from data_intelligence import clean_corpus

def autonomous_pipeline(raw, embed_fn=None, model=None, tokenizer=None):
    data=clean_corpus(raw)
    data+=generate_synthetic(200)

    if model:
        prompts=["Improve dataset"]*10
        data+=model_in_the_loop_generate(model,tokenizer,prompts)

    return data
