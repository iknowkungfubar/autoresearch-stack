def repair(text):
    if not text:
        return None
    text = text.replace("\x00", "")
    text = " ".join(text.split())
    if len(text) < 20:
        return None
    return text


def is_noise(text):
    if len(set(text)) < 5:
        return True
    if sum(c.isalpha() for c in text) < 10:
        return True
    return False


def clean_corpus(texts):
    out = []
    for t in texts:
        t = repair(t)
        if t and not is_noise(t):
            out.append(t)
    return out
