import re


def alphanum(element):
    if re.search(r"[^a-zA-Z0-9\s.,\/#!?$%\^&\*;:{}=\-_`~()@+\'\"<>\[\]\\]", element):
        return None
    else:
        return element


def replace_linebreaks(element):
    return re.sub(r"\\n", " ", element).strip()


def remove_emojis(element):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("<EMOJI>", element)


def replace_hash(element):
    return re.sub(r"[0-9a-f]{40}", "<HASH>", element)
