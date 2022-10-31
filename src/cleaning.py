import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def alphanum(element):
    """Getting rid of Chinese/Japanese characters and other weird stuff"""
    if re.search(r"[^a-zA-Z0-9\s.,\/#!?$%\^&\*;:{}=\-_`~()@+\'\"<>\[\]\\]", element):
        return None
    else:
        return element


def replace_linebreaks(element):
    """Substitute linebreaks with spaces"""
    return re.sub(r"\\n", " ", element).strip()


def filter_long_commits(element):
    """Getting rid of commits that are too long"""
    if element and len(element) > 512:
        return None
    return element


def remove_emojis(element):
    """Substitute emojis with <EMOJI>"""
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
    """Substitute commit hashes and other hashes with <HASH>"""
    return re.sub(r"[0-9a-f]{40}", "<HASH>", element)


def clean_text(element):
    """Applying functions mentioned above"""
    string = replace_linebreaks(str(element))
    string = remove_emojis(string)
    string = replace_hash(string)
    string = alphanum(string)
    string = filter_long_commits(string)
    if string:
        return " ".join(word_tokenize(string))
    return None
