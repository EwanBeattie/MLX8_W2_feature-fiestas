from collections import Counter
import random
import logging

logger = logging.getLogger(__name__)

UNK_TOKEN = "<UNK>"
PUNCTUATION_MAP = {
    "<": "<LESS>",
    ">": "<GREATER>",
    ",": "<COMMA>",
    ".": "<PERIOD>",
    "!": "<EXCLAMATION>",
    "?": "<QUESTION>",
    ":": "<COLON>",
    ";": "<SEMICOLON>",
    "-": "<DASH>",
    "(": "<LPAREN>",
    ")": "<RPAREN>",
    "[": "<LBRACKET>",
    "]": "<RBRACKET>",
    "{": "<LBRACE>",
    "}": "<RBRACE>",
    '"': "<QUOTE>",
    "'": "<APOSTROPHE>",
    "/": "<SLASH>",
    "\\": "<BACKSLASH>",
    "&": "<AMPERSAND>",
    "@": "<AT>",
    "#": "<HASH>",
    "$": "<DOLLAR>",
    "%": "<PERCENT>",
    "*": "<ASTERISK>",
    "+": "<PLUS>",
    "=": "<EQUALS>",
    "|": "<PIPE>",
    "~": "<TILDE>",
    "`": "<BACKTICK>",
}


def tokenise(text: str) -> list[str]:
    """
    Tokenises a long string of text by lowercasing, replacing punctuation with predefined angle bracket words.

    Args:
        text (str): A single string.

    Returns:
        dict: A dictionary mapping each word to a unique index.
    """
    # Convert to lowercase
    text = text.lower()

    # Replace all punctuation with angle bracket words
    for punct, replacement in PUNCTUATION_MAP.items():
        text = text.replace(punct, f" {replacement} ")

    # Split into words (handles multiple spaces)
    words = text.split()
    return words


def build_vocab(
    tokens: list[str],
) -> dict[str, int]:
    """
    Builds a vocabulary of unique words.
    """
    word_counts = Counter(tokens)
    # Remove words with frequency below threshold
    token_list = [UNK_TOKEN] + [
        word for word, count in word_counts.items()
    ]

    vocab = {word: idx for idx, word in enumerate(token_list)}

    return vocab


def get_tokens_as_indices(tokens: list[str], vocab: dict) -> list[int]:
    """
    Converts a list of tokens to their corresponding indices using the provided vocab mapping.
    This is to ensure we have fast, random-access, constant-sized, GPU-friendly data upfront.
    """
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]


def get_words_from_indices(indeces: list[int], vocab: dict) -> list[str]:
    """
    Converts a list of token indeces to a list of token values
    """
    return [
        list(vocab.keys())[list(vocab.values()).index(idx)]
        for idx in indeces
        if idx in vocab.values()
    ]
