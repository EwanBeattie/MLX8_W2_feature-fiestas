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
    # Convert to lowercase
    text = text.lower()

    # Replace all punctuation with angle bracket words
    for punct, replacement in PUNCTUATION_MAP.items():
        text = text.replace(punct, f" {replacement} ")

    # Split into words (handles multiple spaces)
    words = text.split()
    return words