import os

VALID_WORDS_PATH = 'words.txt'

def _load_words(limit=None):
    with open(VALID_WORDS_PATH, 'r') as f:
        lines = [x.strip().upper() for x in f.readlines()]
        if not limit:
            return lines
        else:
            return lines[:limit]
