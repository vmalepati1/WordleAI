import csv
from algorithmic import get_word_table

def evaluate_word(word, actual_word):
    gray_letters = {}
    yellow_letters = {}
    green_letters = {}

    for idx, letter in enumerate(word):
        if letter not in actual_word:
            gray_letters[letter] = idx
        else:
            indices = [pos for pos, char in enumerate(actual_word) if char == letter]

            if idx in indices:
                green_letters[letter] = idx
            else:
                # Letter is in the word but wrong position (yellow)
                # Letter is in the word but duplicate

    return gray_letters, yellow_letters, green_letters

starting_word = 'slate'

with open('solution_words.txt') as words:
    for word in words:
        stripped_word = word.strip()

        

        

