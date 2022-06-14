import csv
import pandas as pd
from tqdm import tqdm

def get_word_table(gray_letters, yellow_letters, green_letters, subset=[]):
    if len(subset) == 0:
        csv_file = open('dataset.csv')
        subset = csv.reader(csv_file, delimiter=',')

    line_count = 0

    table_result = []

    table_result.append(['word', 'freq'])
        
    for row in tqdm(subset):
        if line_count >= 1:
            word = row[0]
            freq = row[1] # Higher indicates more frequent

            # Innocent till proven guilty
            viable_word = True

            for letter in gray_letters:
                if letter in word:
                    viable_word = False
                    break

            if viable_word is True:
                for letter, idx in yellow_letters.items():
                    if letter not in word:
                        viable_word = False
                        break

                    indices = [pos for pos, char in enumerate(word) if char == letter]

                    if idx in indices:
                        viable_word = False
                        break

            if viable_word is True:
                for letter, idx in green_letters.items():
                    if letter not in word:
                        viable_word = False
                        break

                    indices = [pos for pos, char in enumerate(word) if char == letter]

                    if idx not in indices:
                        viable_word = False
                        break

            if viable_word is True:
                table_result.append([word, freq])

        line_count += 1

    return table_result

gray_letters = ['s', 'l']
yellow_letters = {'a': 2, 't': 3}
green_letters = {'e': 4}

subset = get_word_table(gray_letters, yellow_letters, green_letters)
subset = get_word_table(gray_letters, yellow_letters, green_letters, subset=subset)
