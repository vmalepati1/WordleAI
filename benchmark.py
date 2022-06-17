import csv
from algorithmic import get_word_table

def evaluate_word(word, actual_word, gray_letters, yellow_letters, green_letters):
    for idx, letter in enumerate(word):
        if letter not in actual_word:
            gray_letters[letter] = idx
        else:
            indices = [pos for pos, char in enumerate(actual_word) if char == letter]

            if idx in indices:
                green_letters[letter] = idx
            else:
                # Letter is in the word but wrong position (yellow)
                # Letter is in the word but duplicate (gray)

                if letter in green_letters:
                    gray_letters[letter] = idx
                else:
                    yellow_letters[letter] = idx

starting_word = 'slate'

##gray_l, yl, green_l = evaluate_word('PRIOR', 'PRIMO')
##
##print(gray_l)
##print(yl)
##print(green_l)

num_tries = []
successful_trials = 0
unsuccessful_trials = 0

with open('solution_words.txt') as words:
    for word in words:
        green_letters = {}
        yellow_letters = {}
        gray_letters = {}

        subset = []

        solution_word = word.strip()

        word_to_guess = starting_word
        
        for i in range(6):
            evaluate_word(word_to_guess, solution_word, gray_letters, yellow_letters, green_letters)
            
            if solution_word == word_to_guess:
                # print(solution_word == word_to_guess)
                num_tries.append(i + 1)
                successful_trials += 1
                break
            elif i == 5:
                unsuccessful_trials += 1

            subset = get_word_table(gray_letters, yellow_letters, green_letters, subset=subset, verbose=False)

            sorted_subset = sorted(subset[1:], key=lambda x: x[1], reverse=True)

            word_to_guess = sorted_subset[0][0]

avg_tries = sum(num_tries) / len(num_tries)

print(successful_trials)
print(unsuccessful_trials)
success_rate = successful_trials / (successful_trials + unsuccessful_trials)

print('Average steps per game: ' + str(avg_tries))
print('Success rate: ' + str(success_rate))


        

        

        

