import requests
import time
import csv

_wait = 0.5

def get_freq(term):
    response = None
    while True:
        try:
            response = requests.get('https://api.datamuse.com/words?sp='+term+'&md=f&max=1').json()
        except:
            print('Could not get response. Sleep and retry...')
            time.sleep(_wait)
            continue
        break;
    freq = 0.0 if len(response)==0 else float(response[0]['tags'][0][2:])
    return freq

with open('dataset.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow(['word', 'freq_rank'])
    
    with open('solution_words.txt') as words:
        for word in words:
            stripped_word = word.strip()
            
            freq = get_freq(stripped_word)
            
            writer.writerow([stripped_word, freq])


