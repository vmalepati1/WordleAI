import requests
import time

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

with open('solution_words.txt') as words:
    for word in words:
        print(word.strip())
