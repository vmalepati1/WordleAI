import keras
from load_words import _load_words
from dqn.wordle_env import WordleEnv
import numpy as np

model = keras.models.load_model('models/2x256____10.00max___10.00avg___10.00min__1657823083.model')

words = _load_words(10)

env = WordleEnv(words, 6)

print('Vocab:')
print(words)

print('Choose a word:')
obj_word = input().upper()

if obj_word not in words:
    raise Exception('Word not found in vocab')

word_width = 26*5

word_array = np.zeros((word_width, len(words)))
for i, word in enumerate(words):
    for j, c in enumerate(word):
        word_array[j*26 + (ord(c) - ord('A')), i] = 1

current_state = env.reset()
env.set_goal_word(obj_word)

done = False
success = False

while not done:
    qs = model.predict(np.array(current_state).reshape(-1, *current_state.shape))[0]
    q_table = np.asarray(qs)
    argmax_layer = np.dot(q_table, word_array)
    action = np.argmax(argmax_layer)

    print(env.words[action])

    new_state, reward, done = env.step(action)

    if action == env.goal_word_idx:
        success = True

    current_state = new_state

if success is True:
    print('Success!')
else:
    print('Failure!')

    
