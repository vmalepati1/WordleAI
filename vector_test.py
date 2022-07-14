import numpy as np

# See https://api.wandb.ai/files/andrewkho/images/projects/687198/34bab9b2.png

# 5 chars, 4 bytes per char
y = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])

words = np.asarray([[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]])

words = np.transpose(words)

print(words)

print(np.argmax(np.dot(y, words)))
