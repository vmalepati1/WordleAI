import dqn.state
from dqn.state import WordleState
import numpy as np

REWARD = 10

class WordleEnv:

    def __init__(self, words, max_turns, allowable_words = None):
        self.words = words
        self.max_turns = max_turns
        self.allowable_words = allowable_words
        self.observation_space_size = len(dqn.state.new(self.max_turns))
        self.action_space_size = len(self.words)

        if not self.allowable_words:
            self.allowable_words = len(self.words)

        self.done = True
        self.goal_word_idx = -1
        self.state = None

    def reset(self):
        self.state = dqn.state.new(self.max_turns)
        self.done = False
        self.goal_word_idx = int(np.random.random()*self.allowable_words)

        return self.state.copy()

    def set_goal_word(self, goal_word: str):
        self.goal_word_idx = self.words.index(goal_word)

    def set_goal_id(self, goal_id: int):
        self.goal_word_idx = goal_id

    def step(self, action: int):
        if self.done:
            raise ValueError(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior."
            )
        self.state = dqn.state.update(state=self.state,
                                        word=self.words[action],
                                        goal_word=self.words[self.goal_word_idx])

        reward = 0
        if action == self.goal_word_idx:
            self.done = True
            reward = REWARD
##            if dqn.state.remaining_steps(self.state) == self.max_turns-1:
##                reward = 0#-10*REWARD  # No reward for guessing off the bat
##            else:
##                #reward = REWARD*(self.state.remaining_steps() + 1) / self.max_turns
##                reward = REWARD
        elif dqn.state.remaining_steps(self.state) == 0:
            self.done = True
            reward = -REWARD

        return self.state.copy(), reward, self.done
    
