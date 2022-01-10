import math, pickle
import numpy as np
from itertools import count

from tqdm.notebook import tqdm

from stats import EpisodeStats

class QLearning:
    def __init__(self, i, alpha, gamma):
        self.i = i
        self.alpha, self.gamma = alpha, gamma
        self.memory = dict()
    
    def save(self, model):
        with open(model + '.pkl', 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def load(self, model):
        with open(model + '.pkl', 'rb') as f:
            self.memory = pickle.load(f)
            f.close()

    def train(self, env, num_episodes):

        stats = EpisodeStats(
            episode_results=np.empty(num_episodes, dtype=str),
            episode_steps=np.empty(num_episodes, dtype=int),
            episode_rewards=np.empty(num_episodes, dtype=float),
            explored_states=np.empty(num_episodes, dtype=int)
        )

        for i_episode in tqdm(range(num_episodes), position=self.i, desc=f'QL-{self.i}'):
            env.reset(i_episode + 1)

            total_rewards = 0.0

            state = env.state()
            actions = env.actions()

            # tau = 0.5 + (20 - 0.5) / (1 + math.e**(0.5*(i_episode / 1000)))
            # tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.5*(i_episode / 1000)))
            # tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.35*(i_episode / 1000)))
            # tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.5*(i_episode / 1000)))
            tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.07 *(i_episode / 1000)))

            for t in count():
                if state not in self.memory:
                    self.memory[state] = np.zeros(len(actions))

                q_values = self.memory[state]

                # Exploration policy using softmax
                shift = q_values - np.max(q_values)
                prob = np.exp((1 / tau) * shift) / np.sum(np.exp((1 / tau) * shift))

                action = np.random.choice(len(actions), p=prob)

                next_state, next_actions, reward = env.step(actions[action])
                reward = reward[0] + self.gamma * reward[2] - reward[1]

                # Update estimate
                if next_state in self.memory:
                    max_next_q = np.max(self.memory[next_state])
                else:
                    max_next_q = 0.0

                self.memory[state][action] = q_values[action] + self.alpha * (reward + self.gamma * max_next_q - q_values[action])

                total_rewards += reward

                if env.is_over:
                    stats.episode_results[i_episode] = env.winner
                    stats.episode_steps[i_episode] = t + 1
                    stats.episode_rewards[i_episode] = total_rewards
                    stats.explored_states[i_episode] = len(self.memory.keys())
                    break

                state, actions = next_state, next_actions

        return stats
