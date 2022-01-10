import math, pickle
from copy import deepcopy
from itertools import count
from functools import reduce

import numpy as np

# from qiskit import QuantumCircuit, QuantumRegister, execute
# # from qiskit.extensions import Initialize
# from qiskit.circuit.library import Diagonal, GroverOperator

#from tqdm.notebook import tqdm
from tqdm import tqdm

from circuit_builder import CircuitBuilder
from utils import prob_to_angles
from stats import EpisodeStats

class QRQLAgent:
    def __init__(self, i, alpha, gamma, R, exploration='6x6', mode='classical'):
        self.i = i
        # self.backend = backend
        self.alpha, self.gamma, self.R, self.exploration, self.mode = alpha, gamma, R, exploration, mode
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

        for i_episode in tqdm(range(num_episodes), position=self.i, desc=f'QRQL-{self.i}', disable=None):
            env.reset(i_episode + 1)

            total_rewards = 0.0

            state = env.state()
            actions = env.actions()

            if self.exploration == '6x6':
                tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.5 * (i_episode / 1000)))
            elif self.exploration == '8x8':
                tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.35 * (i_episode / 1000)))
            elif self.exploration == '6x6_changing':
                if i_episode < 20000:
                    tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.5*(i_episode / 1000)))
                else:
                    tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.5*((i_episode - 20000) / 1000)))
            elif self.exploration == '8x8_changing':
                if i_episode < 25000:
                    tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.35*(i_episode / 1000)))
                else:
                    tau = 0.2 + (20 - 0.2) / (1 + math.e**(0.35*((i_episode - 12500) / 1000)))

            for t in count():
                if state not in self.memory:
                    self.memory[state] = np.zeros(len(actions)), np.arange(len(actions))

                q_values, flags = self.memory[state]

                # Exploration policy using softmax
                shift = q_values - np.max(q_values)
                prob = np.exp((1 / tau) * shift) / np.sum(np.exp((1 / tau) * shift))

                # Quantum deliberation
                deliberation = self.classical_deliberation if self.mode == 'classical' else self.quantum_deliberation
                action = 0 if prob.size == 1 else deliberation(prob, flags)

                next_state, next_actions, reward = env.step(actions[action])
                reward = reward[0] + self.gamma * reward[2] - reward[1]

                # Update estimate
                if next_state in self.memory:
                    max_next_q = np.max(self.memory[next_state][0])
                else:
                    max_next_q = 0.0

                td_target = reward + self.gamma * max_next_q
                td_error = td_target - q_values[action]
                q_values[action] += self.alpha * td_error

                if q_values.size > 1:
                    if q_values[action] < 0.0:
                        flags = np.delete(flags, np.where(flags == action))
                    else:    
                        flags = np.append(flags, action) if action not in flags else flags

                    if flags.size == 0:
                        f = np.arange(q_values.size)
                        flags = np.delete(f, np.where(f == action))

                self.memory[state] = q_values, flags

                total_rewards += reward

                if env.is_over:
                    stats.episode_results[i_episode] = env.winner
                    stats.episode_steps[i_episode] = t + 1
                    stats.episode_rewards[i_episode] = total_rewards
                    stats.explored_states[i_episode] = len(self.memory.keys())
                    break
                
                state, actions = next_state, next_actions

        return stats

    def classical_deliberation(self, prob, flags):
        if self.R == 0:
            return np.random.choice(prob.size, p=prob)

        action = None

        for i_reflection in count():
            action = np.random.choice(prob.size, p=prob)

            if action in flags or i_reflection + 1 >= self.R:
                break

        return action

    def quantum_deliberation(self, prob, flags):
        if self.R == 0:
            return np.random.choice(prob.size, p=prob)

        epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)
        epsilon = 1.0 if epsilon >= 1.0 else epsilon
        theta = math.asin(math.sqrt(epsilon))

        k = math.ceil(1 / math.sqrt(epsilon))

        action = None

        for i_reflection in count():
            m = np.random.randint(0, k + 1)

            final_prob = np.array([
                (math.sqrt(p) / math.sqrt(epsilon)) * math.sin((2 * m + 1) * theta) if i in flags
                    else (math.sqrt(p) / math.sqrt(1 - epsilon)) * math.cos((2 * m + 1) * theta) for i,p in enumerate(prob)
            ])**2

            # Fix sum is not 1.0
            final_prob /= np.sum(final_prob)

            action = np.random.choice(final_prob.size, p=final_prob)

            if action in flags or i_reflection + 1 >= self.R:
                break

        return action

    # def quantum_deliberation(self, prob, flags):
    #     num_qubits = 1 if prob.size == 1 else math.ceil(math.log2(prob.size))

    #     if prob.size != 2**num_qubits:
    #         prob = np.append(prob, [0] * (2**num_qubits - prob.size))

    #     epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)
    #     epsilon = 1.0 if epsilon >= 1.0 else epsilon

    #     k = math.ceil(1 / math.sqrt(epsilon))

    #     U = CircuitBuilder(self.backend).get_U(num_qubits, prob_to_angles(prob))

    #     for i_reflection in count():
    #         qreg = QuantumRegister(num_qubits, name='q')
    #         circ = QuantumCircuit(qreg)

    #         circ.append(U.to_instruction(), qreg)

    #         m = np.random.randint(0, k)

    #         if m > 25:
    #             print("ERROR! Big number ", m)
    #             m = 25

    #         if m > 0:
    #             grover = GroverOperator(
    #                 oracle=Diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)]),
    #                 state_preparation=U
    #             ).repeat(m)

    #             circ.append(grover.to_instruction(), qreg)

    #         circ.measure_all()

    #         result = execute(circ, backend=self.backend, shots=1).result()
    #         counts = result.get_counts(circ)
    #         action = int(max(counts, key=counts.get), 2)

    #         if action in flags or i_reflection >= self.R:
    #             break

    #     return action
