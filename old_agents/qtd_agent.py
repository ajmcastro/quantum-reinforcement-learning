import math
import numpy as np
from itertools import count

import ipywidgets as widgets
from IPython.display import display

from qiskit import QuantumRegister, QuantumCircuit, execute
from qiskit.extensions import Initialize
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import GroverOperator

from stats import EpisodeStats

# from . import BaseAgent
# from circuit_builder import CircuitBuilder
from utils import prob_to_angles, R_amplify, R_deamplify

class QTDAgent:
    def __init__(self, backend, alpha, gamma, k):
        self.backend = backend
        
        self.alpha = alpha
        self.gamma = gamma
        self.k = k

        self.memory = dict()

    def train(self, env, num_episodes):
        
        learning_progress = widgets.IntProgress(
            min=0, max=num_episodes, step=1, 
            description='0', bar_style='success',
            orientation='horizontal',
            display='flex', flex_flow='column', align_items='stretch',
            layout=widgets.Layout(width='auto', height='auto')
        )

        display(learning_progress)

        stats = EpisodeStats(
            episode_results=np.empty(num_episodes, dtype=str),
            episode_steps=np.empty(num_episodes),
            episode_rewards=np.empty(num_episodes)
        )

        for i_episode in range(num_episodes):
            env.reset()

            total_rewards = 0.0

            state = env.state()
            actions = env.actions()

            for t in count():
                num_actions = len(actions)
                
                if state not in self.memory:
                    self.memory[state] = 0.0, np.array([1 / num_actions] * num_actions)

                if num_actions == 1:
                    action = 0
                else:
                    num_qubits = math.ceil(math.log(num_actions, 2))

                    amplitudes = np.sqrt(self.memory[state][1])

                    if len(amplitudes) != 2**num_qubits:
                        amplitudes = np.append(amplitudes, [0] * (2**num_qubits - len(amplitudes)))

                    U = Initialize(amplitudes).gates_to_uncompute().inverse().copy(name='A')

                    qreg = QuantumRegister(num_qubits)
                    circ = QuantumCircuit(qreg)

                    circ.append(U.to_instruction(), qreg)

                    # amplitudes = np.array([
                    #     1 / math.sqrt(num_actions) * complex(1, 0) if i < num_actions else 0 for i in range(2**num_qubits)
                    # ])

                    # U = Initialize(amplitudes).gates_to_uncompute().inverse().copy(name='A')

                    # qreg = QuantumRegister(num_qubits)
                    # circ = QuantumCircuit(qreg)

                    # circ.append(U.to_instruction(), qreg)

                    # if self.memory[state][1] is not None:
                    #     a, L = self.memory[state][1]
                        
                    #     angle = math.asin(math.sqrt(1 / num_actions))
                    #     max_L = math.floor(math.pi / (4 * angle) - 0.5)
                    #     L = L if L <= max_L else max_L

                    #     grover = GroverOperator(
                    #         oracle=Statevector.from_label(np.binary_repr(a, width=num_qubits)),
                    #         state_preparation=U
                    #     ).repeat(L)

                    #     circ.append(grover.to_instruction(), qreg)

                    circ.measure_all()

                    result = execute(circ, backend=self.backend, shots=1).result()
                    counts = result.get_counts(circ)
                    action = int(max(counts, key=counts.get), 2)

                next_state, next_actions, reward = env.step(actions[action])
                
                # Update state value estimate
                old_v, prob = self.memory[state]

                if next_state in self.memory:
                    next_v = self.memory[next_state][0]
                else:
                    next_v = 0.0

                new_v = old_v + self.alpha * (reward + self.gamma * next_v - old_v)

                L = self.k * (reward + next_v)
                max_L = math.floor(math.pi / (4 * math.asin(math.sqrt(prob[action]))))
                L = L if L <= max_L else max_L

                for i in range(int(L)):
                    amp_ratio = R_amplify(prob[action], math.pi, math.pi)
                    deamp_ratio = R_deamplify(prob[action], math.pi, math.pi)
                    prob = np.array([p * amp_ratio if i == action else p * deamp_ratio for i,p in enumerate(prob)])

                self.memory[state] = new_v, prob

                # self.memory[state] = new_v, (action, int(self.k * (reward + next_v)))

                total_rewards += reward

                if env.is_over:
                    stats.episode_results[i_episode] = env.winner
                    stats.episode_steps[i_episode] = t + 1
                    stats.episode_rewards[i_episode] = total_rewards
                    break

                state, actions = next_state, next_actions

            learning_progress.description = str(i_episode + 1)
            learning_progress.value += 1
        
        return stats

    def compute_angles(self, prob, reward):
        estimations = {
            0.1: ([5.17294298, 1.11024233], [3.14159265, 3.14159265], [4.6010479674466795, 1.6821373425388433]),
            0.2: ([5.09678577, 1.18639955], [3.14159265, 3.14159265], [4.459708731543855, 1.823476588238218]),
            0.3: ([5.00214072, 1.28104463], [2.30052385, 2.30052382], [3.973121599843598, 1.669354158691014]),
            0.4: ([4.87983706, 1.40334824], [1.82347659, 1.82347658], [3.542721084843427, 1.5871486685387224]),
            0.5: ([4.71238898, 1.57079633], [1.57079631, 1.57079632], [3.1415926535897936, 1.5707963234699533]),
            0.6: ([4.45970872, 1.82347658], [1.40334827, 1.40334823], [2.7404642285137673, 1.5871486527193748]),
            0.7: ([3.98266133, 2.30052398], [1.28104463, 1.28104454], [2.3100636806158654, 1.6693541946745145]),
            0.8: ([3.14159267, 3.14159265], [1.18639965, 1.18639955], [1.8234765877090218, 1.823476576164929]),
            0.9: ([3.14159268, 3.14159265], [1.11024236, 1.11024248], [1.6821373510225015, 1.6821373312492471])
        }

        max_p, min_p, b = None, None, None

        for p, v in estimations.items():
            if prob <= p:
                max_p, min_p, b = v
                break

        if max_p is None and min_p is None and b is None:
            return 0.0, 0.0

        # Slope & magnitude
        slope = (max_p[1] - min_p[1]) / (max_p[0] - min_p[0])
        angle = np.arctan(slope)
        magnitude = math.sqrt((max_p[0] - min_p[0])**2 + (max_p[1] - min_p[1])**2)

        # Ascending or descending, depending if min->max is left->right or right->left
        asc = 1 if max_p[0] > min_p[0] else -1

        distance_min = math.sqrt((b[0] - min_p[0])**2 + (b[1] - min_p[1])**2)
        distance_max = math.sqrt((b[0] - max_p[0])**2 + (b[1] - max_p[1])**2)

        lower_bound = -distance_min
        upper_bound = distance_max

        def reward_to_magnitude(reward):
            Q = - upper_bound / lower_bound
            growth_rate = 0.35
            return lower_bound + (upper_bound - lower_bound) / (1 + Q * math.e**(-growth_rate * reward))

        theta1 = asc * reward_to_magnitude(reward) * math.cos(angle) + b[0]
        theta2 = asc * reward_to_magnitude(reward) * math.sin(angle) + b[1]

        return theta1, theta2





# class QTDAgent(BaseAgent):
#     def __init__(self, backend, alpha, gamma, k):
#         super().__init__()

#         self.backend = backend

#         self.alpha = alpha
#         self.gamma = gamma
#         self.k = k

#         self.memory = dict()

#     def add_to_memory(self, state, actions):
#         num_actions = len(actions)

#         if state not in self.memory:
#             self.memory[state] = np.random.uniform(0, 1), np.array([1 / num_actions for i in range(num_actions)])

#     def select_action(self, state, actions):
#         self.add_to_memory(state, actions)

#         _, prob = self.memory[state]

#         num_qubits = 1 if len(prob) == 1 else math.ceil(math.log2(len(prob)))

#         if len(prob) != 2**num_qubits:
#             prob = np.append(prob, [0.0] * (2**num_qubits - len(prob)))
        
#         qreg = QuantumRegister(num_qubits)
#         circ = QuantumCircuit(qreg)
        
#         U = CircuitBuilder(self.backend).get_U(num_qubits, prob_to_angles(prob)).to_instruction()

#         circ.append(U, qreg)
#         circ.measure_all()

#         result = execute(circ, backend=self.backend, shots=1).result()
#         counts = result.get_counts(circ)
#         action = max(counts, key=counts.get)

#         return int(action, 2)

#     def learn(self, state, action, next_state, reward, terminal=False):
#         old_v, prob = self.memory[state]

#         new_v = old_v

#         if terminal:
#             new_v += self.alpha * reward
#         else:
#             next_v, _ = self.memory[next_state]
#             new_v += self.alpha * (reward + self.gamma * next_v - old_v)

#         if prob is not None:
#             epsilon = prob[action]
#             R_amp = R_amplify(epsilon, math.pi, math.pi)
#             R_deamp = R_deamplify(epsilon, math.pi, math.pi)

#             L = self.k * reward if terminal else self.k * (reward + self.memory[next_state][0])

#             angle = math.asin(math.sqrt(epsilon))
#             max_L = math.floor(math.pi / (4 * angle) - 0.5)

#             if L < 0.0:
#                 L = 0.0
#             elif L >= max_L:
#                 L = max_L

#             for i in range(int(L)):
#                 epsilon = prob[action]
#                 R_amp = R_amplify(epsilon, math.pi, math.pi)
#                 R_deamp = R_deamplify(epsilon, math.pi, math.pi)
#                 prob = np.array([round(p * R_amp, 6) if i == action else round(p * R_deamp, 6) for i,p in enumerate(prob)])
            
#         self.memory[state] = new_v, prob

#     def best_action(self, state, actions):
#         if state in self.memory:
#             return np.argmax(self.memory[state])
#         else:
#             return np.random.choice(range(len(actions)))