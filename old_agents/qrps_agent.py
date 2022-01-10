import math
import numpy as np
from itertools import count
from functools import reduce

import ipywidgets as widgets
from IPython.display import display

from qiskit import QuantumRegister, QuantumCircuit, execute
from qiskit.circuit.library import Diagonal, GroverOperator
from qiskit.extensions import Initialize

from stats import EpisodeStats

class QRPSAgent:
    def __init__(self, backend, beta, gamma, n):
        self.backend = backend
        self.beta, self.gamma, self.n = beta, gamma, n
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
                if state not in self.memory:
                    self.memory[state] = np.zeros(len(actions)), np.arange(len(actions)), np.zeros(len(actions))

                weights, flags, glows = self.memory[state]

                shift = weights - np.max(weights)
                prob = np.exp(self.beta * shift) / np.sum(np.exp(self.beta * shift))
                
                max_tries = 3
                action = None

                for _ in range(max_tries):
                    action = self.quantum_deliberation(prob, flags)

                    if action in flags:
                        break

                next_state, next_actions, reward = env.step(actions[action])

                # Update estimate
                glows = np.array([1 if i == action else (1 - self.n) * g for i,g in enumerate(glows)])
                weights = np.array([w - self.gamma * (w - 1) + glows[i] * reward for i,w in enumerate(weights)])

                if len(weights) > 1:
                    flags = np.delete(flags, np.where(flags == action)) if reward < 0.0 else flags
                    if flags.size == 0:
                        flags = np.delete(np.arange(len(weights)), np.where(flags == action))

                self.memory[state] = weights, flags, glows

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

    def quantum_deliberation(self, prob, flags):
        if len(prob) == 1:
            return 0

        num_qubits = 1 if len(prob) == 1 else math.ceil(math.log(len(prob), 2))

        if len(prob) != 2**num_qubits:
            prob = np.append(prob, [0] * (2**num_qubits - len(prob)))

        epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)

        if math.isclose(epsilon, 0.0, abs_tol=0.001):
            epsilon = 0.0
        elif epsilon > 1.0:
            epsilon = 1.0

        U = Initialize(np.sqrt(prob) * complex(1, 0)).gates_to_uncompute().inverse().copy(name='A')

        qreg = QuantumRegister(num_qubits, name='q')
        circ = QuantumCircuit(qreg)

        circ.append(U.to_instruction(), qreg)

        k = 0

        try:
            k = math.floor(math.pi / (4 * math.asin(math.sqrt(epsilon))))
        except:
            k = 0

        if k > 10:
            k = 10

        grover = GroverOperator(
            oracle=Diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)]),
            state_preparation=U
        ).repeat(k)

        circ.append(grover.to_instruction(), qreg)

        circ.measure_all()

        result = execute(circ, backend=self.backend, shots=1).result()
        counts = result.get_counts(circ)
        action_index = max(counts, key=counts.get)

        return int(action_index, 2)




    # def deliberate(self, prob, flags):
    #     num_qubits = 1 if len(prob) == 1 else math.ceil(math.log(len(prob), 2))

    #     # Ensure lenght of probabilities is 2**num_qubits
    #     if len(prob) != 2**num_qubits:
    #         prob = np.append(prob, [0] * (2**num_qubits - len(prob)))

    #     epsilon = reduce(lambda e, i: e + abs(prob[i]), flags, 0.0)

    #     # State preparation
    #     U = Initialize([math.sqrt(p) for p in prob]).gates_to_uncompute().inverse().copy(name='A').to_instruction()

    #     # Quantum circuit
    #     qreg = QuantumRegister(num_qubits, name='q')
    #     circ = QuantumCircuit(qreg)

    #     # Encode stationary distribution
    #     circ.append(U, qreg)

    #     #k = np.random.rand(math.floor(math.pi / (4 * math.sqrt(epsilon))))
    #     # r = math.ceil(1 / math.sqrt(epsilon))
    #     # k = random.randint(0, r)
    #     k = math.floor(math.pi / (4 * math.asin(math.sqrt(epsilon))) - 0.5)
    #     # k = math.floor(math.pi / (4 * math.sqrt(epsilon)))

    #     print("Epsilon ", epsilon, " -> applying it ", k, " times")

    #     for _ in range(k):
    #         # Reflection around the flagged actions
    #         circ.diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)], qreg)

    #         # Reflection around the stationary distribution
    #         circ.append(U.inverse(), qreg)
    #         circ.x(qreg)

    #         if num_qubits == 1:
    #             circ.z(qreg)
    #         else:
    #             circ.h(qreg[-1])
    #             circ.mcx(qreg[:-1], qreg[-1])
    #             circ.h(qreg[-1])
        
    #         circ.x(qreg)
    #         circ.append(U, qreg)

    #     # Sample from stationary distribution
    #     circ.measure_all()

    #     result = execute(circ, backend=self.backend, shots=1).result()

    #     counts = result.get_counts(circ)
    #     action_index = max(counts, key=counts.get)

    #     return int(action_index, 2)

    # def learn(self, state, action, reward):
    #     # Normalize reward
    #     lower_bound, upper_bound = -1, 1
    #     growth_rate = 0.5
    #     reward = lower_bound + (upper_bound - lower_bound) / (1 + math.e**(- growth_rate * reward))

    #     weights, flags, glows = self.memory[state]

    #     glows = np.array([1 if i == action else (1 - self.n) * g for i,g in enumerate(glows)])
    #     weights = np.array([w - self.gamma * (w - 1) + glows[i] * reward for i,w in enumerate(weights)])
    #     weights = np.array([10.0 if w > 10.0 else w for w in weights])

    #     if len(weights) > 1:
    #         flags = np.delete(flags, np.where(flags == action)) if reward < 0.0 else flags
    #         if flags.size == 0:
    #             flags = np.delete(np.arange(len(weights)), np.where(flags == action))

    #     self.memory[state] = weights, flags, glows

