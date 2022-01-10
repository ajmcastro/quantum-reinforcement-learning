import time, math, cmath
import numpy as np

from functools import reduce

from qiskit import *
from qiskit.quantum_info import Statevector

from circuit_builder import CircuitBuilder

from agent import Agent

class QRPS_Agent(Agent):
    def __init__(self, backend):
        self.backend = backend
        self.memory = {}

        self.gamma = 0.0
        self.n = 0.05

    # def act2(self, env):
    #     transitions = np.array([
    #         [0.4, 0.2, 0.2, 0.2],
    #         [0.2, 0.4, 0.4, 0.4],
    #         [0.2, 0.2, 0.2, 0.2],
    #         [0.2, 0.2, 0.2, 0.2]
    #     ])
    #     flags = [1, 2]
    #     self.rank_two(transitions, flags)

    def act(self, env):
        env_state = env.state()
        env_actions = env.actions()

        # If only one action is available, there's nothing to learn here
        if len(env_actions) == 1:
            env.step(env_actions[0])
            return

        # Add to memory
        if env_state not in self.memory:
            self.memory[env_state] = np.array([1] * len(env_actions)), np.array(range(len(env_actions))), 0.0

        weights, flags, glow = self.memory[env_state]

        sum_weights = np.sum(weights)
        prob = np.array([h / sum_weights for h in weights])

        print('Pr:', prob)
        print('Flags:', flags)

        # Quantum deliberation
        max_tries = 3
        action_index = None

        for _ in range(max_tries):
            action_index = self.rank_one(prob, flags, debug=False)

            if action_index in flags:
                break

        reward = env.step(env_actions[action_index])

        print("Action:", env_actions[action_index], reward)

        self.update_values(env_state, env_actions, action_index, reward)
    
    def update_values(self, state, actions, action_index, reward):
        "Updates the weights, flags and glow values according to the received reward"

        weights, flags, glows = self.memory[state]

        glows = np.array([1.0 if action_index == i else (1 - self.n) * g for i, g in enumerate(glows)])
        weights[action_index] = weights[action_index] - self.gamma * (weights[action_index] - 1) + glows[action_index] * reward

        flags = np.delete(flags, action_index) if reward < 0.0 else flags
        if len(flags) == 0:
            flags = np.array([i for i in range(len(actions)) if i is not action_index])

        self.memory[state] = weights, flags, glows

    def prob_to_angles(self, prob, previous=1):
        "Calculates the angles to encode the given probabilities"

        def calc_angle(x):
            return 2 * math.acos(math.sqrt(x))
        
        if len(prob) == 2:
            return [calc_angle(prob[0] / previous)] if previous != 0 else [0]

        lhs, rhs = np.split(prob, 2)
        angles = np.array([calc_angle(np.sum(lhs) / previous)])
        angles = np.append(angles, self.prob_to_angles(lhs, previous=np.sum(lhs)))
        angles = np.append(angles, self.prob_to_angles(rhs, previous=np.sum(rhs)))

        return angles

    def rank_one(self, prob, flags, debug=False):
        "Rank-one implementation of Reflective Projective Simulation"

        num_qubits = math.ceil(math.log(len(prob), 2))

        # Ensure lenght of probabilities is 2**num_qubits
        if len(prob) != 2**num_qubits:
            prob = np.append(prob, [0] * (2**num_qubits - len(prob)))

        # Epsilon (probability of flagged actions)
        epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)

        # State preparation
        U = CircuitBuilder().get_U(num_qubits, self.prob_to_angles(prob)).to_instruction()

        # Quantum circuit
        qreg = QuantumRegister(num_qubits, name='q')
        circ = QuantumCircuit(qreg)

        # Encode stationary distribution
        circ.append(U, qreg)

        k = math.floor(math.pi / (4 * math.sqrt(epsilon)))

        for _ in range(k):
            # Reflection around the flagged actions
            circ.diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)], qreg)

            # Reflection around the stationary distribution
            circ.append(U.inverse(), qreg)
            circ.x(qreg)

            if num_qubits == 1:
                circ.z(qreg)
            else:
                circ.h(qreg[-1])
                circ.mcx(qreg[:-1], qreg[-1])
                circ.h(qreg[-1])
        
            circ.x(qreg)
            circ.append(U, qreg)

        if debug:
            print(circ.draw(fold=140))
            circ.snapshot('sv') 
        
        # Sample from stationary distribution
        circ.measure_all()

        result = execute(circ, backend=self.backend, shots=1).result()

        if debug:
            resulting_sv = result.data()['snapshots']['statevector']['sv'][0]
            print(Statevector(resulting_sv).probabilities_dict())

        counts = result.get_counts(circ)
        action_index = max(counts, key=counts.get)

        return int(action_index, 2)

    def rank_two(self, transitions, flags, debug=False):
        eigvals = np.linalg.eigvals(transitions)
        eigvals.sort()

        num_qubits = int(math.log2(len(transitions)))
        num_ancilla = math.ceil(math.log2(1 / math.sqrt(1 - abs(eigvals[-2])))) + 1

        # Stationary distribution
        S, U = np.linalg.eig(transitions)
        stat_distr = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stat_distr = stat_distr / np.sum(stat_distr)

        print(stat_distr)

        # Epsilon (probability of flagged actions)
        epsilon = reduce(lambda e, i: e + stat_distr[i], flags, 0.0)

        # Reverse transition matrix
        rev_transitions = transitions * np.array(stat_distr)
        rev_transitions = rev_transitions.transpose() / np.array(stat_distr)

        # Angles
        stat_angles = self.prob_to_angles(stat_distr)
        angles = np.concatenate([self.prob_to_angles(transitions[:,i]) for i in range(2**num_qubits)])
        rev_angles = np.concatenate([self.prob_to_angles(rev_transitions[:,i]) for i in range(2**num_qubits)])
    
        # Quantum circuit
        anc = AncillaRegister(num_ancilla, 'anc')
        qreg1 = QuantumRegister(num_qubits, 'reg1')
        qreg2 = QuantumRegister(num_qubits, 'reg2')
        creg = ClassicalRegister(num_qubits, 'creg')
        circ = QuantumCircuit(anc, qreg1, qreg2, creg)

        # Encode stationary distribution
        U = CircuitBuilder().get_U(num_qubits, stat_angles)
        circ.append(U.to_instruction(), qreg1)

        Up = CircuitBuilder().get_Up(num_qubits, angles)
        circ.append(Up.to_instruction(), qreg1[:] + qreg2[:])

        ARO = CircuitBuilder().get_ARO(num_qubits, num_ancilla)

        k = math.floor(math.pi / (4 * math.sqrt(epsilon)))

        for _ in range(k):
            circ.diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)], qreg1)
            circ.append(ARO, anc[:] + qreg1[:] + qreg2[:])

        print(circ.draw(fold=240))
        # circ.snapshot('sv')
        
        circ.measure(qreg1, creg)

        # Bind transition angles
        parameters = CircuitBuilder().get_parameters(num_qubits)
        binds = dict(zip(parameters, np.concatenate([angles, rev_angles])))

        start = time.time()

        result = execute(circ, backend=self.backend, shots=2048, parameter_binds=[binds]).result()

        end = time.time()

        if debug:
            resulting_sv = result.data()['snapshots']['statevector']['sv'][0]
            print(Statevector(resulting_sv).probabilities_dict())

        print("RUN took:", end - start)

        print(result.get_counts(circ))
