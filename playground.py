# from qiskit import *
# from qiskit.circuit import Parameter, ParameterVector
#from qiskit.quantum_info import Statevector, DensityMatrix, Operator
# from qiskit.extensions import Initialize
#from qiskit.circuit.library import GroverOperator, CPhaseGate, CU1Gate, Diagonal, MCPhaseGate, XGate, MCXGate, U3Gate

import sys
import math, cmath

import numpy as np
# np.set_printoptions(precision=6, threshold=sys.maxsize, suppress=True)

from scipy.optimize import root, minimize, basinhopping

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter, FuncFormatter

def R_a(prob_a, theta1, theta2):
    return (1 - math.e**complex(0, theta1) - math.e**complex(0, theta2)) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * prob_a

def amplify_probability(prob, epsilon, theta1, theta2):
    R = (1 - math.e**complex(0, theta1) - math.e**complex(0, theta2)) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * epsilon
    return prob * abs(R)**2

def deamplify_probability(prob, epsilon, theta1, theta2):
    R = -math.e**complex(0, theta2) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * epsilon
    return prob * abs(R)**2

def update_probability(prob, theta1, theta2):
    prob_a, prob_b = prob

    # f = (1 - math.e**complex(0, theta2)) * (math.e**complex(0, theta1) * prob_a + 1 - prob_a)
    # R1 = f - math.e**complex(0, theta1)
    # R2 = f - 1

    R1 = (1 - math.e**complex(0, theta1) - math.e**complex(0, theta2)) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * prob_a
    R2 = -math.e**complex(0, theta2) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * prob_a

    return prob_a * abs(R1)**2, prob_b * abs(R2)**2

def test1():

    # psi = np.array([
    #     [a],
    #     [b]
    # ])

    # q2q1 = np.array([
    #     [math.e**complex(0, theta1) * ((1 - math.e**complex(0, theta2)) * abs(a)**2 - 1), (1 - math.e**complex(0, theta2)) * a * b],
    #     [math.e**complex(0, theta1) * ((1 - math.e**complex(0, theta2)) * b * a), (1 - math.e**complex(0, theta2)) * abs(b)**2 - 1]
    # ])

    # res1 = np.dot(q2q1, psi)

    # print(abs(res1[0][0])**2, abs(res1[1][0])**2)
    # print(abs(new_a)**2, abs(new_b)**2)

    theta1 = math.pi
    theta2 = math.pi

    prob_a = 0.25
    prob_b = 1 - prob_a

    a = complex(math.sqrt(prob_a), 0)
    b = complex(math.sqrt(prob_b), 0)

    new_a = a * ((1 - math.e**complex(0, theta2)) * (math.e**complex(0, theta1) * abs(a)**2 + 1 - abs(a)**2) - math.e**complex(0, theta1))
    new_b = b * ((1 - math.e**complex(0, theta2)) * (math.e**complex(0, theta1) * abs(a)**2 + 1 - abs(a)**2) - 1)
    
    R = (1 - math.e**complex(0, theta1) - math.e**complex(0, theta2)) - (1 - math.e**complex(0, theta1)) * (1 - math.e**complex(0, theta2)) * abs(a)**2

    # print('AMPLITUDE:', abs(a), abs(b), '->', abs(new_a), abs(new_b))
    # print('PROBABILITY:', abs(a)**2, abs(b)**2, '->', abs(new_a)**2, abs(new_b)**2)
    # print('RATIO:', abs(new_a) / abs(a), '->', abs(new_a)**2 / abs(a)**2)
    # print("R & R^2:", abs(R), '->', abs(R)**2)

    # print(amplify_probability(prob_a, theta1, theta2))
    # print(deamplify_probability(prob_b, theta1, theta2))

    # new_prob = update_probability((prob_a, prob_b), theta1, theta2) 

    # print("TEST:", new_prob)
    # print(cmath.phase(new_prob[0]), cmath.phase(new_prob[1]))

def test2():
    prob_a = 0.4
    prob_b = 1 - prob_a

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    theta1_values = np.linspace(0, 2 * math.pi, num=100)
    theta2_values = np.linspace(0, 2 * math.pi, num=100)

    X, Y = np.meshgrid(theta1_values, theta2_values)
    Z = np.array([[amplify_probability(prob_a, t1, t2) for t2 in theta2_values] for t1 in theta1_values])
    # Z = np.array([[abs(R_a(prob_a, t1, t2))**2 for t2 in theta2_values] for t1 in theta1_values])

    max_i = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    min_i = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

    max_p = X[max_i[0]][max_i[1]], Y[max_i[0]][max_i[1]]
    min_p = X[min_i[0]][min_i[1]], Y[min_i[0]][min_i[1]]
    
    print("MAX:", max_p, '=>', Z[max_i[0]][max_i[1]])
    print("MIN:", min_p, '=>', Z[min_i[0]][min_i[1]])

    slope = (max_p[1] - min_p[1]) / (max_p[0] - min_p[0])
    angle = np.arctan(slope)

    print("Angle:", angle)

    v = max_p[0] - min_p[0], max_p[1] - min_p[1]
    mag = math.sqrt(v[0]**2 + v[1]**2)

    print("Vector:", v, mag)





    # n = np.arctan(slope) / np.pi

    # print(np.arctan(-0.1458333333333334) / np.pi)

    # print(max_t1, max_t2)
    # print(min_t1, min_t2)

    # print(Z[max_i[0]][max_i[1]], '==', abs(R_a(prob_a, max_t1, max_t2))**2)

    # print(z_max * prob_a, z_min * prob_a)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_title('New probability')

    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    ax.set_zlabel('Pr')

    # ax.set_zlim(0, 1)

    # ax.xaxis.set_major_formatter(FuncFormatter(
    #     lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
    # ))
    # ax.yaxis.set_major_formatter(FuncFormatter(
    #     lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
    # ))
    ax.zaxis.set_major_formatter('{x:.02f}')

    # ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
    # ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    ax.zaxis.set_major_locator(LinearLocator(5))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def test3():
    prob_a = 0.4
    prob_b = 1 - prob_a

    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(211)
    
    ax.set_ylabel('Pr')

    # x_values = np.arange(3.7, 4.7, 0.1)
    x_values = np.arange(0, 3.1, 0.05)

    angle = -0.1448124982389391
    b1 = 4.886921905584122
    b2 = 1.3962634015954636

    # TODO
    # https://stackoverflow.com/questions/21486917/finding-cell-coordinates-between-two-points-in-meshgrid-using-python

    y_values = []
    for x in x_values:
        t1 = -x * math.cos(angle) + b1
        t2 = -x * math.sin(angle) + b2

        # t1 = x * math.cos(math.pi * n) + b1
        # t2 = x * math.sin(math.pi * n) + b2

        # t1 = math.pi * x
        # t2 = 0.15 * t1

        # y_values.append(math.log(abs(R_a(prob_a, t1, t2))**2))

        y_values.append(amplify_probability(prob_a, t1, t2))

    t1_max = -3.078616816727874 * math.cos(angle) + b1
    t2_max = -3.078616816727874 * math.sin(angle) + b2

    t1_0 = -3.078616816727874/2 * math.cos(angle) + b1
    t2_0 = -3.078616816727874/2 * math.sin(angle) + b2

    print("T:", amplify_probability(prob_a, t1_0, t2_0))

    # ax.set_ylim(0, 1)
    ax.plot(x_values, y_values)

    plt.show()

def test4():
    prob_a = 0.5

    theta1_values = np.linspace(0, 2 * math.pi, num=100)
    theta2_values = np.linspace(0, 2 * math.pi, num=100)

    X, Y = np.meshgrid(theta1_values, theta2_values)
    Z = np.array([[amplify_probability(prob_a, t1, t2) for t2 in theta2_values] for t1 in theta1_values])

    # Index of a maximum and a minimum of Pr
    max_i = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    min_i = np.unravel_index(np.argmin(Z, axis=None), Z.shape)

    max_p = X[max_i[0]][max_i[1]], Y[max_i[0]][max_i[1]]
    min_p = X[min_i[0]][min_i[1]], Y[min_i[0]][min_i[1]]

    max_z = Z[max_i[0]][max_i[1]]
    min_z = Z[min_i[0]][min_i[1]]
    
    print("MAX:", max_p, '=>', max_z)
    print("MIN:", min_p, '=>', min_z)
    
    slope = (max_p[1] - min_p[1]) / (max_p[0] - min_p[0])
    rad = np.arctan(slope)

    print("ANGLE:", rad)

    magnitude = math.sqrt((max_p[0] - min_p[0])**2 + (max_p[1] - min_p[1])**2)

    # print("MAG:", magnitude)

    # Go up the line left->right or right->left
    a = 1 if max_p[0] > min_p[0] else -1

    def f(x):
        t1 = a * x * math.cos(rad) + min_p[0]
        t2 = a * x * math.sin(rad) + min_p[1]

        return amplify_probability(prob_a, t1, t2) - prob_a

    from scipy.optimize import root, minimize, basinhopping

    # Find point where Z = prob_a
    r = root(f, magnitude / 2)

    print("ROOT:", r.x)

    b1 = a * r.x * math.cos(rad) + min_p[0]
    b2 = a * r.x * math.sin(rad) + min_p[1]

    print("MID:", b1, b2)

    distance_min = math.sqrt((b1 - min_p[0])**2 + (b2 - min_p[1])**2)
    distance_max = math.sqrt((b1 - max_p[0])**2 + (b2 - max_p[1])**2)

    # print("DISTANCES:", distance_min, distance_max)

    lower_bound = -distance_min # -min(distance_min, distance_max)
    upper_bound = distance_max # max(distance_min, distance_max)

    # print("BOUNDS:", lower_bound, upper_bound)

    def final_f(x):
        t1 = a * x * math.cos(rad) + b1
        t2 = a * x * math.sin(rad) + b2

        return amplify_probability(prob_a, t1, t2)

    # Plot
    def plot_pr():
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax = fig.add_subplot(211)

        x_values = np.arange(lower_bound, upper_bound, 0.1)
        y_values = np.array([final_f(x) for x in x_values])

        ax.plot(x_values, y_values)
        plt.show()

    def plot_3d():
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0) #, antialiased=False)
        ax.scatter(b1, b2, prob_a, s=144.0, color='green')

        ax.set_title('New probability')

        ax.set_xlabel('θ1')
        ax.set_ylabel('θ2')
        ax.set_zlabel('Pr')

        ax.set_zlim(0, 1)

        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
        ))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.zaxis.set_major_locator(LinearLocator(5))

        # fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    plot_pr()

def test5(prob_a):
    max_pr = lambda thetas: -amplify_probability(prob_a, thetas[0], thetas[1])
    min_pr = lambda thetas: amplify_probability(prob_a, thetas[0], thetas[1])

    minimizer_kwargs = { 'method': 'L-BFGS-B', 'bounds': ((0.0, 2*math.pi), (0.0, math.pi)) }
    max_p = basinhopping(max_pr, [math.pi, math.pi], minimizer_kwargs=minimizer_kwargs).x
    min_p = basinhopping(min_pr, [math.pi, math.pi], minimizer_kwargs=minimizer_kwargs).x

    # print("MIN:", min_p, '->', amplify_probability(prob_a, min_p[0], min_p[1]))
    # print("MAX:", max_p, '->', amplify_probability(prob_a, max_p[0], max_p[1]))

    slope = (max_p[1] - min_p[1]) / (max_p[0] - min_p[0])
    angle = np.arctan(slope)
    magnitude = math.sqrt((max_p[0] - min_p[0])**2 + (max_p[1] - min_p[1])**2)

    # Ascending or descending, depending if min->max is left->right or right->left
    asc = 1 if max_p[0] > min_p[0] else -1

    def find_base_pr(x):
        theta1 = asc * x * math.cos(angle) + min_p[0]
        theta2 = asc * x * math.sin(angle) + min_p[1]
        return amplify_probability(prob_a, theta1, theta2) - prob_a

    base_prob = root(find_base_pr, [magnitude / 2]).x[0]
    b = asc * base_prob * math.cos(angle) + min_p[0], asc * base_prob * math.sin(angle) + min_p[1]

    # print("B:", b)

    distance_min = math.sqrt((b[0] - min_p[0])**2 + (b[1] - min_p[1])**2)
    distance_max = math.sqrt((b[0] - max_p[0])**2 + (b[1] - max_p[1])**2)

    lower_bound = -distance_min
    upper_bound = distance_max

    # print("BOUNDS:", lower_bound, upper_bound)

    def new_pr(x):
        theta1 = asc * x * math.cos(angle) + b[0]
        theta2 = asc * x * math.sin(angle) + b[1]
        return amplify_probability(prob_a, theta1, theta2)

    def reward_to_magnitude(reward):
        Q = - upper_bound / lower_bound
        growth_rate = 0.35
        return lower_bound + (upper_bound - lower_bound) / (1 + Q * math.e**(-growth_rate * reward))

    def reward_to_pr(reward):
        theta1 = asc * reward_to_magnitude(reward) * math.cos(angle) + b[0]
        theta2 = asc * reward_to_magnitude(reward) * math.sin(angle) + b[1]
        return amplify_probability(prob_a, theta1, theta2)

    # Plots

    def plot_pr_mesh(fig):
        ax = fig.add_subplot(2, 2, 1, projection='3d')

        theta1_values = np.linspace(0, 2 * math.pi, num=100)
        theta2_values = np.linspace(0, 2 * math.pi, num=100)

        # Plot probability mesh
        xx, yy = np.meshgrid(theta1_values, theta2_values)
        zz = np.array([[amplify_probability(prob_a, t1, t2) for t2 in theta2_values] for t1 in theta1_values])
        surf = ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.7) #, linewidth=0, antialiased=False)

        # Plot probability line
        m_values = np.arange(lower_bound, upper_bound, 0.1)
        x_values = np.array([asc * x * math.cos(angle) + b[0] for x in m_values])
        y_values = np.array([asc * x * math.sin(angle) + b[1] for x in m_values])
        z_values = np.array([new_pr(x) for x in m_values])

        ax.plot(x_values, y_values, z_values, color='green')
        ax.scatter(b[0], b[1], prob_a, s=72.0, color='green')

        # Misc
        ax.set_title('Probability mesh')
        ax.set_xlabel('θ1')
        ax.set_ylabel('θ2')
        ax.set_zlabel('Pr')

        # Limits
        ax.set_xlim(0, 2 * math.pi)
        ax.set_ylim(0, 2 * math.pi)
        ax.set_zlim(0, 1)

        # Formatters
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
        ))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val != 0 else '0'
        ))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))
        ax.zaxis.set_major_locator(LinearLocator(5))

        # fig.colorbar(surf, shrink=0.5, aspect=5)

    def plot_new_pr(fig):
        ax = fig.add_subplot(2, 2, 2)

        x_values = np.arange(lower_bound, upper_bound, 0.1)
        y_values = np.array([new_pr(x) for x in x_values])

        ax.plot(x_values, y_values)

        # Misc
        ax.set_title('Magnitude to probability')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Pr')

    def plot_reward_magnitude(fig):
        ax = fig.add_subplot(2, 2, 3)

        x_values = np.arange(-10, 10, 0.1)
        y_values = np.array([reward_to_magnitude(x) for x in x_values])

        ax.plot(x_values, y_values)

        # Misc
        ax.set_title('Reward to magnitude')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Magnitude')

    def plot_reward_pr(fig):
        ax = fig.add_subplot(2, 2, 4)

        x_values = np.arange(-10, 10, 0.1)
        y_values = np.array([reward_to_pr(x) for x in x_values])

        ax.plot(x_values, y_values)

        # Misc
        ax.set_title('Reward to probability')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Probability')

    # Plot
    fig = plt.figure(figsize=plt.figaspect(0.5))

    plot_pr_mesh(fig)
    plot_new_pr(fig)
    plot_reward_magnitude(fig)
    plot_reward_pr(fig)

    plt.show()

def gui():
    from checkers import Checkers, Color
    from checkers_gui import CheckersGui

    env = Checkers(shape=(8,8), opponent='optimal3')
    env.reset()

    agent1 = None

    CheckersGui(env, None, None)

    print(f'Winner: {env.winner}')

#gui()

from functools import reduce

prob = np.array([0.20, 0.20, 0.05, 0.05, 0.00000000001, 0.0, 0.25, 0.25 - 0.00000000001])
flags = [2]
epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)

print(np.sum(prob))

print("INITIAL:", epsilon, prob)

k = math.ceil(1 / math.sqrt(epsilon))

#k = math.floor(math.pi / (4 * math.asin(math.sqrt(epsilon))) - 0.5)

m = np.random.randint(0, k + 1)
print(m)



# k = math.floor(math.pi / (4 * math.sqrt(epsilon)) - 0.5)

# print(k)

for i in range(m):
    prob = np.array([amplify_probability(p, epsilon, math.pi, math.pi) if i in flags else deamplify_probability(p, epsilon, math.pi, math.pi) for i,p in enumerate(prob)])
    epsilon = reduce(lambda e, i: e + prob[i], flags, 0.0)

print("FINAL:", epsilon, prob)

# from qiskit import QuantumRegister, QuantumCircuit, execute
# from qiskit.quantum_info import Statevector
# from qiskit.circuit.library import Diagonal, GroverOperator
# from qiskit.extensions import Initialize
# from qiskit.providers.aer import QasmSimulator

# backend = QasmSimulator(method='statevector')

# num_qubits = 3

# U = Initialize(np.sqrt(prob) * complex(1, 0)).gates_to_uncompute().inverse().copy(name='U')

# qreg = QuantumRegister(num_qubits, name='q')
# circ = QuantumCircuit(qreg)

# circ.append(U.to_instruction(), qreg)

# try:
#     k = math.floor(math.pi / (4 * math.asin(math.sqrt(epsilon))))
#     k = min(k, 20)
# except:
#     k = 0

# if k > 0:
#     grover = GroverOperator(
#         oracle=Diagonal([-1 if i in flags else 1 for i in range(2**num_qubits)]),
#         state_preparation=U
#     ).repeat(k)

# circ.snapshot('sv')

# result = execute(circ, backend=backend, shots=1).result()
# resulting_sv = result.data()['snapshots']['statevector']['sv'][0]
# print(Statevector(resulting_sv).probabilities_dict())

#gui()

# from itertools import islice, count

# def f(num_cols, L):
#     cols = np.zeros(num_cols, dtype=int)

#     for i in range(num_cols):
#         cols[i] = 4 if L - np.sum(cols[:i]) > 4 else L - np.sum(cols[:i])

#     while True:
#         action = ''
        
#         for i in range(num_cols):
#             action += str(i) * cols[i]

#         yield action

#         updated = False

#         for i,j in zip(range(num_cols - 2, -1, -1), range(num_cols - 1, -1, -1)):
#             if cols[i] > 0 and cols[j] < 4:
#                 cols[i] -= 1
#                 cols[j] += 1
#                 updated = True
#                 break

#         if not updated:
#             break
        
            

# gen = f(5, 10)

# for i in range(25):
#     next(gen)

# n = 3
# x = next(islice(f(), n, n+1))
# print(x)






#from connect4 import Connect4
#from agents import RandomAgent

#env = Connect4(opponent=RandomAgent())


# from agents import QRPSAgent

# a = QRPSAgent(None, 0, 0)

# a.select_action('xxxx', [0, 1, 2])

#gui()

#test5(0.8)

# from qiskit.providers.aer import QasmSimulator

# from checkers_gui import CheckersGui

# from checkers import Checkers
# from agents import RandomAgent, HumanAgent
# from quantum_agent import QuantumAgent

# backend = QasmSimulator(method='statevector')

# env = Checkers()
# env.reset()

# agent1 = None
# agent2 = QuantumAgent('black', backend)

# print(len(agent2.memory))


# CheckersGui(env, agent1, agent2)


# print(f'Winner: {env.winner}')
