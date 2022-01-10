import math
from copy import deepcopy

def negamax(env, depth, alpha, beta, color):
    if env.is_over or depth == 0:
        return None, color * env.evaluate()
    
    best_value = -math.inf
    best_action = None

    possible_actions = env.actions()
    possible_actions.sort(key=lambda t: len(t[2]), reverse=True)

    for a in possible_actions:
        env_copy = deepcopy(env)
        env_copy._step(a)
        
        value = -negamax(env_copy, depth - 1, -beta, -alpha, -color)[1]

        if value > best_value:
            best_value = value
            best_action = a

        alpha = max(alpha, best_value)

        if alpha >= beta:
            break
    
    return best_action, best_value



