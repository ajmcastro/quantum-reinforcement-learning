import math
import numpy as np

def prob_to_angles(prob, previous=1.0):
    "Calculates the angles to encode the given probabilities"

    def calc_angle(x):
        try:
            return 2 * math.acos(math.sqrt(x))
        except:
            print(x)
            raise()
    
    if len(prob) == 2:
        return [calc_angle(prob[0] / previous)] if previous != 0.0 else [0.0]

    lhs, rhs = np.split(prob, 2)

    angles = np.array([
        calc_angle((np.sum(lhs)/ previous) if previous != 0.0 else 0.0)
    ])

    angles = np.append(angles, prob_to_angles(lhs, previous=np.sum(lhs)))
    angles = np.append(angles, prob_to_angles(rhs, previous=np.sum(rhs)))

    return angles

def R_amplify(prob, t1, t2):
    R = (1 - math.e**complex(0, t1) - math.e**complex(0, t2)) - (1 - math.e**complex(0, t1)) * (1 - math.e**complex(0, t2)) * prob
    return abs(R)**2

def R_deamplify(prob, t1, t2):
    R = -math.e**complex(0, t2) - (1 - math.e**complex(0, t1)) * (1 - math.e**complex(0, t2)) * prob
    return abs(R)**2