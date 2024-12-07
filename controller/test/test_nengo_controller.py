import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.path.abspath('../controller/'))

from helper import run_controller


target = np.array([64,64])
actual = np.array([20,30])

result = run_controller(actual, target)
print(result)
