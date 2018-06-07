import multiprocessing
from tqdm import tqdm
import numpy as np
import time


def func(x):
    time.sleep(1)
    return x[0], x[1]**2.


pool = multiprocessing.Pool(processes=10)

xv, yv = np.array(list(pool.imap(func, [[0, 1], [2, 3], [4, 5]]))).T

print xv
print yv
