import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[6, 5, 4], [3, 2, 1]])
print(np.linalg.norm((a - b), axis=1)**2)
print(np.sum((a - b)**2, axis=1))