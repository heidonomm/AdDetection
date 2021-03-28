import numpy as np


# print(np.zeros((30)))
# arr = np.array([[10, 10], [3, 3]], dtype='float32')
# arr2 = np.array([[1, 1]])
# print(arr.mean(axis=0))
# print(np.concatenate((arr, arr2)))

kaks = [[1, 2]]
lb = [0, 1]
predicted_class = [0 if prob < 0.5 else 1 for prob in lb]

n = zip(lb, kaks)
print(list(n))
