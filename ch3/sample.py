import numpy as np

sample = [1,2,3,4,5,6,7,8]
print(sample[1:-1])

sample = np.array([
    [0,2],
    [1,3],
    [2,4],
    [3,1],
    [4,5],
    [1,6]
])
print(sample[:,1])


a = np.zeros((2,3,4))
print(a)

print(a.shape[0])

