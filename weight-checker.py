import numpy as np

data = np.load("weights.npz")

total = 0
for name in data.files:
    arr = data[name]
    print(name, arr.shape, arr.size)
    total += arr.size

print("total weights:", total)
