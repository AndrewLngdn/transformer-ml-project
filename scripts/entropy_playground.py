import numpy as np


for i in range(2, 10):
    p = 1 / i
    probs = np.array([p] * i)
    entropy = np.sum(-np.log2(probs) * probs)
    print(f"probs: {probs}")
    print(f"entropy: {entropy}")
    print("-" * 25)
