import numpy as np
from scipy.sparse import diags

#
n = 10
k = np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)])
offset = [1, 0, -1]

A = diags(k, offset, dtype=complex).toarray()

f = np.arange(0, n, dtype=complex)

h = (n - 0) / n
A /= h**2

print(A)
print(f)

v = A * f
print(v)

v2 = np.dot(A, f)
print(v2)