import numpy
from scipy.sparse import diags

#
n = 10
k = numpy.array([numpy.ones(n - 1), -2 * numpy.ones(n), numpy.ones(n - 1)])
offset = [1, 0, -1]

A = diags(k, offset, dtype=complex).toarray()

f = numpy.arange(0, n, dtype=complex)

h = (n - 0) / n
A /= h ** 2

print(A)
print(f)

v = A * f
print(v)

v2 = numpy.dot(A, f)
print(v2)
