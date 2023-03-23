# if "__file__" in globals():
#     import os, sys

#     sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

import dezero.datasets

x, t = dezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

d1 = x[:, 0]
d2 = x[:, 1]

plt.scatter(d1, d2)
plt.show()

