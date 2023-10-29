import numpy as np

array1 = np.array([1.0, 2.0, 3.0])
array2 = np.array([2.0, 4.0, 6.0])

are_equal = np.allclose(array1, array2, rtol=0.01)

print(are_equal)