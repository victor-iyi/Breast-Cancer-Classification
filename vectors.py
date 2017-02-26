import numpy as np
from math import sqrt

######## Working with vector and magnitude of a vector #########
# Calculating the Magnitude of a Vector

v = [3,4]
print('Magnitude of vector ||v|| =',v)
# magnitude denoted as ||v||
mag_v = sqrt(pow(3,2)+pow(4,2))
print(mag_v)
# using numpy
mag_v = np.linalg.norm(np.array(v))
print(mag_v)


######## Working with the dot product of a vector ########
a = [4,3]
b = [1,2]
print('Dot product a.b, where a =',a,'and b =',b)
# dot product written as "a.b" with the -> on top denoting vector
a_dot_b = (a[0]*b[0]) + (a[1]*b[1])
print(a_dot_b)
# using numpy
a_dot_b = np.dot(a,b)
print(a_dot_b)
# inner product also gives same answer as dot product
print('Inner product of a and b, where a =',a,'and b =',b)
a_inner_b = np.inner(a,b)
print(a_inner_b)

            
