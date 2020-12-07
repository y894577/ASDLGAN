import numpy as np
from PIL import Image
from matplotlib import mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt

q = 10000
k = 1000
p = 0.8
x_1 = 0.2
dict = [x_1]
m = 10
n = 2
i = 1

A = np.zeros([n, n])

for j in range(q):
    w = np.random.normal()
    x_n = p * dict[j] + w
    dict.append(x_n)

array = []
for j in range(i, q - n * n):
    cur = dict[j:j + n * n]
    y = [(int(a * k) % m) for a in cur]
    # print(y)
    a = np.reshape(y, (n, n))
    # print(a)
    # print(np.linalg.det(A))

    if round(np.linalg.det(a)) == 1:
        # print(np.linalg.det(A))
        # print(A)
        array.append(a)

A = array[0]
for i in range(1, 6):
    A = np.kron(A, array[i])

np.savetxt('array.txt', A, fmt='%d', delimiter=',')

A_1 = np.linalg.inv(np.loadtxt('array.txt', delimiter=','))

im = Image.open('资源 407 (2).png')
width, height = im.size

new_img = im.convert('L')
B = np.array(new_img)

test = B % 256
test = Image.fromarray(test.astype(np.uint8))
test.show()

mat = np.matmul(A, B) % 256
new = Image.fromarray(mat.astype(np.uint8))
new.show()

B_1 = np.matmul(A_1, mat) % 256
new2 = Image.fromarray(B_1.astype(np.uint8))
new2.convert('L')
new2.show()

X = np.tile(mat, (8, 8))

mat = mat.reshape([-1, 2])
# print(mat / 256)


