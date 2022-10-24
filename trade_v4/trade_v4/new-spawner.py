import numpy as np

def draw_quarter_circle_in_array(radius):
    arr = np.zeros((radius, radius))
    for i in range(radius):
        for j in range(radius):
            if (i)**2 + (j)**2 <= radius**2:
                arr[i, j] = 1
    return arr
def stack_quarter_circles(radius):
    # produces 0-radius quarter circles
    arr = draw_quarter_circle_in_array(radius)
    for i in range(radius):
        quart_circ_i = draw_quarter_circle_in_array(i)
        arr[:i,:i] += quart_circ_i

    return arr

def gen_point(radius):

    qc = stack_quarter_circles(3)
    idxs = np.indices(qc.shape).reshape(2, -1).T
    probs = qc.flatten() / np.sum(qc)

    while True:
        yield idxs[np.random.choice(np.arange(len(idxs)), p=probs)]

x = np.zeros((4, 4))
point_generator = gen_point(4)
for i in range(100):
    point = next(point_generator)
    print(point)
    x[point[0], point[1]] += 1

import matplotlib.pyplot as plt
plt.imshow(x)
plt.show()
