import numpy as np

def create_matrix(size):
    current_size = [size - 1, (size -1) // 2]
    matrix = np.ones((size, size)) * 0.5
    matrix[size - 1, size - 1] = 0.5
    iteration = 2
    while True:
        matrix[current_size[-1], current_size[-2]] = 1 - 1/2**iteration
        matrix[current_size[-2], current_size[-1]] = 1/2**iteration

        plane = find_plane([np.array((current_size[-1], current_size[-1], 0.5)),
                            np.array([current_size[-1], current_size[-2], 1 - 1 / 2 ** iteration]),
                            np.array([current_size[-2], current_size[-1], 1 / 2 ** iteration])])
        for i in range(current_size[-1], current_size[-2] + 1):
            for j in range(current_size[-1], current_size[-2] + 1):
                matrix[i, j] = - plane[0] / plane[2] * i - plane[1] / plane[2] * j - plane[3] / plane[2]

        # if iteration > 2:
        #     plane = find_plane([np.array((current_size[-2], current_size[-2], 0.5)),
        #             np.array([current_size[-2], current_size[-3], 1 - 1 / 2 ** (iteration - 1)]),
        #             np.array([current_size[-1], current_size[-2], 1 - 1 / 2 ** (iteration)])])
        #     for i in range(current_size[-1], current_size[-2]):
        #         for j in range(current_size[-2] + 1, int(np.ceil(current_size[-2] + (current_size[-3] - current_size[-2]) * (i - current_size[-1]) / (current_size[-2] - current_size[-1])))):
        #             matrix[i, j] = -plane[0] / plane[2] * i - plane[1] / plane[2] * j - plane[3] / plane[2]
        #
        #     plane = find_plane([np.array((current_size[-2], current_size[-2], 0.5)),
        #             np.array([current_size[-2], current_size[-1], 1 / 2 ** (iteration)]),
        #             np.array([current_size[-3], current_size[-2], 1 / 2 ** (iteration - 1)])])
        #     for i in range(current_size[-2] + 1, current_size[-3] + 1):
        #         for j in range(current_size[-1], current_size[-2]):
        #             matrix[i, j] = -plane[0] / plane[2] * i - plane[1] / plane[2] * j - plane[3] / plane[2]

        iteration += 1
        current_size.append(current_size[-1] // 2)
        if current_size[-1] == 0:
            break
    return matrix

def find_plane(points):
    p1, p2, p3 = points[0], points[1], points[2]
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = -np.dot(cp, p3)
    return np.array((a,b,c,d))

if __name__ == "__main__":
    size = 64
    m = create_matrix(size)

    import matplotlib.pyplot as plt
    plt.imshow(m)
    plt.colorbar()
    plt.show()

    from ODO import online_double_oracle
    from DO import double_oracle
    res = online_double_oracle(m)
    print(res)
