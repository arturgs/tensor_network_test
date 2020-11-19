import numpy as np
import opt_einsum as oe

import time

N = 8
rows = 10
phys_dim = 2

def create_sycamore_row(start_idx, indices):
    einsum_str = ""
    # First row of 2-qubit gates
    for i in range(0, N, 2):
        # set the four indices
        # indices(i)-------start_idx+i
        #              |X|
        # indices(i+1)-----start_idx+i+1
        ul, ll, ur, lr = (oe.get_symbol(i) for i in (indices[i], indices[i+1], start_idx+i, start_idx+i+1))
        einsum_str += "{}{}{}{},".format(ul, ll, ur, lr)
    right_idx = [start_idx]

    # second row       
    for i in range(0, N-3, 2):
        # set the four indices
        # start_idx+i+1-----------start_idx+i+N
        #                  |X|
        # startx_idx+i+2----------start_idx+i+N+1
        ul, ll, ur, lr = (oe.get_symbol(i) for i in (start_idx+i+1, start_idx+i+2, start_idx+i+N, start_idx+i+N+1))
        einsum_str += "{}{}{}{},".format(ul, ll, ur, lr)
        right_idx += [start_idx+i+N, start_idx+i+N+1]
    # compute final column of indices
    right_idx += [start_idx+N-1]

    return einsum_str, right_idx


def gen_shapes():
    for i in range(rows):
        for i in range(0, N-1):
            yield(phys_dim, phys_dim, phys_dim, phys_dim)

einsum_total = ""
right_idx = [i for i in range(N)]

for i in range(rows):
    einsum_str, right_idx = create_sycamore_row(N+i*(2*N-2), right_idx)
    einsum_total += einsum_str

shapes = tuple(gen_shapes())
einsum_total = einsum_total[:-1]
print(einsum_total)

arrays = [np.random.randn(*shp) for shp in shapes]
print(oe.contract_path(einsum_total, *arrays, memory_limit=-1)[0])
print(oe.contract_path(einsum_total, *arrays, memory_limit=-1)[1])

start_time = time.time()
res = oe.contract(einsum_total, *arrays, memory_limit=-1)
_time = time.time() - start_time
print("numpy backend %f" % _time)



start_time = time.time()
res = oe.contract(einsum_total, *arrays, memory_limit=-1, backend="pycompss")
_time = time.time() - start_time
print("pycompss backend %f" % _time)


