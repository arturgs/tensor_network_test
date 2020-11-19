from pycompss.api.task import task
from pycompss.api.api import compss_wait_on

import numpy as np
import time
import opt_einsum as oe


def printState(s):
    st = np.ndarray.flatten(s)
    print(st[::-1],sep='')


@task(returns = np.ndarray)
def partial_einsum(*args):
    return oe.contract(*args)


def main_contraction():

    # we define some initial quantum states (vectors) and operators (matrices)
    s0 = np.array([1,0])
    s1 = np.array([0,1])

    opX = np.array([[0,1],[1,0]])
    opZ = np.array([[1,0],[0,-1]])
    opY = np.array([[0,-1j],[1j,0]])

    cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    cnot = np.reshape(cnot,[2,2,2,2])


    # we take the circuit contraction representation
    arg = ['a,b,c,d,abij,jckl,ldmn',s0,s1,s1,s1,cnot,cnot,cnot]


    # first we try the global einsum operation over the full circuit
    start_time = time.time()
    res = oe.contract(*arg, optimize='greedy', backend='pycompss')
    direct_time = time.time() - start_time
    print("Direct Einsum contraction")
    printState(res)
    print("total time %f" % direct_time)

    '''
    # then we compute the contraction sequence ... 
    path = oe.contract_path(*arg,optimize='greedy')
    print(path[1])
    print(path[0])


    start_time = time.time()
    # ... and a sequence of contractions
    res1 = partial_einsum('abij,a->bij', cnot, s0) 
    res2 = partial_einsum('jckl,c->jkl', cnot, s1)
    res3 = partial_einsum('ldmn,d->lmn', cnot, s1)
    res4 = partial_einsum('bij,b->ij', res1, s1)
    res5 = partial_einsum('ij,jkl->ikl', res4, res2)
    res6 = partial_einsum('ikl,lmn->ikmn', res5, res3)

    print("Partial Einsum contraction")
    res6 = compss_wait_on(res6)
    partial_time = time.time() - start_time

    printState(res6)
    print("partial time %f" % partial_time)
    '''


if __name__=="__main__":
    main_contraction()
	
