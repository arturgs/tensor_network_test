import random
import time
import quimb as qu
import quimb.tensor as qtn

N = 90
circ = qtn.Circuit(N)

# randomly permute the order of qubits
regs = list(range(N))
random.shuffle(regs)

# hamadard on one of the qubits
circ.apply_gate('H', regs[0])

# chain of cnots to generate GHZ-state
for i in range(N - 1):
    circ.apply_gate('CNOT', regs[i], regs[i + 1])

for i in range(N - 2):
    circ.apply_gate('CNOT', regs[i], regs[i + 2])

# sample it 100 times, count results:
start_time = time.time()
circ.sample(100, backend='numpy')
direct_time = time.time() - start_time
print("total time %f" % direct_time)


start_time = time.time()
circ.sample(100, backend='pycompss')
direct_time = time.time() - start_time
print("total time %f" % direct_time)
