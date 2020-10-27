import dislib as ds
import numpy as np
from pycompss.api.api import compss_delete_object
base = 2
max_par = 8


'''
	FULLY QUANTUM MODE: python simulator of a Quantum state

'''

class quantum_state(object):

	def __init__(self, N):
		self._N = N
		self._hilbert_size = pow(2,N)
		self.zero_state()
		return


	def empty_state(self):
		bs = self._hilbert_size//max_par
		self._state = ds.zeros(shape=(1,self._hilbert_size), block_size=(1,bs), dtype=np.complex64)
		return


	def zero_state(self):
		self.empty_state()
		self._state[0,0]=1
		self._state = self._state.T
		return


	def one_state(self):
		self.empty_state()
		self._state[0,-1]=1
		self._state = self._state.T
		return


	def random_state(self):
		return


	def apply_op(self, operator):
		return


	def apply_gate(self, gate, pos):
		bs = self._hilbert_size//max_par
		if len(pos) == 1:		
			pos = [pos[0], pos[0]]
		g_0 = self.build_gate(gate, pos)
		g_0_ds = ds.array (g_0, block_size=(bs,bs))

		self._state = g_0_ds @ self._state		
		return

	
	def norm(self):
		return


	def normalize(self):
		return	

	
	def exp_value(self, operator, pos):
		bs = self._hilbert_size//max_par
		if len(pos) == 1:		
			pos = [pos[0], pos[0]]
		g_0 = self.build_gate(operator, pos)
		g_0_ds = ds.array (g_0, block_size=(bs,bs))
		state_conj = self._state.conj()
		first_prod = g_0_ds @ self._state
		state_conj = state_conj.T
		nn = state_conj @ first_prod
		dd = state_conj @ self._state
		n = nn.collect()
		d = dd.collect()
		return n/d


	def __str__(self):
		return str(self.squares())		

	def squares(self):
		return np.multiply(np.conj(self._state), self._state)



	def build_gate(self, gate, pos):
		gates = [np.identity(pow(base,pos[0])), gate, np.identity(pow(base,self._N-pos[1]-1))]
		g_0 = gates[0]
		for pos in range(1,len(gates)):
			g_0 = np.kron(g_0, gates[pos])
			print("g_0 shape")
			print (g_0.shape)
		return g_0
		
	
