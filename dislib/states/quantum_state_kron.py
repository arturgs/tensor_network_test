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
		self._state = g_0 @ self._state		
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
		state_conj = self._state.conj()
		first_prod = g_0 @ self._state
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
		bs = self._hilbert_size//max_par
		gates = [np.identity(pow(base,pos[0])), gate, np.identity(pow(base,self._N-pos[1]-1))]
	
		#gate[0]
		bsize_x = gates[0].shape[0]//max_par
		if (bsize_x < 1 ):
			bsize_x=1
		bsize_y = gates[0].shape[1]//max_par
		if (bsize_y < 1):
			bsize_y=1
		bsize = (bsize_x, bsize_y)
		g_0 = ds.array(gates[0], block_size=bsize)
		#gate[1]
		bsize_x = gates[1].shape[0]//max_par
		if (bsize_x < 1 ):
			bsize_x=1
		bsize_y = gates[1].shape[1]//max_par
		if (bsize_y < 1):
			bsize_y=1
		bsize = (bsize_x, bsize_y)
		g_0 = ds.kron(g_0, ds.array(gates[1], block_size=bsize))

		#gate[2]
		bsize_x = gates[2].shape[0]//max_par
		if (bsize_x < 1 ):
			bsize_x=1
		bsize_y = gates[2].shape[1]//max_par
		if (bsize_y < 1):
			bsize_y=1
		bsize = (bsize_x, bsize_y)
		g_0 = ds.kron(g_0, ds.array(gates[2], block_size=bsize), block_size=(bs,bs))
		
		return g_0

