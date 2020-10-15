import numpy as np


class mps:


	'''
		Initialize MPs data structure
		It consists in 2 matrices for each position
		We access these matrices with a dictionary indexed by (dimension,position)
		where dimension = [0,1] and position = [0,...,N-1]
	'''
	def __init__(self, size, dimension):
	    self.N = size
	    self.D = dimension
	    self.data = {}
	    for i in range(self.N):
	        for spin in range(2):
	            self.data[(i, spin)] = np.random.rand(self.D,self.D)
	    return


	''' 
		The MP rewrite the coefficients of a vector as a product of matrices
		We may recover the original vector's coefficients by multiplying the corresponding matrices
	'''
	def get_coeff(self, indexes):
	    m_index = np.eye(self.D, self.D) 
	    position = 0
	    for i in indexes:
	        m_index = np.dot(m_index, self.data[(position, i)])
	        position += 1
	    return np.trace(m_index)


	'''
		We  may recover the matrices (x2) of a particular vector position
		This is useful to apply a single-site quantum gate 
	'''
	def get_matrix(self, position):
	    m0 = self.data[(position,0)]
	    m1 = self.data[(position,1)]
	    return m0, m1

	def set_matrix(self, position, m0, m1):
	    self.data[(position,0)] = m0
	    self.data[(position,1)] = m1
	    return 1



	'''
		This is a sinlge-site quantum gate operation
		We first recover the matrices of this particular position
		After combining these matrices in a single matrix of size (2,D^2), we apply the Quantum gate
		After applying the gate, we recover the original matrix form by forming 2 matrices of size (D,D)
	''' 
	def apply_gate(self, gate, position):
	    m0, m1 = self.get_matrix(position)
	    m0 = np.reshape(m0,(1, self.D**2))
	    m1 = np.reshape(m1,(1, self.D**2))
	    base_m = np.vstack((m0,m1))
	    base_m = np.dot(gate,base_m)
	    m0 = base_m[0,:]
	    m1 = base_m[1,:]
	    self.data[(position,0)] = np.reshape(m0,(self.D, self.D))
	    self.data[(position,1)] = np.reshape(m1,(self.D, self.D))
	    return 1




if __name__=="__main__":
	'''
		We perform some tests:
		- create an MPs
		- recover matrices
		- get coefficients
		- swap coefficients by applying a Quantum gate
	'''
	mps_1 = mps(5,3)
	m0, m1 = mps_1.get_matrix(2)
	print(m0, m1, m0.dot(m1))
	print(mps_1.get_coeff([0,1,1,0,1]), mps_1.get_coeff([0,0,1,0,1]))

	gate = np.array([[0,1],[1,0]])
	mps_1.apply_gate(gate, 1)
	print(mps_1.get_coeff([0,1,1,0,1]), mps_1.get_coeff([0,0,1,0,1]))

