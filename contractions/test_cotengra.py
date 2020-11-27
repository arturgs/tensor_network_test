import quimb.tensor as qtn
import cotengra as ctg

def load_circuit(
    n=53,
    depth=10,
    seed=0 ,
    elided=0,
    sequence='ABCDCDAB',
    swap_trick=False
):
    file = f'circuit_n{n}_m{depth}_s{seed}_e{elided}_p{sequence}.qsim'

    if swap_trick:
        gate_opts={'contract': 'swap-split-gate', 'max_bond': 2}  
    else:
        gate_opts={}
    
    # instantiate the `Circuit` object that 
    # constructs the initial tensor network:
    return qtn.Circuit.from_qasm_file(file, gate_opts=gate_opts)

circ = load_circuit(depth=10)
psi_f = qtn.MPS_computational_state('0' * (circ.N))
tn = circ.psi & psi_f
output_inds = []


# inplace full simplify and cast to single precision
tn.full_simplify_(output_inds=output_inds)
tn.astype_('complex64')

opt = ctg.HyperOptimizer(
   # methods=['kahypar', 'greedy', 'walktrap'],
    methods = ['greedy','kahypar'],
    max_repeats=128,
    progbar=True,
    minimize='flops',
    score_compression=0.5,  # deliberately make the optimizer try many methods 
)

info = tn.contract(all, optimize=opt, get='path-info')

print(tn.contract(all, optimize=opt.path, backend='pycompss'))
