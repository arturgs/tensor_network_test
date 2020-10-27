#module load dislib/unstable
#module load dislib/0.5.0
export COMPSS_PYTHON_VERSION="3-ML"
module load COMPSs/Trunk
#module load COMPSs
module load mkl/2019.2


enqueue_compss --qos=debug --job_name=quantum_k  --log_level=off  --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler --worker_in_master_cpus=0 --worker_working_dir=/gpfs/scratch/bsc19/compss/COMPSs_Sandbox/ --max_tasks_per_node=48 --exec_time=60 --num_nodes=2 --python_interpreter=python3 --pythonpath=/home/bsc19/bsc19776/my_apps/dislib hybrid_optimization_kron.py



# --qos=debug
#    enqueue_compss -t --qos=debug --job_name=lr-scratch  --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler --worker_in_master_cpus=0 --max
#_tasks_per_node=48 --worker_working_dir=/gpfs/scratch/bsc19/bsc19776 --exec_time=120 --num_nodes=${nodes} --base_log_dir=/gpfs/scratch/bsc19/bsc19776/ line
#ar_regression.py
#
# for nodes in 2 3 5 9 

#for nodes in 2 3 5 9 
#do
#    enqueue_compss --qos=debug --job_name=lr-scratch  --log_level=off  --scheduler=es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler --worker_in_master
#_cpus=0 --worker_working_dir=gpfs --max_tasks_per_node=48 --exec_time=10 --num_nodes=${nodes} linear_regression.py
#done

