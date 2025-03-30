import numpy as np
import matplotlib.pyplot as plt
import time
from cpo import cpo
from benchmark_functions import get_function_details, get_cec_function

# Algorithm parameters
pop_size = 130  # Number of search agents (crested porcupines)
max_iter = 1000000  # Maximum number of function evaluations
num_runs = 30  # Number of independent runs

# CEC benchmark selection
# 1: CEC-2014, 2: CEC-2017, 3: CEC-2020, 4: CEC-2022
cec_year = 2  

# Function IDs to test
if cec_year == 1:  # CEC-2014
    benchmark_name = 'CEC-2014'
    func_ids = list(range(1, 31))
elif cec_year == 2:  # CEC-2017
    benchmark_name = 'CEC-2017'
    func_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # Skip function 2 in CEC-2017 as per original MATLAB code
elif cec_year == 3:  # CEC-2020
    benchmark_name = 'CEC-2020'
    func_ids = list(range(1, 11))
elif cec_year == 4:  # CEC-2022
    benchmark_name = 'CEC-2022'
    func_ids = list(range(1, 13))
    
# Get the appropriate benchmark function
cec_func = get_cec_function(cec_year)

# Run the optimization for each function
for func_id in func_ids:
    # Skip functions not available for certain CEC years
    if cec_year == 2 and func_id == 2:
        continue
    elif cec_year == 3 and func_id > 10:
        break
    elif cec_year == 4 and func_id > 12:
        break
    
    # Get function details (bounds and dimension)
    lb, ub, dim = get_function_details(func_id, cec_year)
    
    # For storing results
    all_fitness = np.zeros(num_runs)
    all_convergence = np.zeros((num_runs, max_iter))
    
    start_time = time.time()
    
    # Run algorithm num_runs times
    for run in range(num_runs):
        best_score, best_pos, convergence_curve = cpo(pop_size, max_iter, ub, lb, dim, cec_func, func_id)
        all_fitness[run] = best_score
        all_convergence[run, :] = convergence_curve
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    # Print results
    print(f"Benchmark: {benchmark_name}, Function ID: {func_id}, "
          f"Average Fitness: {np.mean(all_fitness):.20f}, "
          f"Average Time: {avg_time:.5f}")
    
    # Plot convergence curve
    plt.figure(func_id)
    mean_convergence = np.mean(all_convergence, axis=0)
    plt.semilogy(mean_convergence, '-<', markersize=8, linewidth=1.5, markevery=20000)
    plt.xlabel('Function Evaluation')
    plt.ylabel('Best Fitness obtained so-far')
    plt.title(f'CPO on {benchmark_name} F{func_id}')
    plt.grid(False)
    plt.box(True)
    plt.legend(['CPO'])
    plt.tight_layout()
    plt.savefig(f'CPO_{benchmark_name}_F{func_id}.png')
    plt.close()

print("Optimization completed") 