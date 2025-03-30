import numpy as np
import matplotlib.pyplot as plt
from cpo import cpo

def sphere_function(x):
    """
    Sphere function - a simple benchmark function
    f(x) = sum(x^2)
    Global minimum: f(0,...,0) = 0
    """
    return np.sum(x**2)

def rosenbrock_function(x):
    """
    Rosenbrock function - a more complex benchmark function
    f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2)
    Global minimum: f(1,...,1) = 0
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def rastrigin_function(x):
    """
    Rastrigin function - a multimodal benchmark function
    f(x) = 10*n + sum(x^2 - 10*cos(2*pi*x))
    Global minimum: f(0,...,0) = 0
    """
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

if __name__ == "__main__":
    # Define the problem parameters
    dim = 10  # Dimension of the problem
    pop_size = 50  # Population size
    max_iter = 1000  # Maximum number of iterations
    
    # Define bounds for the search space
    lb = -5 * np.ones(dim)  # Lower bounds
    ub = 5 * np.ones(dim)   # Upper bounds
    
    # Run CPO on different objective functions
    functions = [
        {"name": "Sphere", "func": sphere_function},
        {"name": "Rosenbrock", "func": rosenbrock_function},
        {"name": "Rastrigin", "func": rastrigin_function}
    ]
    
    for func_info in functions:
        func_name = func_info["name"]
        obj_func = func_info["func"]
        
        print(f"\nRunning CPO on {func_name} function...")
        
        # Run the algorithm
        best_fitness, best_position, convergence = cpo(pop_size, max_iter, ub, lb, dim, obj_func)
        
        # Print results
        print(f"Best fitness: {best_fitness}")
        print(f"Best position: {best_position}")
        
        # Plot convergence curve
        plt.figure(figsize=(10, 6))
        plt.semilogy(convergence, linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Best Score')
        plt.title(f'CPO on {func_name} Function')
        plt.grid(True)
        plt.savefig(f'CPO_{func_name}_Convergence.png')
        plt.close()
    
print("All tests completed!") 