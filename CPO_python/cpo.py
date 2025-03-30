import numpy as np

def cpo(pop_size, max_iter, ub, lb, dim, obj_func, func_id=None):
    """
    Crested Porcupine Optimizer: A new nature-inspired metaheuristic
    
    Parameters:
    -----------
    pop_size : int
        Number of search agents (crested porcupines)
    max_iter : int
        Maximum number of iterations
    ub : numpy array
        Upper bounds for each dimension
    lb : numpy array
        Lower bounds for each dimension
    dim : int
        Problem dimension
    obj_func : function
        Objective function to optimize
    func_id : int, optional
        Function ID for benchmark functions
        
    Returns:
    --------
    gb_fit : float
        Best fitness value
    gb_sol : numpy array
        Best solution found
    conv_curve : numpy array
        Convergence curve (fitness value per iteration)
    """
    # Definitions
    gb_fit = float('inf')  # Best-so-far fitness
    gb_sol = np.zeros(dim)  # Best-so-far solution
    conv_curve = np.zeros(max_iter)
    
    # Controlling parameters
    n = pop_size  # Initial population size
    n_min = 120  # Minimum population size
    t_cycles = 2  # Number of cycles
    alpha = 0.2  # Convergence rate
    tf = 0.8  # Tradeoff between third and fourth defense mechanisms
    
    # Initialization
    x = initialization(pop_size, dim, ub, lb)  # Initialize the positions of crested porcupines
    t = 0  # Function evaluation counter
    
    # Evaluation
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        if func_id is not None:
            fitness[i] = obj_func(x[i, :], func_id)
        else:
            fitness[i] = obj_func(x[i, :])
            
    # Update the best-so-far solution
    idx = np.argmin(fitness)
    gb_fit = fitness[idx]
    gb_sol = x[idx, :].copy()
    
    # Store personal best position for each porcupine
    xp = x.copy()
    
    # Set optimal fitness value for termination (if known)
    opt = 0  # Can be adjusted for benchmark functions
    
    # Optimization Process of CPO
    while t <= max_iter and gb_fit != opt:
        r2 = np.random.random()
        
        for i in range(pop_size):
            # Generate binary mask for exploration/exploitation decisions
            u1 = np.random.random(dim) > np.random.random()
            
            if np.random.random() < np.random.random():  # Exploration phase
                if np.random.random() < np.random.random():  # First defense mechanism
                    # Calculate y_t
                    y = (x[i, :] + x[np.random.randint(0, pop_size), :]) / 2
                    x[i, :] = x[i, :] + (np.random.randn()) * np.abs(2 * np.random.random() * gb_sol - y)
                else:  # Second defense mechanism
                    y = (x[i, :] + x[np.random.randint(0, pop_size), :]) / 2
                    x[i, :] = (u1) * x[i, :] + (1 - u1) * (y + np.random.random() * (x[np.random.randint(0, pop_size), :] - x[np.random.randint(0, pop_size), :]))
            else:  # Exploitation phase
                yt = 2 * np.random.random() * (1 - t / max_iter) ** (t / max_iter)
                u2 = np.random.random(dim) < 0.5 * 2 - 1
                s = np.random.random() * u2
                
                if np.random.random() < tf:  # Third defense mechanism
                    st = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))  # Add eps to avoid division by zero
                    s = s * yt * st
                    x[i, :] = (1 - u1) * x[i, :] + u1 * (x[np.random.randint(0, pop_size), :] + 
                              st * (x[np.random.randint(0, pop_size), :] - x[np.random.randint(0, pop_size), :]) - s)
                else:  # Fourth defense mechanism
                    mt = np.exp(fitness[i] / (np.sum(fitness) + np.finfo(float).eps))
                    vt = x[i, :]
                    vtp = x[np.random.randint(0, pop_size), :]
                    ft = np.random.random(dim) * (mt * (-vt + vtp))
                    s = s * yt * ft
                    x[i, :] = (gb_sol + (alpha * (1 - r2) + r2) * (u2 * gb_sol - x[i, :])) - s
            
            # Boundary check - return search agents to the search space if they exceed bounds
            x[i, :] = np.clip(x[i, :], lb, ub)
            
            # If still outside bounds after clipping, reinitialize position randomly
            out_of_bounds = np.logical_or(x[i, :] < lb, x[i, :] > ub)
            if np.any(out_of_bounds):
                x[i, out_of_bounds] = lb[out_of_bounds] + np.random.random(np.sum(out_of_bounds)) * (ub[out_of_bounds] - lb[out_of_bounds])
            
            # Calculate fitness for the new position
            if func_id is not None:
                new_fitness = obj_func(x[i, :], func_id)
            else:
                new_fitness = obj_func(x[i, :])
            
            # Update personal best
            if fitness[i] < new_fitness:
                x[i, :] = xp[i, :].copy()  # Return to previous position
            else:
                xp[i, :] = x[i, :].copy()
                fitness[i] = new_fitness
                
                # Update global best
                if fitness[i] <= gb_fit:
                    gb_sol = x[i, :].copy()
                    gb_fit = fitness[i]
            
            t += 1  # Move to the next evaluation
            if t > max_iter:
                break
                
            conv_curve[t-1] = gb_fit
        
        # Update population size
        pop_size = int(n_min + (n - n_min) * (1 - (t % (max_iter / t_cycles)) / (max_iter / t_cycles)))
    
    return gb_fit, gb_sol, conv_curve


def initialization(pop_size, dim, ub, lb):
    """
    Initialize the population of search agents
    
    Parameters:
    -----------
    pop_size : int
        Population size
    dim : int
        Problem dimension
    ub : numpy array
        Upper bounds for each dimension
    lb : numpy array
        Lower bounds for each dimension
        
    Returns:
    --------
    positions : numpy array
        Initial positions of search agents
    """
    # If all variables have the same bounds
    if np.size(ub) == 1:
        positions = np.random.random((pop_size, dim)) * (ub - lb) + lb
    else:
        # If each variable has different bounds
        positions = np.zeros((pop_size, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.random(pop_size) * (ub_i - lb_i) + lb_i
            
    return positions 