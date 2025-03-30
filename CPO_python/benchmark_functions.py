import numpy as np
import os
import warnings

# Import the new CEC wrapper and data manager
from cec_functions_wrapper import get_cec_function
from cec_functions_data import CECData

def get_cec_function(cec_year):
    """
    Return the appropriate CEC benchmark function based on the year
    
    Parameters:
    -----------
    cec_year : int
        CEC benchmark year: 1 for 2014, 2 for 2017, 3 for 2020, 4 for 2022
        
    Returns:
    --------
    function
        The benchmark function
    """
    warnings.warn("This is a placeholder for the CEC benchmark functions. In Python, you need to install or implement the CEC benchmark functions separately.")
    
    # Placeholder implementation for CEC benchmark functions
    # In a real implementation, you would use specialized libraries or Python implementations of CEC functions
    def cec_function(x, func_id):
        # This is a placeholder that returns a simple function value
        # In real implementation, this would call the actual CEC benchmark function
        
        # For demo purposes, we implement some basic test functions
        if func_id == 1:  # Sphere function
            return np.sum(x**2)
        elif func_id == 2:  # Rosenbrock function
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)
        elif func_id == 3:  # Rastrigin function
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        elif func_id == 4:  # Griewank function
            return 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        elif func_id == 5:  # Ackley function
            return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x))) - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.e
        else:
            # For other functions, just return a simple function
            return np.sum(x**2)
    
    return cec_function


def get_function_details(func_id, cec_year=2):
    """
    Get the details (lower bound, upper bound, dimension) for a specific benchmark function
    
    Parameters:
    -----------
    func_id : int
        Function ID
    cec_year : int
        CEC benchmark year (default: 2 for CEC-2017)
        
    Returns:
    --------
    lb : numpy array
        Lower bounds for each dimension
    ub : numpy array
        Upper bounds for each dimension
    dim : int
        Problem dimension
    """
    # Default dimension
    dim = 10
    
    # For most CEC benchmark functions, bounds are [-100, 100] for all dimensions
    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)
    
    # Special cases based on CEC year
    if cec_year == 2:  # CEC-2017
        if func_id == 11:
            lb = -600 * np.ones(dim)
            ub = 600 * np.ones(dim)
    elif cec_year == 1:  # CEC-2014
        if func_id == 11:
            lb = -600 * np.ones(dim)
            ub = 600 * np.ones(dim)
    elif cec_year == 3:  # CEC-2020
        pass  # No special cases for CEC-2020
    elif cec_year == 4:  # CEC-2022
        pass  # No special cases for CEC-2022
    
    return lb, ub, dim


def load_shift_data(func_id, dim, cec_year=2):
    """
    Load shift data for CEC benchmark functions
    
    Parameters:
    -----------
    func_id : int
        Function ID
    dim : int
        Problem dimension
    cec_year : int
        CEC benchmark year (default: 2 for CEC-2017)
        
    Returns:
    --------
    shift_data : numpy array
        Shift data for the function
    """
    # Map the cec_year to the actual year
    year_map = {1: 2014, 2: 2017, 3: 2020, 4: 2022}
    actual_year = year_map.get(cec_year, 2017)
    
    # Initialize the CEC data manager and load the shift data
    cec_data = CECData(actual_year)
    return cec_data.load_shift_data(func_id, dim)


def load_rotation_matrix(func_id, dim, cec_year=2):
    """
    Load rotation matrix for CEC benchmark functions
    
    Parameters:
    -----------
    func_id : int
        Function ID
    dim : int
        Problem dimension
    cec_year : int
        CEC benchmark year (default: 2 for CEC-2017)
        
    Returns:
    --------
    rotation_matrix : numpy array
        Rotation matrix for the function
    """
    # Map the cec_year to the actual year
    year_map = {1: 2014, 2: 2017, 3: 2020, 4: 2022}
    actual_year = year_map.get(cec_year, 2017)
    
    # Initialize the CEC data manager and load the rotation matrix
    cec_data = CECData(actual_year)
    return cec_data.load_rotation_matrix(func_id, dim)


def load_shuffle_data(func_id, dim, cec_year=2):
    """
    Load shuffle data for CEC benchmark functions
    
    Parameters:
    -----------
    func_id : int
        Function ID
    dim : int
        Problem dimension
    cec_year : int
        CEC benchmark year (default: 2 for CEC-2017)
        
    Returns:
    --------
    shuffle_data : numpy array
        Shuffle data for the function
    """
    # Map the cec_year to the actual year
    year_map = {1: 2014, 2: 2017, 3: 2020, 4: 2022}
    actual_year = year_map.get(cec_year, 2017)
    
    # Initialize the CEC data manager and load the shuffle data
    cec_data = CECData(actual_year)
    return cec_data.load_shuffle_data(func_id, dim) 