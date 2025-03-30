import numpy as np
import matplotlib.pyplot as plt
import time
from cec_functions_wrapper import get_cec_function
from cec_functions_data import CECData
from benchmark_functions import get_function_details, load_shift_data, load_rotation_matrix

def test_cec_function(cec_year, func_id, dim=10):
    """
    Test the CEC function interface
    
    Parameters:
    -----------
    cec_year : int
        CEC benchmark year (1 for 2014, 2 for 2017, 3 for 2020, 4 for 2022)
    func_id : int
        Function ID
    dim : int
        Problem dimension
    """
    # Print test information
    year_map = {1: 2014, 2: 2017, 3: 2020, 4: 2022}
    actual_year = year_map.get(cec_year, 2017)
    print(f"\nTesting CEC-{actual_year} Function {func_id} with dimension {dim}")
    
    # Get function details
    lb, ub, dim = get_function_details(func_id, cec_year)
    print(f"Bounds: [{lb[0]}, {ub[0]}]")
    
    # Test data loading
    shift_data = load_shift_data(func_id, dim, cec_year)
    print(f"Shift data (first 5 elements): {shift_data[:5]}")
    
    rotation_matrix = load_rotation_matrix(func_id, dim, cec_year)
    print(f"Rotation matrix shape: {rotation_matrix.shape}")
    
    # Get the CEC function
    cec_func = get_cec_function(cec_year)
    
    # Test function with origin
    origin = np.zeros(dim)
    try:
        value_at_origin = cec_func(origin, func_id)
        print(f"Function value at origin: {value_at_origin}")
    except Exception as e:
        print(f"Error evaluating function at origin: {e}")
    
    # Test function with random point
    x = lb + np.random.random(dim) * (ub - lb)
    try:
        value_at_random = cec_func(x, func_id)
        print(f"Function value at random point: {value_at_random}")
    except Exception as e:
        print(f"Error evaluating function at random point: {e}")
    
    # Test with optimal point (shifted origin)
    try:
        value_at_optimal = cec_func(shift_data, func_id)
        print(f"Function value at optimal point: {value_at_optimal}")
    except Exception as e:
        print(f"Error evaluating function at optimal point: {e}")
    
    return True

def test_performance(cec_year, func_id, dim=10, num_samples=1000):
    """
    Test the performance of the CEC function
    
    Parameters:
    -----------
    cec_year : int
        CEC benchmark year (1 for 2014, 2 for 2017, 3 for 2020, 4 for 2022)
    func_id : int
        Function ID
    dim : int
        Problem dimension
    num_samples : int
        Number of evaluations for performance test
    """
    # Get function details
    lb, ub, dim = get_function_details(func_id, cec_year)
    
    # Get the CEC function
    cec_func = get_cec_function(cec_year)
    
    # Generate random points
    points = lb + np.random.random((num_samples, dim)) * (ub - lb)
    
    # Measure evaluation time
    start_time = time.time()
    
    for i in range(num_samples):
        value = cec_func(points[i], func_id)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_samples
    
    print(f"Average evaluation time: {avg_time * 1000:.4f} ms per evaluation")
    
    return avg_time

if __name__ == "__main__":
    # Test CEC-2017 functions
    for func_id in [1, 3, 5, 10]:
        test_cec_function(2, func_id)
        test_performance(2, func_id)
    
    print("\nCEC function testing completed!") 