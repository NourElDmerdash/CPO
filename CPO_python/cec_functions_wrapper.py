import os
import numpy as np
import ctypes
from ctypes import c_int, c_double, POINTER, cdll
import platform

class CECFunctions:
    """Wrapper for the CEC benchmark functions implemented in C++"""
    
    def __init__(self, cec_year=2017):
        """
        Initialize the CEC functions wrapper
        
        Parameters:
        -----------
        cec_year : int
            CEC benchmark year (2014, 2017, 2020, or 2022)
        """
        self.cec_year = cec_year
        self.lib = None
        self.func = None
        self._load_library()
        
    def _load_library(self):
        """Load the appropriate shared library for the CEC functions"""
        # Define library names based on the operating system
        system = platform.system()
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent directory
        
        if system == 'Windows':
            lib_name = f'cec{str(self.cec_year)[-2:]}_func.dll'
        elif system == 'Darwin':  # macOS
            lib_name = f'libcec{str(self.cec_year)[-2:]}_func.dylib'
        else:  # Linux and others
            lib_name = f'libcec{str(self.cec_year)[-2:]}_func.so'
            
        lib_path = os.path.join(root_dir, lib_name)
        
        # Check if we need to compile the library
        if not os.path.exists(lib_path):
            self._compile_library()
        
        try:
            # Load the library
            self.lib = cdll.LoadLibrary(lib_path)
            
            # Define the function signature
            self.lib.cec_function.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int]
            self.lib.cec_function.restype = None
            
            print(f"CEC {self.cec_year} library loaded successfully")
        except Exception as e:
            print(f"Error loading library: {e}")
            print("Using placeholder functions instead")
            self.lib = None
    
    def _compile_library(self):
        """Compile the C++ code into a shared library"""
        print(f"Compiling CEC {self.cec_year} library...")
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cpp_file = os.path.join(root_dir, f'cec{str(self.cec_year)[-2:]}_func.cpp')
        
        if not os.path.exists(cpp_file):
            print(f"C++ source file {cpp_file} not found.")
            return False
            
        # Create a simplified C wrapper for the CEC functions
        wrapper_code = self._generate_wrapper_code()
        wrapper_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec_wrapper.cpp")
        
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        # Compile based on the platform
        system = platform.system()
        
        if system == 'Windows':
            os.system(f'g++ -shared -o libcec{str(self.cec_year)[-2:]}_func.dll {wrapper_file} {cpp_file} -fPIC')
        elif system == 'Darwin':  # macOS
            os.system(f'g++ -shared -o libcec{str(self.cec_year)[-2:]}_func.dylib {wrapper_file} {cpp_file} -fPIC')
        else:  # Linux and others
            os.system(f'g++ -shared -o libcec{str(self.cec_year)[-2:]}_func.so {wrapper_file} {cpp_file} -fPIC')
            
        print("Compilation complete")
        return True
    
    def _generate_wrapper_code(self):
        """Generate a C wrapper code for the CEC functions"""
        wrapper_code = """
        #include <cstdlib>
        #include <cstdio>
        
        // Forward declarations for the CEC functions
        extern "C" {
            void cec%s_test_func(double*, double*, int, int, int);
            
            // Wrapper function with a simpler interface for ctypes
            void cec_function(double* x, double* f, int dim, int func_num) {
                cec%s_test_func(x, f, dim, 1, func_num);
            }
        }
        """ % (str(self.cec_year)[-2:], str(self.cec_year)[-2:])
        
        return wrapper_code
    
    def evaluate(self, x, func_id):
        """
        Evaluate the CEC function
        
        Parameters:
        -----------
        x : numpy array
            The solution to evaluate
        func_id : int
            Function ID
            
        Returns:
        --------
        float
            Function value
        """
        if self.lib is None:
            return self._placeholder_function(x, func_id)
            
        # Convert inputs to the appropriate C types
        dim = len(x)
        x_array = (c_double * dim)(*x)
        f_array = (c_double * 1)()
        
        # Call the C function
        self.lib.cec_function(x_array, f_array, c_int(dim), c_int(func_id))
        
        return f_array[0]
    
    def _placeholder_function(self, x, func_id):
        """Placeholder implementation of standard test functions"""
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


def get_cec_function(cec_year):
    """
    Get the CEC benchmark function for a specific year
    
    Parameters:
    -----------
    cec_year : int
        CEC benchmark year (1 for 2014, 2 for 2017, 3 for 2020, 4 for 2022)
        
    Returns:
    --------
    function
        The benchmark function
    """
    # Map the cec_year to the actual year
    year_map = {1: 2014, 2: 2017, 3: 2020, 4: 2022}
    actual_year = year_map.get(cec_year, 2017)
    
    # Initialize the CEC functions wrapper
    cec_functions = CECFunctions(actual_year)
    
    # Return the evaluation function
    return cec_functions.evaluate 