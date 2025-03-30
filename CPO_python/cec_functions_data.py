import os
import numpy as np
import shutil

class CECData:
    """Class to manage CEC benchmark data files"""
    
    def __init__(self, cec_year=2017):
        """
        Initialize the CEC data manager
        
        Parameters:
        -----------
        cec_year : int
            CEC benchmark year (2014, 2017, 2020, or 2022)
        """
        self.cec_year = cec_year
        self.data_dir = f"input_data{str(cec_year)[-2:]}"
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure that the data directory exists and contains the required files"""
        data_path = os.path.join(self.root_dir, self.data_dir)
        
        # Check if the directory exists
        if not os.path.exists(data_path):
            print(f"Creating directory: {data_path}")
            os.makedirs(data_path, exist_ok=True)
        
        # Check if the required files exist
        original_data_path = os.path.join(os.path.dirname(self.root_dir), self.data_dir)
        if os.path.exists(original_data_path):
            # Copy files from the original directory
            self._copy_data_files(original_data_path, data_path)
        else:
            print(f"Warning: Original data directory {original_data_path} not found.")
            print("Data files need to be provided manually.")
    
    def _copy_data_files(self, src_dir, dst_dir):
        """Copy data files from source to destination directory"""
        try:
            file_count = 0
            for file_name in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file_name)
                dst_file = os.path.join(dst_dir, file_name)
                
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                    file_count += 1
            
            if file_count > 0:
                print(f"Copied {file_count} data files to {dst_dir}")
            
        except Exception as e:
            print(f"Error copying data files: {e}")
    
    def load_shift_data(self, func_id, dim):
        """
        Load shift data for a CEC function
        
        Parameters:
        -----------
        func_id : int
            Function ID
        dim : int
            Problem dimension
            
        Returns:
        --------
        numpy array
            Shift data vector
        """
        try:
            shift_file = os.path.join(self.root_dir, self.data_dir, f"shift_data_{func_id}.txt")
            
            if not os.path.exists(shift_file):
                print(f"Warning: Shift data file not found: {shift_file}")
                return np.zeros(dim)
            
            # Read shift data from file
            with open(shift_file, 'r') as f:
                shift_data = []
                for i, line in enumerate(f):
                    if i >= dim:
                        break
                    shift_data.append(float(line.strip()))
            
            # Ensure the shift vector has the correct dimension
            if len(shift_data) < dim:
                shift_data.extend([0.0] * (dim - len(shift_data)))
            
            return np.array(shift_data[:dim])
            
        except Exception as e:
            print(f"Error loading shift data: {e}")
            return np.zeros(dim)
    
    def load_rotation_matrix(self, func_id, dim):
        """
        Load rotation matrix for a CEC function
        
        Parameters:
        -----------
        func_id : int
            Function ID
        dim : int
            Problem dimension
            
        Returns:
        --------
        numpy array
            Rotation matrix
        """
        try:
            matrix_file = os.path.join(self.root_dir, self.data_dir, f"M_{func_id}_D{dim}.txt")
            
            if not os.path.exists(matrix_file):
                print(f"Warning: Rotation matrix file not found: {matrix_file}")
                return np.eye(dim)
            
            # Read rotation matrix from file
            matrix = np.zeros((dim, dim))
            with open(matrix_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= dim:
                        break
                    values = line.strip().split()
                    for j, val in enumerate(values):
                        if j >= dim:
                            break
                        matrix[i, j] = float(val)
            
            return matrix
            
        except Exception as e:
            print(f"Error loading rotation matrix: {e}")
            return np.eye(dim)

    def load_shuffle_data(self, func_id, dim):
        """
        Load shuffle data for a CEC function
        
        Parameters:
        -----------
        func_id : int
            Function ID
        dim : int
            Problem dimension
            
        Returns:
        --------
        numpy array
            Shuffle data vector
        """
        try:
            shuffle_file = os.path.join(self.root_dir, self.data_dir, f"shuffle_data_{func_id}_D{dim}.txt")
            
            if not os.path.exists(shuffle_file):
                print(f"Warning: Shuffle data file not found: {shuffle_file}")
                return np.arange(1, dim + 1)
            
            # Read shuffle data from file
            with open(shuffle_file, 'r') as f:
                shuffle_data = []
                for i, line in enumerate(f):
                    if i >= dim:
                        break
                    shuffle_data.append(int(line.strip()))
            
            # Ensure the shuffle vector has the correct dimension
            if len(shuffle_data) < dim:
                shuffle_data.extend(list(range(len(shuffle_data) + 1, dim + 1)))
            
            return np.array(shuffle_data[:dim])
            
        except Exception as e:
            print(f"Error loading shuffle data: {e}")
            return np.arange(1, dim + 1) 