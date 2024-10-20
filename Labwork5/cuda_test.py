import numpy as np
import matplotlib.pyplot as plt
from numba import cuda

# Check if CUDA is available
print(cuda.is_available())  # Should return True if CUDA is available