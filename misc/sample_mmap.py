import numpy as np
import mmap
import struct
import tracemalloc
import os

test_fname = "test_data.tmp"

# Shape of whole array to be written to file
arr_shape = (100000, 2)

# Number of elements to load from memory
elements_to_load = [10, 100, 500, 1000, 5000]

# Write array to file (row-by-row/line-by-line)
with open(test_fname, 'wb') as f: 
    for _ in range(arr_shape[0]):
        data = np.random.rand(arr_shape[1]).astype(np.double)
        f.write(data.tobytes())
   
# Test max memory usage when loading different numbers of elements from disk
for num in elements_to_load:
    
    # Start memory usage monitoring
    tracemalloc.start()  

    with open(test_fname, 'r+b') as f:
        # Create memory map - array not loaded to memory
        mm = mmap.mmap(f.fileno(), 0)
        
        # Create memory-mapped array from buffer - array not loaded to memory
        mm_arr = np.frombuffer(mm, dtype=np.double)
        
        mm_arr = np.reshape(mm_arr, arr_shape)
        
        # Load some number of elements from buffer - array not loaded to memory
        new_arr = mm_arr[:num]
        
        # Perform some calculation on subset array - subset of array loaded to memory
        new_arr.mean()

        # tracemalloc.get_traced_memory() returns (current usage, peak usage) in memory blocks
        print(num, tracemalloc.get_traced_memory())
    
    # End memory usage monitoring
    tracemalloc.stop()

os.remove(test_fname)
    