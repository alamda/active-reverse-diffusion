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

# Write array or random numbers to file (row-by-row/line-by-line)
with open(test_fname, 'wb') as f: 
    for _ in range(arr_shape[0]):
        data = np.random.rand(arr_shape[1]).astype(np.double)
        f.write(data.tobytes())
   
# Test max memory usage when loading different numbers of elements from disk
for num in elements_to_load:
    
    # Start memory usage monitoring
    tracemalloc.start()  

    with open(test_fname, 'r+b') as f:
        # Create memory map
        # Array not yet loaded to memory
        mm = mmap.mmap(f.fileno(), 0)
        
        # Create memory-mapped array from buffer
        # Array not yet loaded to memory
        mm_arr = np.frombuffer(mm, dtype=np.double)
        
        mm_arr = np.reshape(mm_arr, arr_shape)
        
        # Load some number of elements from buffer
        # Array not loaded to memory, only memory mapped
        new_arr = mm_arr[:num]
        
        # Perform some calculation on subset array
        # Subset of array loaded to memory
        new_arr.mean()

        # tracemalloc.get_traced_memory(): (current usage, peak usage) in memory blocks
        print(num, tracemalloc.get_traced_memory())
        
        # Change values in loaded portion of the array to 0s
        mm_arr[:num] = np.zeros(new_arr.shape)
       
        # Write updated buffer information to disk/file
        mm.flush()
    
    # End memory usage monitoring
    tracemalloc.stop()
    
# Open file again to check that the original values were successfully changed to 0s 
with open(test_fname, 'r+b') as f:
    mm = mmap.mmap(f.fileno(), 0)
    
    mm_arr = np.frombuffer(mm, dtype=np.double)
    mm_arr = np.reshape(mm_arr, arr_shape)
    
    print(mm_arr[:10])

os.remove(test_fname)
    