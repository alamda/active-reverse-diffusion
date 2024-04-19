import numpy as np
import torch
import mmap
import os
import gc
import pickle

class DataHandler:
    def __init__(self, name="data_handler",
                 fname=None):
        self.name = name
        
        self.fname = None
        self.set_fname(fname=fname)

    def set_fname(self, fname=None):
        try:
            if fname is not None:
                self.fname = fname
            elif self.fname is None:
                raise TypeError
        except TypeError:
            print("No file name provided for data object")
            
    def create_new_file(self, fname=None, overwrite=True):
        if fname is not None:
            self.set_fname(fname=fname)
        
        if (not os.path.isfile(self.fname)) or (overwrite is True):
            with open(self.fname, 'wb'):
                os.utime(self.fname, None)                
                
class DiffusionSampleHandler(DataHandler):
    def __init__(self, name="diffusion_sample_handler", 
                 fname=None,
                 sample_size=None,
                 sample_dim=None):
        
        super().__init__(name=name, fname=fname)

        self.sample_size = sample_size
        self.sample_dim = sample_dim
        
                
        self.mmap = None
        self.mmap_arr = None
        self.mmap_tensor = None 
        # ^ Actually a list of tensors in the case of 
        # forward/reverse diffusion trajectories
                
    def write_tensor_to_file(self, tensor=None, fname=None):
        if (not os.path.isfile(self.fname)) or (fname is not None) :
            self.create_new_file(fname=fname)
        
        if tensor is not None:
            if isinstance(tensor, list):
                tensor = torch.DoubleTensor(tensor)
            with open(self.fname, "ab") as f:
                f.write(tensor.detach().numpy().tobytes())
                
            del tensor
        gc.collect()
    
    def mmap_tensor_from_file(self, fname=None, shape=None):
        self.set_fname(fname=fname)
        
        with open(self.fname, "r+b") as f:  
            self.mmap = mmap.mmap(f.fileno(), 0)
            
            self.mmap_arr = np.frombuffer(self.mmap, dtype=np.double)
            
            try:
                if (shape is None) and \
                    (None not in (self.sample_size, self.sample_dim)):
                    
                    shape = (-1, self.sample_size, self.sample_dim)
                    
                    self.mmap_tensor = torch.from_numpy(
                        np.reshape(self.mmap_arr, shape))
                    
                    if self.mmap_tensor.shape[0] > 1:
                        self.mmap_tensor = [x for x in self.mmap_tensor]
                    else:
                        self.mmap_tensor = \
                            torch.reshape(self.mmap_tensor, 
                                          (self.sample_size, self.sample_dim))
                    
                    return self.mmap_tensor
                else:
                    raise TypeError
            except TypeError:
                print("Need to provide shape for tensor \
                       (num diff steps, sample size, sample dim)")
        
    def close_mmap(self):
        if self.mmap is not None:
            del self.mmap_arr
            del self.mmap_tensor
            gc.collect()
            
            self.mmap.close()
            self.mmap = None
            
class ModelHandler(DataHandler):
    def __init__(self, name="model_handler",
                 fname=None):
        
        super().__init__(name=name, fname=fname)
        
        self.models = []
        
    def clear_models(self):
        self.models = []
        
    def write_model_to_file(self, model=None, fname=None):
        if (not os.path.isfile(self.fname)) or (fname is not None) :
            self.create_new_file(fname=fname)
            
        if None not in (model):
            with open(self.fname, "wb") as f:
                self.models.append(model)
                pickle.dump(self.models, f)
                
    def load_models(self, fname=None):
        self.set_fname(fname=fname)
        
        with open(self.fname, "rb") as f:
            self.models = pickle.load(f)
            
            return self.models