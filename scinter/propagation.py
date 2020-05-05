import warnings
import os
import numpy as np
import math

class propagation:
    #Useful constants
    ly =  9460730472580800. #m
    au =      149597870700. #m
    pc = 648000./math.pi*au #m
    LightSpeed = 299792458. #m/s
    r_terra = 6371.*1000. #m
    mas = 1./1000.*math.pi/648000. #radians
    
    def __init__(self,path_data,dict_params):
        #read in specifications from file, this should contain global specification
        #about the dynamic spectrum to be computed resp. loaded
        #and specifically marked model specifications
        
        #construct dynamic spectrum container or better not to save memory
        
        #save specifications
        
        #read arguments
        self.path_data = path_data
        self.dict_params = dict_params
        self.file_DynSpec = os.path.join(self.path_data,"DynSpec.npy")
        self.file_flags = os.path.join(self.path_data,"flags.npy")
        self.file_positions = os.path.join(self.path_data,"positions.npy")
        
        #read parameters
        self.N_t = dict_params["N_t"]
        self.N_nu = dict_params["N_nu"]
        self.t_min = dict_params["t_min"]
        self.nu_min = dict_params["nu_min"]
        self.bandwidth = dict_params["bandwidth"]
        self.timespan = dict_params["timespan"]
        
    def simulate(self):
        #override in model specific inheritor to return and save dynamic spectrum
        # - take this function as a reference
        warnings.warn("The simulate() method is not specified!")
        
        # define the number of telescopes
        N_p = 4
        
        #compute output quantities
        DynSpec = np.zeros((N_p,N_p,self.N_t,self.N_nu),dtype=np.complex)
        flags = np.zeros((self.N_t,self.N_nu),dtype=bool)
        positions = np.zeros((N_p,self.N_t,3),dtype=float) #in m (x,y,z)
        
        #update the dictionary if data could not be formed as requested
        self.dict_params.update({"N_p":N_p})
        
        #save them
        np.save(self.file_DynSpec,DynSpec)
        np.save(self.file_flags,flags)
        np.save(self.file_positions,positions)
        
        return self.dict_params