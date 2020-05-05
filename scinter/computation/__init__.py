import os
import numpy as np
import time
from ruamel.yaml import YAML
import warnings
import ctypes

from .defined_computations import defined_computations
        
class computation(defined_computations):
    
    def __init__(self,dict_paths,name_save,dict_base=None):
        self.time_start = time.time()
        self.yaml = YAML(typ='safe')
        self.dict_paths = dict_paths
        self.path_computations = dict_paths['computations']
        self.path_data = dict_paths['data']
        self.path_c_source = dict_paths['c_source']
        self.name_computation = name_save
        self.path_computation = os.path.join(self.path_computations,name_save)
        if not dict_base==None:
            # - make frequently used quantities available
            self.N_p = dict_base['N_p']
            self.N_t = dict_base['N_t']
            self.N_nu = dict_base['N_nu']
            self.t = dict_base['t']
            self.nu = dict_base['nu']
            self.t_half = dict_base['t_half']
            self.nu_half = dict_base['nu_half']
            self.delta_t = dict_base['delta_t']
            self.delta_nu = dict_base['delta_nu']
        # - log user defined specifications
        self.file_specs = os.path.join(self.path_computation,"specs_{0}.yaml".format(self.name_computation))
        if not os.path.exists(self.file_specs):
            self.dict_specs = {}
        else:
            with open(self.file_specs,'r') as readfile:
                self.dict_specs =self.yaml.load(readfile)
        
    def compute(self):
        #perform customary main part of the computation
        print("Computing {0} ...".format(self.name_computation))
        exec("self.{0}()".format(self.name_computation))
        # - log potentially created specifications
        with open(self.file_specs,'w') as writefile:
            self.yaml.dump(self.dict_specs,writefile)
        time_current = time.time()-self.time_start
        print("{0}s: Successfully computed {1}.".format(time_current,self.name_computation)) 
        
    def load_result(self,list_names):
        #load a list of requested results from previous computations
        results = []
        for data_name in list_names:
            file_result = os.path.join(self.path_computation,"{0}.npy".format(data_name))
            # - load the file or tell what is missing
            try:
                result = np.load(file_result)
            except IOError:
                warnings.warn("The file '{0}.npy' cannot be read! Did you forget to run 'compute {1}'?".format(data_name,self.name_computation))
                result = None
            results.append(result)
        return results
    
    def _warning(self,message,category = UserWarning,filename = '',lineno = -1):
        warn_message = '/!\\ ' + str(message)
        print(warn_message)
    
    def _add_result(self,data_object,data_name):
        #save a data object resulting from the computation
        file_save = os.path.join(self.path_computation,"{0}.npy".format(data_name))
        np.save(file_save,data_object)
        
    def _add_specification(self,spec_name,spec_value):
        #log standard value if not yet existent and read current value
        if spec_name not in self.dict_specs:
            self.dict_specs.update({spec_name:spec_value})
        return self.dict_specs[spec_name]
    
    def _load_base_file(self,file_name):
        file_base = os.path.join(self.path_data,"{0}.npy".format(file_name))
        return np.load(file_base)
    
    def _load_c_lib(self,file_name):
        file_c = os.path.join(self.path_c_source,"{0}.so".format(file_name))
        return ctypes.CDLL(file_c)