import os
import numpy as np
import time
from ruamel.yaml import YAML
import warnings
        
#similar class as computation; needed to load previous results within another computation
class result_source():
    
    def __init__(self,dict_paths,name_save):
        self.time_start = time.time()
        self.yaml = YAML(typ='safe')
        self.dict_paths = dict_paths
        self.path_computations = dict_paths['computations']
        self.path_data = dict_paths['data']
        self.path_c_source = dict_paths['c_source']
        self.name_computation = name_save
        self.path_computation = os.path.join(self.path_computations,name_save)
        # - log user defined specifications
        self.file_specs = os.path.join(self.path_computation,"specs_{0}.yaml".format(self.name_computation))
        if not os.path.exists(self.file_specs):
            self.dict_specs = {}
        else:
            with open(self.file_specs,'r') as readfile:
                self.dict_specs =self.yaml.load(readfile)
        
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