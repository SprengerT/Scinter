import os
from ruamel.yaml import YAML
#import warnings
import shutil
import numpy as np
from numpy import newaxis as na
from scipy import interpolate
import progressbar
import time
import math
import scipy
from scipy import ndimage
from scipy import signal
import random as rd
from scipy.optimize import curve_fit
import imageio
from skimage.transform import hough_line, hough_line_peaks
from skimage.measure import block_reduce
import skimage
import skimage.morphology as morphology
from skimage.restoration import denoise_bilateral, denoise_wavelet
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
from cv2 import fastNlMeansDenoising
from numpy.ctypeslib import ndpointer
import ctypes

from scinter.computation import computation as scinter_computation
from scinter.plot import set_figure
from scinter.plot import plot as scinter_plot
from scinter.defined_analyses import defined_analyses

class Scintillation(defined_analyses):
    #Useful constants
    ly =  9460730472580800. #m
    au =      149597870700. #m
    pc = 648000./math.pi*au #m
    LightSpeed = 299792458. #m/s
    r_terra = 6371.*1000. #m
    mas = 1./1000.*math.pi/648000. #radians
    
    def __init__(self,name_data):
        #load the given data as an object
        #if not yet existent look for input file specifying source and parameters
        #create the data from stored measurements and alongside placed scripts
        #do not only save the dynamic spectrum but also all underlying parameters
        
        #initiate software tools
        #warnings.showwarning = self._warning
        self.yaml = YAML(typ='safe')

        #read user defined settings
        with open(os.path.join("input","user_settings.yaml"),'r') as readfile:
            self.user_settings =self.yaml.load(readfile)
        
        #check for directories
        self.path_data = os.path.join(os.path.join(self.user_settings["output_path"],"data"),name_data)
        if not os.path.exists(self.path_data):
            os.makedirs(self.path_data)
        self.path_plots = os.path.join(self.path_data,"plots")
        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)
        self.path_computations = os.path.join(self.path_data,"computations")
        if not os.path.exists(self.path_computations):
            os.makedirs(self.path_computations)
        self.path_animations = os.path.join(self.path_data,"animations")
        if not os.path.exists(self.path_animations):
            os.makedirs(self.path_animations)
            
        #locate paths
        self.path_dataset = os.path.join(self.user_settings["output_path"],"input")
        self.path_plot_source = os.path.join("input","plots")
        self.path_c_source = os.path.join("scinter","c_source")
           
        #check for parameter logfiles and load them
        file_input = os.path.join(self.path_dataset,name_data+".yaml")
        file_params = os.path.join(self.path_data,name_data+".yaml")
        if os.path.exists(file_params):
            with open(file_params,'r') as readfile:
                dict_params =self.yaml.load(readfile)
            #warnings.warn("Loading existing parameters instead of input file.")
            self._warning("Loading existing parameters instead of input file.")
        else:
            with open(file_input,'r') as readfile:
                dict_params =self.yaml.load(readfile)   
            with open(file_params,'w') as writefile:
               self.yaml.dump(dict_params,writefile)
                
        #check for the model file
        self.file_model = os.path.join(self.path_data,"model.yaml")
        if not os.path.exists(self.file_model):
            with open(self.file_model,'w') as writefile:
                if "model" in dict_params:
                   self.yaml.dump(dict_params["model"],writefile)
                else:
                   self.yaml.dump({},writefile)
                
        #check for dynamic spectrum and load it
        name_measurement = dict_params["measurement"]
        file_DynSpec = os.path.join(self.path_data,"DynSpec.npy")
        file_flags = os.path.join(self.path_data,"flags.npy")
        if not (os.path.exists(file_DynSpec) and os.path.exists(file_flags)):
            self._warning("Data processed for the first time! Creating new data directory.")
            vars_from_exec = {"self":self,"dict_params":dict_params}
            exec("import measurements.{0}\nmeasurement = measurements.{0}.{0}(self.path_data,dict_params)".format(name_measurement), vars_from_exec)
            measurement = vars_from_exec["measurement"]
            dict_params = measurement.simulate()
            with open(file_params,'w+') as writefile:
               self.yaml.dump(dict_params,writefile)
            
        #create basic objects from the logfile
        # - read input
        self.N_p = dict_params["N_p"]
        self.N_t = dict_params["N_t"]
        self.N_nu = dict_params["N_nu"]
        self.t_min = dict_params["t_min"]
        self.nu_min = dict_params["nu_min"]
        self.bandwidth = dict_params["bandwidth"]
        self.timespan = dict_params["timespan"]
        # - set up coordinate vectors
        if dict_params["flag_load_t_explicit"]:
            self.t = np.array(dict_params["t"])
        else:
            self.t = np.linspace(0.,self.timespan,num=self.N_t,endpoint=True)
        self.nu = np.linspace(self.nu_min,self.nu_min+self.bandwidth,num=self.N_nu,endpoint=True)
        # - deduce useful quantities
        self.t_half = self.t[int(self.N_t/2)]
        self.nu_half = self.nu[int(self.N_nu/2)]
        self.delta_t = self.t[1]-self.t[0]
        self.delta_nu = self.nu[1]-self.nu[0]
        self.nu_max = self.nu[-1]
        self.t_max = self.t[-1]
        # - create dictionaries to share with other classes
        self.dict_paths = {}
        self.dict_paths.update({'computations':self.path_computations,'data':self.path_data,'c_source':self.path_c_source})
        self.dict_base = {}
        self.dict_base.update({'N_p':self.N_p,'N_t':self.N_t,'N_nu':self.N_nu})
        self.dict_base.update({'t':self.t,'nu':self.nu})
        self.dict_base.update({'t_half':self.t_half,'nu_half':self.nu_half})
        self.dict_base.update({'delta_t':self.delta_t,'delta_nu':self.delta_nu})
            
    def _warning(self,message,category = UserWarning,filename = '',lineno = -1):
        warn_message = '/!\\ ' + str(message)
        print(warn_message)
    
    def compute(self,name_save,name_archive,name_load):
        #check path
        path_computation = os.path.join(self.path_computations,name_save)
        if not os.path.exists(path_computation):
            os.makedirs(path_computation)
            
        #archive previous results
        if not name_archive==None:
            path_archive = os.path.join(path_computation,name_archive)
            print("Archiving previous results under {0}".format(path_archive))
            if not os.path.exists(path_archive):
                os.makedirs(path_archive)
            list_objects = os.listdir(path_computation)
            for obj in list_objects:
                path_obj = os.path.join(path_computation,obj)
                if os.path.isfile(path_obj):
                    name,extension = obj.split('.')
                    newname = "{0}_{1}.{2}".format(name,name_archive,extension)
                    file_dest = os.path.join(path_archive,newname)
                    shutil.copy(path_obj,file_dest)
            
        if not name_load==None:
            #load results from archive folder
            path_load = os.path.join(path_computation,name_load)
            print("Loading previous results from {0}".format(path_load))
            if not os.path.exists(path_load):
                self._warning("{0} does not exist! Cannot load archived results.".format(path_load))
                return 0
            list_objects = os.listdir(path_load)
            for obj in list_objects:
                path_obj = os.path.join(path_load,obj)
                name,extension = obj.split("_{0}".format(name_load))
                newname = name + extension
                file_dest = os.path.join(self.path_computations,name_save,newname)
                shutil.copy(path_obj,file_dest)
        else:
            #execute computation
            computation = scinter_computation(self.dict_paths,name_save,dict_base=self.dict_base)
            computation.compute()
            return 1
    
    def plot(self,name_save,name_archive,name_load):
        #load specifications
        file_plot_class = os.path.join(self.path_plot_source,"{0}.yaml".format(name_save))
        
        #check for directories and files
        path_plot = os.path.join(self.path_plots,name_save)
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        file_plot_specs = os.path.join(path_plot,"{0}.yaml".format(name_save))
        if not os.path.exists(file_plot_specs):
            if not os.path.exists(file_plot_class):
                self._warning("The plot {0} is not defined in input/plots! Attempting to use the standard_single_plot instead...".format(name_save))
                file_plot_class = os.path.join(self.path_plot_source,"standard_single_plot.yaml")
                shutil.copyfile(file_plot_class,file_plot_specs)
                with open(file_plot_specs,"r") as readfile:
                    text = readfile.read()
                    text = text.replace('name_of_the_plot',name_save)
                with open(file_plot_specs,"w") as writefile:
                    writefile.write(text)
            else:
                shutil.copyfile(file_plot_class,file_plot_specs)
        else:
            self._warning("The plot {0} already exists and will be overwritten according to the settings found.".format(name_save))
            
        #archive previous results
        if not name_archive==None:
            path_archive = os.path.join(path_plot,name_archive)
            print("Archiving previous results under {0}".format(path_archive))
            if not os.path.exists(path_archive):
                os.makedirs(path_archive)
            list_objects = os.listdir(path_plot)
            for obj in list_objects:
                path_obj = os.path.join(path_plot,obj)
                if os.path.isfile(path_obj):
                    name,extension = obj.split('.')
                    newname = "{0}_{1}.{2}".format(name,name_archive,extension)
                    file_dest = os.path.join(path_archive,newname)
                    shutil.copy(path_obj,file_dest)
            
        if not name_load==None:
            #load results from archive folder
            path_load = os.path.join(path_plot,name_load)
            print("Loading previous results from {0}".format(path_load))
            if not os.path.exists(path_load):
                self._warning("{0} does not exist! Cannot load archived results.")
                return 0
            list_objects = os.listdir(path_load)
            for obj in list_objects:
                path_obj = os.path.join(path_load,obj)
                name,extension = obj.split("_{0}".format(name_load))
                newname = name + extension
                file_dest = os.path.join(path_plot,newname)
                shutil.copy(path_obj,file_dest)
        else:
            #create the plot from the specifications provided
            with open(file_plot_specs,'r') as readfile:
                dict_plot =self.yaml.load(readfile)
                
            #set up matplotlib
            textsize = self._add_specification("textsize",14,dict_plot)
            labelsize = self._add_specification("labelsize",16,dict_plot)
            # - enter specifications
            set_figure(textsize,labelsize)
                
            #set up the canvas
            plt.clf()
            figure, ax = plt.subplots(len(dict_plot["plots"]),len(dict_plot["plots"][0]),figsize=(dict_plot["width"]/dict_plot["dpi"],dict_plot["height"]/dict_plot["dpi"]),dpi=dict_plot["dpi"])
            plt.subplots_adjust(bottom=dict_plot["bottom"],top=dict_plot["top"],left=dict_plot["left"],right=dict_plot["right"],wspace=dict_plot["wspace"],hspace=dict_plot["hspace"])

            #create plot defining lists
            plot_names = []
            tags = []
            axes = []
            if (len(dict_plot["plots"][0])==1 and len(dict_plot["plots"])==1):
                plot_names.append(dict_plot["plots"][0][0])
                tags.append("")
                axes.append(ax)
            elif len(dict_plot["plots"][0])==1:
                for i_row,row in enumerate(dict_plot["plots"]):
                    plot_names.append(row[0])
                    tags.append("_{0}".format(i_row))
                    axes.append(ax[i_row])
            elif len(dict_plot["plots"])==1:
                for i_col,name_plot in enumerate(dict_plot["plots"][0]):
                    plot_names.append(name_plot)
                    tags.append("_{0}".format(i_col))
                    axes.append(ax[i_col])
            else:
                for i_row,row in enumerate(dict_plot["plots"]):
                    for i_col,name_plot in enumerate(row):
                        plot_names.append(name_plot)
                        tags.append("_{0}_{1}".format(i_row,i_col))
                        axes.append(ax[i_row,i_col])
                        
            #draw the subplots
            for i_plot,name_plot in enumerate(plot_names):
                key_specs = "plot_{0}{1}".format(name_plot,tags[i_plot])
                if not key_specs in dict_plot:
                    dict_plot.update({key_specs:{}})
                dict_subplot = dict_plot[key_specs]
                plot = scinter_plot(name_plot,tags[i_plot],dict_subplot)
                list_category = []
                list_data = []
                exec("self.{0}(dict_subplot,list_category,list_data)".format(name_plot))
                # - save intermediate specifications to allow for adaption in case of failed plotting
                with open(file_plot_specs,'w+') as writefile:
                    self.yaml.dump(dict_plot,writefile)
                # - finally plot the data
                plot.plot(list_category,list_data,figure,axes[i_plot])
                dict_plot.update({key_specs:plot.dict_subplot})
            
            #save the plot
            figure.savefig(os.path.join(path_plot,"{0}.png".format(name_save)))
            #figure.savefig(os.path.join(path_plot,"{0}.jpg".format(name_save)))
            plt.clf()
            
            #overwrite dict_plot to incorporate all settings of the subplots
            with open(file_plot_specs,'w+') as writefile:
               self.yaml.dump(dict_plot,writefile)
               
    def animate(self,name_save,name_archive,name_load):
        #load specifications
        file_plot_class = os.path.join(self.path_plot_source,"{0}.yaml".format(name_save))
        
        #check for directories and files
        path_anim = os.path.join(self.path_animations,name_save)
        if not os.path.exists(path_anim):
            os.makedirs(path_anim)
        file_anim_specs = os.path.join(path_anim,"{0}.yaml".format(name_save))
        if not os.path.exists(file_anim_specs):
            if not os.path.exists(file_plot_class):
                self._warning("The plot {0} is not defined in input/plots! Attempting to use the standard_single_plot instead...".format(name_save))
                file_plot_class = os.path.join(self.path_plot_source,"standard_single_plot.yaml")
                shutil.copyfile(file_plot_class,file_anim_specs)
                with open(file_anim_specs,"r") as readfile:
                    text = readfile.read()
                    text = text.replace('name_of_the_plot',name_save)
                with open(file_anim_specs,"w") as writefile:
                    writefile.write(text)
                with open(file_anim_specs,'r') as readfile:
                    dict_plot = self.yaml.load(readfile)
            else:
                with open(file_plot_class,'r') as readfile:
                    dict_plot = self.yaml.load(readfile)
            dict_anim = {}
            dict_anim.update({"iteration": {"N_iter":1,"t_frame":0.2}})
            dict_anim.update({"masterplot":dict_plot})
            with open(file_anim_specs,'w+') as writefile:
               self.yaml.dump(dict_anim,writefile)
        else:
            self._warning("The animation {0} already exists and will be overwritten according to the settings found.".format(name_save))
            
        #archive previous results
        if not name_archive==None:
            path_archive = os.path.join(path_anim,name_archive)
            print("Archiving previous results under {0}".format(path_archive))
            if not os.path.exists(path_archive):
                os.makedirs(path_archive)
            list_objects = os.listdir(path_anim)
            for obj in list_objects:
                path_obj = os.path.join(path_anim,obj)
                if os.path.isfile(path_obj):
                    name,extension = obj.split('.')
                    newname = "{0}_{1}.{2}".format(name,name_archive,extension)
                    file_dest = os.path.join(path_archive,newname)
                    shutil.copy(path_obj,file_dest)
            #MISSING: delete previous files (because for animate these may not all be overwritten)
                    
        if not name_load==None:
            #load results from archive folder
            path_load = os.path.join(path_anim,name_load)
            print("Loading previous results from {0}".format(path_load))
            if not os.path.exists(path_load):
                warnings.warn("{0} does not exist! Cannot load archived results.")
                return 0
            list_objects = os.listdir(path_load)
            for obj in list_objects:
                path_obj = os.path.join(path_load,obj)
                name,extension = obj.split("_{0}".format(name_load))
                newname = name + extension
                file_dest = os.path.join(path_anim,newname)
                shutil.copy(path_obj,file_dest)
        else:
            #create the animation from the specifications provided
            with open(file_anim_specs,'r') as readfile:
                dict_anim = self.yaml.load(readfile)
            dict_plot = dict_anim["masterplot"]
            dict_iter = dict_anim["iteration"]
                
            #iterate over the frames
            N_iter = dict_iter["N_iter"]
            for i_frame in range(N_iter):
                print("Plotting frame {0} ...".format(i_frame))
                
                #set up the canvas
                plt.clf()
                figure, ax = plt.subplots(len(dict_plot["plots"]),len(dict_plot["plots"][0]),figsize=(dict_plot["width"]/dict_plot["dpi"],dict_plot["height"]/dict_plot["dpi"]),dpi=dict_plot["dpi"])
                plt.subplots_adjust(bottom=dict_plot["bottom"],top=dict_plot["top"],left=dict_plot["left"],right=dict_plot["right"],wspace=dict_plot["wspace"],hspace=dict_plot["hspace"])
                
                #create plot defining lists
                plot_names = []
                tags = []
                axes = []
                if (len(dict_plot["plots"][0])==1 and len(dict_plot["plots"])==1):
                    tag = ""
                    key_plot = "plot_{0}{1}".format(dict_plot["plots"][0][0],tag)
                    if key_plot in dict_iter:
                        for key in dict_iter[key_plot]:
                            if key_plot in dict_plot:
                                frame_value = dict_iter[key_plot][key][i_frame]
                                dict_plot[key_plot].update({key:frame_value})
                    plot_names.append(dict_plot["plots"][0][0])
                    tags.append(tag)
                    axes.append(ax)
                elif len(dict_plot["plots"][0])==1:
                    for i_row,row in enumerate(dict_plot["plots"]):
                        tag = "_{0}".format(i_row)
                        key_plot = "plot_{0}{1}".format(row[0],tag)
                        if key_plot in dict_iter:
                            for key in dict_iter[key_plot]:
                                if key_plot in dict_plot:
                                    frame_value = dict_iter[key_plot][key][i_frame]
                                    dict_plot[key_plot].update({key:frame_value})
                        plot_names.append(row[0])
                        tags.append(tag)
                        axes.append(ax[i_row])
                elif len(dict_plot["plots"])==1:
                    for i_col,name_plot in enumerate(dict_plot["plots"][0]):
                        tag = "_{0}".format(i_col)
                        key_plot = "plot_{0}{1}".format(name_plot,tag)
                        if key_plot in dict_iter:
                            for key in dict_iter[key_plot]:
                                if key_plot in dict_plot:
                                    frame_value = dict_iter[key_plot][key][i_frame]
                                    dict_plot[key_plot].update({key:frame_value})
                        plot_names.append(name_plot)
                        tags.append(tag)
                        axes.append(ax[i_col])
                else:
                    for i_row,row in enumerate(dict_plot["plots"]):
                        for i_col,name_plot in enumerate(row):
                            tag = "_{0}_{1}".format(i_row,i_col)
                            key_plot = "plot_{0}{1}".format(name_plot,tag)
                            if key_plot in dict_iter:
                                for key in dict_iter[key_plot]:
                                    if key_plot in dict_plot:
                                        frame_value = dict_iter[key_plot][key][i_frame]
                                        dict_plot[key_plot].update({key:frame_value})
                            plot_names.append(name_plot)
                            tags.append(tag)
                            axes.append(ax[i_row,i_col])
                            
                #draw the subplots
                for i_plot,name_plot in enumerate(plot_names):
                    key_specs = "plot_{0}{1}".format(name_plot,tags[i_plot])
                    if not key_specs in dict_plot:
                        dict_plot.update({key_specs:{}})
                    dict_subplot = dict_plot[key_specs]
                    plot = scinter_plot(name_plot,tags[i_plot],dict_subplot)
                    list_category = []
                    list_data = []
                    exec("self.{0}(dict_subplot,list_category,list_data)".format(name_plot))
                    # - save intermediate specifications to allow for adaption in case of failed plotting
                    dict_anim["masterplot"].update(dict_plot)
                    with open(file_anim_specs,'w+') as writefile:
                       self.yaml.dump(dict_anim,writefile)
                    # - finally plot the data
                    plot.plot(list_category,list_data,figure,axes[i_plot])
                    dict_plot.update({key_specs:plot.dict_subplot})
                
                #save the plot
                figure.savefig(os.path.join(path_anim,"{0}_{1}.png".format(name_save,i_frame)))
                plt.clf()
                
            #overwrite dict_anim to incorporate all settings of the subplots
            dict_anim["masterplot"].update(dict_plot)
            with open(file_anim_specs,'w+') as writefile:
               self.yaml.dump(dict_anim,writefile)
               
            #create the animation
            file_animation = os.path.join(path_anim,"{0}.gif".format(name_save))
            with imageio.get_writer(file_animation, mode='I',duration=dict_iter["t_frame"]) as writer:
                for i_frame in range(N_iter):
                    filename = os.path.join(path_anim,"{0}_{1}.png".format(name_save,i_frame))
                    image = imageio.imread(filename)
                    writer.append_data(image) 
                    
#-----INTERNAL-----
    
    def _add_specification(self,spec_name,spec_value,dict_current):
        #log standard value if not yet existent and read current value
        if spec_name not in dict_current:
            dict_current.update({spec_name:spec_value})
        return dict_current[spec_name]
        
    def _load_add_specification(self,spec_name,spec_value,dict_current):
        """
        use like 'fDs = self._load_add_specification("source_fDs",["thfD_diagram",None,"fDs"],dict_subplot)'
        """
        source_location = self._add_specification(spec_name,spec_value,dict_current)
        source = scinter_computation(self.dict_paths,source_location[0])
        desired, = source.load_result([source_location[2]])
        return desired
    
    def _load_base_file(self,file_name):
        file_base = os.path.join(self.path_data,"{0}.npy".format(file_name))
        return np.load(file_base)
    
    def _load_c_lib(self,file_name):
        file_c = os.path.join(self.path_c_source,"{0}.so".format(file_name))
        return ctypes.CDLL(file_c)
        
                
#-----POSITIONS-----
                
    def plot_earth_positions(self,dict_plot,figure,ax,tag):
        #load and check data
        positions = np.load(os.path.join(self.path_data,"positions.npy"))
        
        #load specifications
        key_sepcs = "plot_earth_positions{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"title":"Projected telescope positions over time [$h$]"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #draw the plot
        ax.set_xlim([-1.*self.r_terra/1000.,1.*self.r_terra/1000.])
        ax.set_ylim([-1.*self.r_terra/1000.,1.*self.r_terra/1000.])
        # - plot the earth
        earth = plt.Circle((0,0), self.r_terra/1000., color='black', fill=False)
        ax.add_artist(earth)
        # - plot the telescopes
        for i_p in range(self.N_p):
            im = ax.scatter(positions[i_p,:,0]/1000.,positions[i_p,:,1]/1000.,c=self.t/3600.,cmap=dict_subplot["cmap"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$x$ [km]")
        ax.set_ylabel(r"$y$ [km]")
        
        print("Successfully plotted positions of telescopes on earth.")
                
#-----DYNAMIC SPECTRUM-----
        
#-----SECONDARY SPECTRUM-----
        
    def plot_secondary_cross_spectrum(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_computation = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_computation,"secondary_cross_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        SecCrossSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #load specifications
        key_sepcs = "plot_secondary_cross_spectrum{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":float(np.max(np.real(SecCrossSpec))),"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"secondary cross spectrum $P$ [$(W/m^2)^2$] (log10)"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        SecSpec = np.abs(SecCrossSpec[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1])
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(SecSpec[np.nonzero(SecSpec)])
        SecSpec[SecSpec == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(SecSpec)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted secondary cross spectrum.")
        
    def plot_secondary_phase(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_computation = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_cross = os.path.join(path_computation,"secondary_cross_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        SecCrossSpec = np.load(file_cross)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #load specifications
        key_sepcs = "plot_secondary_phase{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'bwr',"vmin":-math.pi,"vmax":math.pi,"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"phase of secondary spectrum $\Delta\Phi$ [rad]"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        Phase = np.angle(SecCrossSpec[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1])
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*Phase)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted phase of secondary cross spectrum.")
        
    def plot_Fourier_phase(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_computation = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_phase = os.path.join(path_computation,"Fourier_phase.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        phase = np.load(file_phase)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #load specifications
        key_sepcs = "plot_secondary_phase{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'bwr',"vmin":-math.pi,"vmax":math.pi,"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"phase of secondary spectrum [rad]","f_t_sampling":1,"f_nu_sampling":1}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        Phase = phase[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        f_t_sampling = dict_subplot["f_t_sampling"]
        f_nu_sampling = dict_subplot["f_nu_sampling"]
        def func_add(block,axis=(0,1)):
            result = np.sum(block+math.pi,axis)%(2.*math.pi)-math.pi
            return result
        Phase = block_reduce(Phase, block_size=(f_t_sampling,f_nu_sampling), func=func_add)
        coordinates = np.array([f_t,f_t])
        coordinates = block_reduce(coordinates, block_size=(1,f_t_sampling), func=np.mean, cval=f_t[-1])
        f_t = coordinates[0,:]
        coordinates = np.array([f_nu,f_nu])
        coordinates = block_reduce(coordinates, block_size=(1,f_nu_sampling), func=np.mean, cval=f_nu[-1])
        f_nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*Phase)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted phase of secondary spectrum.")
        
#-----SECONDARY SPECTRUM PHASE ANALYSIS-----
        
    def compute_FFT_delta_phi(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        
        #define files to compute
        file_delph = os.path.join(path_computation,"delph.npy")
        file_cdelph = os.path.join(path_computation,"cdelph.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
    
        #import c code for discrete Fourier transform
        filename = os.path.join(os.path.join("scinter","c_source"), 'nut_transform.so')
        lib = ctypes.CDLL(filename)
        # lib.omp_set_num_threads (4)   # if you do not want to use all cores
        lib.comp_dft_for_secspec.argtypes = [
            ctypes.c_int,   # ntime
            ctypes.c_int,   # nfreq
            ctypes.c_int,   # nr    Doppler axis
            ctypes.c_double,  # r0
            ctypes.c_double,  # delta r
            ndpointer(dtype=np.float64,
                      flags='CONTIGUOUS', ndim=1),  # freqs [nfreq]
            ndpointer(dtype=np.float64,
                      flags='CONTIGUOUS', ndim=1),  # src [ntime]
            ndpointer(dtype=np.float64,
                      flags='CONTIGUOUS', ndim=2),  # input pow [ntime,nfreq]
            ndpointer(dtype=np.complex128,
                      flags='CONTIGUOUS', ndim=2),  # result [nr,nfreq]
        ]
        
        #create containers
        t0 = self.t[0]
        nu1 = self.nu[0]
        N_nucut = self.N_nu/2
        nu2 = self.nu[N_nucut]
        nuref = self.nu_half
        DynSpec1 = DynSpec[:,:,:,0:N_nucut]
        DynSpec2 = DynSpec[:,:,:,N_nucut:N_nucut*2]
        assert (DynSpec1.shape==DynSpec2.shape)
        SecSpec1 = np.zeros(DynSpec1.shape,dtype=np.complex)
        SecSpec2 = np.zeros(DynSpec2.shape,dtype=np.complex)
        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=N_nucut,endpoint=False)
        delph = np.zeros(SecSpec1.shape,dtype=float)
        cdelph = np.zeros(SecSpec1.shape,dtype=float)
    
        #preparations
        ntime = self.N_t
        nfreq = N_nucut
        r0 = np.fft.fftfreq(ntime)
        delta_r = r0[1] - r0[0]
        src = np.linspace(0, 1, ntime).astype('float64')
        src = np.arange(ntime).astype('float64')
    
        # Common reference freq.
        fscale1 = (self.nu[0:N_nucut] / nuref).astype('float64')
        fscale2 = (self.nu[N_nucut:N_nucut*2] / nuref).astype('float64')
    
        #perform the computation
        time_start = time.time()
        print("Computing phase shift of nut transformed secondary spectrum ...")
        bar = progressbar.ProgressBar(maxval=self.N_p**2, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                bar.update(i_p1*self.N_p+i_p2)
                # - 1: lower frequencies
                # - declare the empty result array:
                SS = np.empty((ntime, nfreq), dtype=np.complex128)
                # - call the DFT:
                lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale1, src, DynSpec1[i_p1,i_p2,:,:].astype('float64'), SS)
                # - flip along time
                SS = SS[::-1]
                # - correct zero point
                SS = np.roll(SS,1,axis=0)
                # - Still need to FFT y axis, should change to pyfftw for memory and
                #   speed improvement
                SS = np.fft.fft(SS, axis=1)
                SS = np.fft.fftshift(SS, axes=1)
                SecSpec1[i_p1,i_p2] = SS
                
                # - 2: higher frequencies
                # - declare the empty result array:
                SS = np.empty((ntime, nfreq), dtype=np.complex128)
                # - call the DFT:
                lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale2, src, DynSpec2[i_p1,i_p2,:,:].astype('float64'), SS)
                # - flip along time
                SS = SS[::-1]
                # - correct zero point
                SS = np.roll(SS,1,axis=0)
                # - Still need to FFT y axis, should change to pyfftw for memory and
                #   speed improvement
                SS = np.fft.fft(SS, axis=1)
                SS = np.fft.fftshift(SS, axes=1)
                SecSpec2[i_p1,i_p2] = SS
        bar.finish()
        # - compute the phases
        diff = SecSpec1*np.conj(SecSpec2)
        delph = np.angle(diff)
        cdelph = np.angle(diff*np.exp(-1.0j*(f_nu[na,na,na,:]+f_t[na,na,:,na]*t0/nuref)*(nu1-nu2)))
        
        #save the results
        np.save(file_delph,delph)
        np.save(file_cdelph,cdelph)
        np.save(file_doppler,f_t)
        np.save(file_delay,f_nu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the FFT phase difference.".format(time_current))
        
    def plot_FFT_phase_shift(self,dict_plot,figure,ax,tag):
        #load and check data
        # - FFT_delta_phi
        path_computation = os.path.join(self.path_computations,"FFT_delta_phi")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute FFT_delta_phi' first!")
            return 0
        file_delph = os.path.join(path_computation,"delph.npy")
        file_cdelph = os.path.join(path_computation,"cdelph.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        delph = np.load(file_delph)
        cdelph = np.load(file_cdelph)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        print("delph: [{0},{1}], cdelph: [{2},{3}]".format(np.min(delph),np.max(delph),np.min(cdelph),np.max(cdelph)))
        
        #load specifications
        key_sepcs = "plot_secondary_phase{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'bwr',"vmin":-math.pi,"vmax":math.pi,"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"$\Delta\phi$ [rad]","f_t_sampling":1,"f_nu_sampling":1, "correction":False}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
#        for index_t in range(len(f_t)):
#            if f_t[index_t]<dict_subplot["f_t_min"]:
#                min_index_t = index_t
#            elif f_t[index_t]>dict_subplot["f_t_max"]:
#                max_index_t = index_t
#                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
#        for index_nu in range(len(f_nu)):
#            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
#                min_index_nu = index_nu
#            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
#                max_index_nu = index_nu
#                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        if dict_subplot["correction"]:
            Phase = cdelph[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        else:
            Phase = delph[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        f_t_sampling = dict_subplot["f_t_sampling"]
        f_nu_sampling = dict_subplot["f_nu_sampling"]
        def func_add(block,axis=(0,1)):
            result = np.sum(block+math.pi,axis)%(2.*math.pi)-math.pi
            return result
        Phase = block_reduce(Phase, block_size=(f_t_sampling,f_nu_sampling), func=func_add)
        coordinates = np.array([f_t,f_t])
        coordinates = block_reduce(coordinates, block_size=(1,f_t_sampling), func=np.mean, cval=f_t[-1])
        f_t = coordinates[0,:]
        coordinates = np.array([f_nu,f_nu])
        coordinates = block_reduce(coordinates, block_size=(1,f_nu_sampling), func=np.mean, cval=f_nu[-1])
        f_nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh((f_t-offset_f_t)/1.0e-03,(f_nu-offset_f_nu)/1.0e-06,map(list, zip(*Phase)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$f_D$ [mHz]")
        ax.set_ylabel(r"$\tau$ [$\mu$s]")
        
        print("Successfully plotted phase shift of secondary spectrum.")
        
    def plot_FFT_phi_evolution(self,dict_plot,figure,ax,tag):
        #load and check data
        # - FFT_phi_evolution
        path_computation = os.path.join(self.path_computations,"FFT_phi_evolution")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute FFT_phi_evolution' first!")
            return 0
        file_phase = os.path.join(path_computation,"phase.npy")
        file_cphase = os.path.join(path_computation,"cphase.npy")
        file_midnus = os.path.join(path_computation,"midnus.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        phase = np.load(file_phase)
        cphase = np.load(file_cphase)
        midnus = np.load(file_midnus)*1.0e-06 #MHz
        f_t = np.load(file_doppler)/(2.*math.pi*1.0e-03) #mHz
        f_nu = np.load(file_delay)/(2.*math.pi*1.0e-06) #us
        print("phase: [{0},{1}], cphase: [{2},{3}]".format(np.min(phase),np.max(phase),np.min(cphase),np.max(cphase)))
        
        #load specifications
        key_sepcs = "plot_secondary_phase{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"f_D":0.,"tau":0.,"title":"$\phi$ [rad]","phi_max":math.pi,"phi_min":-math.pi,
                            "nu_min":float(midnus[0]),"nu_max":float(midnus[-1])}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #draw pixel
        i_t = (np.abs(f_t-dict_subplot["f_D"])).argmin()
        i_nu = (np.abs(f_nu-dict_subplot["tau"])).argmin()
        
        #draw the plot
        ax.set_xlim([dict_subplot["nu_min"],dict_subplot["nu_max"]])
        ax.set_ylim([dict_subplot["phi_min"],dict_subplot["phi_max"]])
        ax.plot(midnus,phase[:,i_t,i_nu],label="raw",linestyle='dotted',marker="o",markersize=8)
        ax.plot(midnus,cphase[:,i_t,i_nu],label="corrected",linestyle='dotted',marker="o",markersize=8)
        ax.legend(loc='best')
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\nu$ [MHz]")
        ax.set_ylabel("$\phi$ [rad] at $f_D$={0}, $\\tau$={1}".format(dict_subplot["f_D"],dict_subplot["tau"]))
        
        print("Successfully plotted phase shift of secondary spectrum.")
        
    def plot_FFT_phi_substep(self,dict_plot,figure,ax,tag):
        #load and check data
        # - FFT_phi_evolution
        path_computation = os.path.join(self.path_computations,"FFT_phi_evolution")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute FFT_phi_evolution' first!")
            return 0
        file_phase = os.path.join(path_computation,"phase.npy")
        file_cphase = os.path.join(path_computation,"cphase.npy")
        file_midnus = os.path.join(path_computation,"midnus.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        phase = np.load(file_phase)
        cphase = np.load(file_cphase)
        midnus = np.load(file_midnus)*1.0e-06 #MHz
        f_t = np.load(file_doppler)/(2.*math.pi*1.0e-03) #mHz
        f_nu = np.load(file_delay)/(2.*math.pi*1.0e-06) #us
        print("phase: [{0},{1}], cphase: [{2},{3}]".format(np.min(phase),np.max(phase),np.min(cphase),np.max(cphase)))
        
        #load specifications
        key_sepcs = "plot_FFT_phi_substep{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'hsv',"vmin":-math.pi,"vmax":math.pi,"f_t_min":float(f_t[0]),"f_t_max":float(f_t[-1]),"f_nu_min":float(f_nu[0]),"f_nu_max":float(f_nu[-1]),
                            "index_nu":0, "correction":False}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        i_nu = dict_subplot["index_nu"]
            
        #Preprocess data
        if dict_subplot["correction"]:
            Phase = cphase[i_nu,:,:]
        else:
            Phase = phase[i_nu,:,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh((f_t-offset_f_t),(f_nu-offset_f_nu),map(list, zip(*Phase)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title("$\phi$ [rad] at $\\nu_c$={0} MHz".format(midnus[i_nu]))
        ax.set_xlabel(r"$f_D$ [mHz]")
        ax.set_ylabel(r"$\tau$ [$\mu$s]")
        
        print("Successfully plotted phase of secondary spectrum.")
        
    def plot_phase_gradient_SSpec(self,dict_plot,figure,ax,tag):
        #load and check data
        # - extracted_SSpec
        path_computation = os.path.join(self.path_computations,"extracted_SSpec")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute extracted_SSpec' first!")
            return 0
        file_ESSpec = os.path.join(path_computation,"ESSpec.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        ESSpec = np.load(file_ESSpec)
        Slope = ESSpec[1,:,:]*1.0e+6 #/MHz
        f_t = np.load(file_doppler)/(2.*math.pi*1.0e-03) #mHz
        f_nu = np.load(file_delay)/(2.*math.pi*1.0e-06) #us
        #print("phase: [{0},{1}], cphase: [{2},{3}]".format(np.min(ESSpec),np.max(ESSpec),np.min(cphase),np.max(cphase)))
        
        #load specifications
        key_sepcs = "plot_phase_gradient_SSpec{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":float(np.min(Slope)),"vmax":float(np.max(Slope)),"f_t_min":float(f_t[0]),"f_t_max":float(f_t[-1]),"f_nu_min":float(f_nu[0]),"f_nu_max":float(f_nu[-1])}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh((f_t-offset_f_t),(f_nu-offset_f_nu),map(list, zip(*Slope)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(r"$\partial\phi / \partial\nu$ [rad/MHz]")
        ax.set_xlabel(r"$f_D$ [mHz]")
        ax.set_ylabel(r"$\tau$ [$\mu$s]")
        
        print("Successfully plotted phase gradient of secondary spectrum.")
        
#-----COMBINED SPECTRUM-----
    
    def compute_combined_spectrum(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.

        #load and check list of theta combinations
        file_thetas = os.path.join(path_computation,"thetas.txt")
        if not os.path.exists(file_thetas):
            with open(file_thetas,'w') as writefile:
                writefile.write("#in mas: theta00\ttheta01\ttheta10\ttheta11\n")
                writefile.write("0.\t0.\t0.\t0.")
        with open(file_thetas,'r') as readfile:
            N_th = 0
            text = []
            for line in readfile.readlines():
                if not (line[0]=='#' or line[0]=='\n'):
                    text.append(line)
                    N_th += 1
        
        
        #define files to compute
        file_combo = os.path.join(path_computation,"combined_spectrum.npy")
        file_theta = os.path.join(path_computation,"theta.npy")
        
        #create containers
        Combo = np.zeros((N_th,self.N_t,self.N_nu),dtype=np.complex)
        theta = np.zeros((N_th,2,2),dtype=float)
        
        #load theta combinations
        for index,line in enumerate(text):
            thetas = line.split("\t")
            theta[index,0,0] = float(thetas[0])*self.mas
            theta[index,0,1] = float(thetas[1])*self.mas
            theta[index,1,0] = float(thetas[2])*self.mas
            theta[index,1,1] = float(thetas[3])*self.mas
        
        #perform the computation
        time_start = time.time()
        print("Computing combined spectrum ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - perform the combination
        f_doppler = 2.*math.pi*self.nu_half/self.LightSpeed
        bar = progressbar.ProgressBar(maxval=N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in xrange(N_th):
            bar.update(i_th)
            Combo[i_th,:,:] = np.sum(np.sum( DynSpec*np.exp(1j*f_doppler*(theta[i_th,0,0]*p[:,na,:,na,0]+theta[i_th,0,1]*p[:,na,:,na,1]-theta[i_th,1,0]*p[na,:,:,na,0]-theta[i_th,1,1]*p[na,:,:,na,1])) ,axis=0),axis=0)/self.N_p**2
        bar.finish()
        
        #remove worst frequency channels
        freqsum = np.sum(np.abs(Combo),axis=(0,1))
        freqmed = np.median(freqsum)
        freqflags = np.zeros(self.N_nu,dtype=bool)
        freqflags = (freqsum/freqmed>3.)
        for i_nu in xrange(self.N_nu):
            if freqflags[i_nu]:
                Combo[:,:,i_nu] = 0.
        
        #save the results
        np.save(file_combo,Combo)
        np.save(file_theta,theta)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the combined spectrum.".format(time_current))
        
    def plot_dynamic_spectrum_c(self,dict_plot,figure,ax,tag):
        #load and check data
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_combo = os.path.join(path_results,"combined_spectrum.npy")
        #file_theta = os.path.join(path_results,"theta.npy")
        Combo = np.load(file_combo)
        #theta = np.load(file_theta)
        #N_th = len(theta[:,0,0])
        
        #load specifications
        key_sepcs = "plot_dynamic_spectrum_c{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":-1.,"vmax":float(np.max(np.real(Combo))),"t_min":float(self.t_min),"t_max":float(self.t_max),"nu_min":float(self.nu_min),"nu_max":float(self.nu_max),
                            "title":"combined dynamic spectrum","theta_combination":0,"t_sampling":1,"nu_sampling":1}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - choice of theta combination
        i_th = dict_subplot["theta_combination"]
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(self.t)-1
        for index_t in range(len(self.t)):
            if self.t[index_t]<dict_subplot["t_min"]:
                min_index_t = index_t
            elif self.t[index_t]>dict_subplot["t_max"]:
                max_index_t = index_t
                break
        t = self.t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(self.nu)-1
        for index_nu in range(len(self.nu)):
            if self.nu[index_nu]<dict_subplot["nu_min"]:
                min_index_nu = index_nu
            elif self.nu[index_nu]>dict_subplot["nu_max"]:
                max_index_nu = index_nu
                break
        nu = self.nu[min_index_nu:max_index_nu+2]
        DynSpectrum = np.abs(Combo[i_th,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        # - downsampling
        t_sampling = dict_subplot["t_sampling"]
        nu_sampling = dict_subplot["nu_sampling"]
        DynSpectrum = block_reduce(DynSpectrum, block_size=(t_sampling,nu_sampling), func=np.mean)
        coordinates = np.array([t,t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=t[-1])
        t = coordinates[0,:]
        coordinates = np.array([nu,nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=nu[-1])
        nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_t = (t[1]-t[0])/2.
        offset_nu = (nu[1]-nu[0])/2.
        
        
        #draw the plot
        ax.set_xlim([dict_subplot["t_min"],dict_subplot["t_max"]])
        ax.set_ylim([dict_subplot["nu_min"],dict_subplot["nu_max"]])
        im = ax.pcolormesh(t-offset_t,nu-offset_nu,map(list, zip(*DynSpectrum)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"time $t$ [$s$]")
        ax.set_ylabel(r"frequency $\nu$ [Hz]")
        
        print("Successfully plotted combined dynamic spectrum.")
        
    def compute_combined_secondary_spectrum(self,path_computation):
        #load and check data
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_combo = os.path.join(path_results,"combined_spectrum.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        Combo = np.load(file_combo)
        theta = np.load(file_theta)
        N_th = len(theta[:,0,0])
        
        #define files to compute
        file_cpower = os.path.join(path_computation,"combined_secondary_spectrum.npy")
        
        #create containers
        cpower = np.zeros((N_th,self.N_t,self.N_nu),dtype=float)
        
        #perform the computation
        time_start = time.time()
        print("Computing combined secondary spectrum ...")
        # - compute the power spectrum
        A_DynSpec = np.fft.fft2(Combo,axes=(1,2))
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(1,2))
        cpower = self.delta_nu**2*self.delta_t**2*np.abs(A_DynSpec)**2
        # - correct for shift of spectrum
        for i_th in range(N_th):
            max_cpower = np.amax(cpower[i_th,:,:])
            location = np.where(np.abs(cpower[i_th,:,:]-max_cpower)==0.)
            i_t,i_nu = list(zip(location[0],location[1]))[0]
            shift_t = i_t-self.N_t/2
            shift_nu = i_nu-self.N_nu/2
            print("shifted by ({0},{1})".format(shift_t,shift_nu))
            cpower[i_th,:,:] = np.roll(np.roll(cpower[i_th,:,:],-shift_t,axis=0),-shift_nu,axis=1)

        #save the results
        np.save(file_cpower,cpower)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the combined secondary spectrum.".format(time_current))
        
    def plot_secondary_spectrum_c(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_theta = os.path.join(path_results,"theta.npy")
        theta = np.load(file_theta)
        # - combined secondary spectrum
        path_results = os.path.join(self.path_computations,"combined_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_secondary_spectrum' first!")
            return 0
        file_cpower = os.path.join(path_results,"combined_secondary_spectrum.npy")
        cpower = np.load(file_cpower)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum_c{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":0.,"f_nu_max":float(math.pi/self.delta_nu),
                            "title":"combined secondary spectrum $P$ (log10)","theta_combination":0,"f_t_sampling":1,"f_nu_sampling":1}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - choice of theta combination
        i_th = dict_subplot["theta_combination"]
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        cpower = cpower[i_th,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        f_t_sampling = dict_subplot["f_t_sampling"]
        f_nu_sampling = dict_subplot["f_nu_sampling"]
        cpower = block_reduce(cpower, block_size=(f_t_sampling,f_nu_sampling), func=np.mean)
        coordinates = np.array([f_t,f_t])
        coordinates = block_reduce(coordinates, block_size=(1,f_t_sampling), func=np.mean, cval=f_t[-1])
        f_t = coordinates[0,:]
        coordinates = np.array([f_nu,f_nu])
        coordinates = block_reduce(coordinates, block_size=(1,f_nu_sampling), func=np.mean, cval=f_nu[-1])
        f_nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(cpower[np.nonzero(cpower)])
        cpower[cpower == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(cpower)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted combined secondary spectrum for theta = {0} mas.".format(theta[i_th,:,:]/self.mas))
        
    def plot_secondary_spectrum_cr(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_theta = os.path.join(path_results,"theta.npy")
        theta = np.load(file_theta)
        # - combined secondary spectrum
        path_results = os.path.join(self.path_computations,"combined_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_secondary_spectrum' first!")
            return 0
        file_cpower = os.path.join(path_results,"combined_secondary_spectrum.npy")
        cpower = np.load(file_cpower)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum_cr{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":1.,"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":0.,"f_nu_max":float(math.pi/self.delta_nu),
                            "threshold":0.1,"title":"ratio combined to classic secondary spectrum","theta_combination":0}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - choice of theta combination
        i_th = dict_subplot["theta_combination"]
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        SecSpec = SecSpec[0,0,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        cpower = cpower[i_th,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - smooth the data to reduce noise
        SecSpec = ndimage.uniform_filter(SecSpec, (3,4))
        cpower = ndimage.uniform_filter(cpower, (3,4))
        # - compute relative power
        ratio = cpower/SecSpec
        # - remove noise
        ratio[SecSpec<dict_subplot["threshold"]] = 0.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*ratio)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted relative combined secondary spectrum for theta = {0} mas.".format(theta[i_th,:,:]/self.mas))
        
    def plot_secondary_spectrum_cd(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_theta = os.path.join(path_results,"theta.npy")
        theta = np.load(file_theta)
        # - combined secondary spectrum
        path_results = os.path.join(self.path_computations,"combined_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_secondary_spectrum' first!")
            return 0
        file_cpower = os.path.join(path_results,"combined_secondary_spectrum.npy")
        cpower = np.load(file_cpower)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum_cd{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":1.,"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":0.,"f_nu_max":float(math.pi/self.delta_nu),
                            "threshold":0.1,"title":"difference of combined secondary spectra","theta_combination1":0,"theta_combination2":1,"smoothing_f_t":2,"smoothing_f_nu":2}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - choice of theta combination
        i_th1 = dict_subplot["theta_combination1"]
        i_th2 = dict_subplot["theta_combination2"]
        # - remove contaminated f_nu=0
        cpower[i_th1,:,self.N_nu/2] = 0.
        cpower[i_th2,:,self.N_nu/2] = 0.
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        cpower1 = cpower[i_th1,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        cpower2 = cpower[i_th2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - smooth the data to reduce noise
        cpower1 = ndimage.uniform_filter(cpower1, (dict_subplot["smoothing_f_nu"],dict_subplot["smoothing_f_nu"]))
        cpower2 = ndimage.uniform_filter(cpower2, (dict_subplot["smoothing_f_nu"],dict_subplot["smoothing_f_nu"]))
        # - compute difference in power
        diff = (cpower1 - cpower2)/np.abs(cpower1)
        # - remove noise
        diff[cpower1<dict_subplot["threshold"]] = 0.
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*diff)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted difference of combined secondary spectra for theta1 = {0} mas and theta2 = {1}.".format(theta[i_th1,:,:]/self.mas,theta[i_th2,:,:]/self.mas))
        
    def plot_secondary_spectrum_cs(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        file_theta = os.path.join(path_results,"theta.npy")
        theta = np.load(file_theta)
        N_th = len(theta)
        # - combined secondary spectrum
        path_results = os.path.join(self.path_computations,"combined_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_secondary_spectrum' first!")
            return 0
        file_cpower = os.path.join(path_results,"combined_secondary_spectrum.npy")
        cpower = np.load(file_cpower)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum_cs{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"sum_min":None,"sum_max":None,"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":-float(math.pi/self.delta_nu),"f_nu_max":float(math.pi/self.delta_nu),
                            "title":"summed combined secondary spectrum","x_min":0.,"x_max":180.,"x_label":"degrees"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - create theta label vector
        x_theta = np.linspace(dict_subplot["x_min"],dict_subplot["x_max"],num=N_th,endpoint=True)
        # - remove contaminated f_nu=0
        cpower[:,:,self.N_nu/2] = 0.
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        cpower = cpower[:,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - sum spectra
        cs = np.sum(cpower,axis=(1,2))
        
        #draw the plot
        ax.set_xlim([dict_subplot["x_min"],dict_subplot["x_max"]])
        ax.set_ylim([dict_subplot["sum_min"],dict_subplot["sum_max"]])
        ax.plot(x_theta,cs,label="summed power")
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(dict_subplot["x_label"])
        ax.set_ylabel(r"$\Sigma$ Power")
        plt.grid(True)
        
        print("Successfully plotted sums of combined secondary spectra.")
        
    def compute_c_D_eff(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)

        #load and check list of theta combinations
#        file_modes = os.path.join(path_computation,"mode.txt")
#        if not os.path.exists(file_modes):
#            with open(file_modes,'w') as writefile:
#                writefile.write("#f_t\tf_nu\n")
#                writefile.write("0.\t0.")
#        with open(file_modes,'r') as readfile:
#            N_m = 0
#            text = []
#            for line in readfile.readlines():
#                if not (line[0]=='#' or line[0]=='\n'):
#                    text.append(line)
#                    N_m += 1
                    
        #define files to compute
        file_maxima = os.path.join(path_computation,"maxima.npy")
        file_D_eff = os.path.join(path_computation,"D_eff.npy")
        file_tracer = os.path.join(path_computation,"tracer.npy")
        file_mode = os.path.join(path_computation,"mode.npy")
        
        #create modes
        N_m = 1
        mode = np.zeros((N_m,2),dtype=int)
        i_m = 0
        while i_m<N_m:
            p_f_t = rd.uniform(-0.25,0.25)
            p_f_nu = rd.uniform(-0.003,0.003)
            if not (-0.03<p_f_t<0.03):
                if not (-0.0003<p_f_nu<0.0003):
                    i_t = (np.abs(f_t-p_f_t)).argmin()
                    i_nu = (np.abs(f_nu-p_f_nu)).argmin()
                    if SecSpec[0,0,i_t,i_nu]>5.0E+12:
                        mode[i_m,0] = i_t
                        mode[i_m,1] = i_nu
                        i_m += 1
        
        #create containers
        Combo = np.zeros((self.N_t,self.N_nu),dtype=np.complex)
        maxima = np.zeros((N_m,5),dtype=float)
        ind_Deff = np.zeros(N_m,dtype=float)
        tracer = []
        
        #load choice of modes
#        for index,line in enumerate(text):
#            modes = line.split("\t")
#            mode[index,0] = float(modes[0])
#            mode[index,1] = float(modes[1])
        
        #perform the computation
        time_start = time.time()
        print("Computing D_eff ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - perform the combination
        f_doppler = 2.*math.pi*self.nu_half/self.LightSpeed
        # - start the loop
        time_current = time.time()-time_start
        while time_current<5.*3600.:
            # - throw dices
            theta00 = rd.uniform(-25.,25.)*self.mas
            theta01 = rd.uniform(-25.,25.)*self.mas
            theta10 = rd.uniform(-25.,25.)*self.mas
            theta11 = rd.uniform(-25.,25.)*self.mas
            # - compute combined spectrum
            Combo = np.sum(np.sum( DynSpec*np.exp(1j*f_doppler*(theta00*p[:,na,:,na,0]+theta01*p[:,na,:,na,1]-theta10*p[na,:,:,na,0]-theta11*p[na,:,:,na,1])) ,axis=0),axis=0)/self.N_p**2
            # - compute the power spectrum
            A_DynSpec = np.fft.fft2(Combo,axes=(0,1))
            A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(0,1))
            cpower = np.abs(A_DynSpec)**2
            # - correct shift of spectrum
            max_cpower = np.amax(cpower)
            location = np.where(np.abs(cpower-max_cpower)==0.)
            i_t,i_nu = list(zip(location[0],location[1]))[0]
            shift_t = i_t-self.N_t/2
            shift_nu = i_nu-self.N_nu/2
            cpower = np.roll(np.roll(cpower,-shift_t,axis=0),-shift_nu,axis=1)
            for i_m in xrange(N_m):
                # - compute the power spectrum at each mode
                power = cpower[mode[i_m,0],mode[i_m,1]]
                if maxima[i_m,0]<power:
                    maxima[i_m,0] = power
                    maxima[i_m,1] = theta00/self.mas
                    maxima[i_m,2] = theta01/self.mas
                    maxima[i_m,3] = theta10/self.mas
                    maxima[i_m,4] = theta11/self.mas
                    # - compute D_eff
                    ind_Deff[i_m] = self.LightSpeed/math.pi*f_nu[mode[i_m,1]]/(theta10**2+theta11**2-theta00**2-theta01**2)
                    D_eff = np.mean(ind_Deff)/self.pc
                    std_Deff = np.std(ind_Deff)/self.pc
                    print("D_eff={0} +/- {1} pc".format(D_eff,std_Deff))
            tracer.append([cpower[mode[0,0],mode[0,1]],theta00/self.mas,theta01/self.mas,theta10/self.mas,theta11/self.mas])
            time_current = time.time()-time_start
            
        #save the results
        np.save(file_maxima,maxima)
        np.save(file_D_eff,ind_Deff)
        np.save(file_tracer,np.array(tracer))
        np.save(file_mode,mode)
        
        time_current = time.time()-time_start
        print("{0}s: Finished search for maxima.".format(time_current))
        
    def plot_c_tracer(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - c_D_eff
        path_results = os.path.join(self.path_computations,"c_D_eff")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute c_D_eff' first!")
            return 0
        file_tracer = os.path.join(path_results,"tracer.npy")
        file_mode = os.path.join(path_results,"mode.npy")
        tracer = np.load(file_tracer)
        mode = np.load(file_mode)
        
        #load specifications
        key_sepcs = "plot_c_tracer{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"theta_min":None,"theta_max":None,"n_bins":20,"cmap":'viridis','theta_choice':0,
                            "title":"$\\theta$ brightness distribution","vmin":0.,"vmax":None}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - create theta coordinates
        if dict_subplot["theta_choice"]==0:
            theta_x = tracer[:,1]
            theta_y = tracer[:,2]
        elif dict_subplot["theta_choice"]==1:
            theta_x = tracer[:,3]
            theta_y = tracer[:,4]
        # - get power of these points
        power = tracer[:,0]
        N_th = len(tracer[:,0])
        # - perform the binning
        N_mu = dict_subplot["n_bins"]
        theta = np.linspace(dict_subplot["theta_min"],dict_subplot["theta_max"],num=N_mu,endpoint=True)
        mean_mu = np.zeros((N_mu,N_mu),dtype=float)
        amount_mu = np.zeros((N_mu,N_mu),dtype=int)
        for i_th in xrange(N_th):
            i_thx = (np.abs(theta_x[i_th]-theta)).argmin()
            i_thy = (np.abs(theta_y[i_th]-theta)).argmin()
            amount_mu[i_thx,i_thy] += 1
            mean_mu[i_thx,i_thy] += power[i_th]
        amount_mu[amount_mu==0] = 1
        mean_mu /= amount_mu
        # - compute grid offset
        offset = (theta[1]-theta[0])/2.
        # - compute best fit
        max_cpower = np.amax(mean_mu)
        location = np.where(np.abs(mean_mu-max_cpower)==0.)
        i_x,i_y = list(zip(location[0],location[1]))[0]
        print(theta[i_x],theta[i_y])
        print(self.mas,self.LightSpeed,self.pc)
        
        #draw the plot
        ax.set_xlim([dict_subplot["theta_min"],dict_subplot["theta_max"]])
        ax.set_ylim([dict_subplot["theta_min"],dict_subplot["theta_max"]])
        im = ax.pcolormesh(theta,theta,map(list, zip(*mean_mu)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"],alpha=1.0)
        im = ax.scatter(theta_x,theta_y,c=power,cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta_x$ [mas]")
        ax.set_ylabel(r"$\theta_x$ [mas]")
        
        print("Successfully plotted the tracer of combined spectra at f_t={0}, f_nu={1}.".format(f_t[mode[0,0]],f_nu[mode[0,1]]))
        
    def compute_c_screen_angle(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
                    
        #define files to compute
        file_csum = os.path.join(path_computation,"csum.npy")
        file_thangle = os.path.join(path_computation,"thangle.npy")
        
        #create containers
        N_a = 100
        Combo = np.zeros((self.N_t,self.N_nu),dtype=np.complex)
        csum = np.zeros(N_a,dtype=float)
        thangle = np.linspace(0.,2.*math.pi,num=N_a,endpoint=True)
        
        #perform the computation
        time_start = time.time()
        print("Computing screen orientation from combined spectra ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - perform the combination
        f_doppler = 2.*math.pi*self.nu_half/self.LightSpeed
        # - start the loop
        bar = progressbar.ProgressBar(maxval=N_a, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_a in xrange(N_a):
            bar.update(i_a)
            # - set angles
            ang = thangle[i_a]
            mag = 40.*self.mas
            theta00 = mag*np.cos(ang)
            theta01 = mag*np.sin(ang)
            theta10 = mag*np.cos(ang)
            theta11 = mag*np.sin(ang)
            # - compute combined spectrum
            Combo = np.sum(np.sum( DynSpec*np.exp(1j*f_doppler*(theta00*p[:,na,:,na,0]+theta01*p[:,na,:,na,1]-theta10*p[na,:,:,na,0]-theta11*p[na,:,:,na,1])) ,axis=0),axis=0)/self.N_p**2
            # - compute the power spectrum
            A_DynSpec = np.fft.fft2(Combo,axes=(0,1))
            A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(0,1))
            cpower = np.abs(A_DynSpec)**2
            # - safely remove zeros if there are any
            min_nonzero = np.min(cpower[np.nonzero(cpower)])
            cpower[cpower == 0] = min_nonzero
            # - apply logarithmic scale
            cpower = np.log10(cpower)
            # sum the spectrum
            csum[i_a] = np.sum(cpower)
        bar.finish()
            
        #save the results
        np.save(file_csum,csum)
        np.save(file_thangle,thangle)
        
        time_current = time.time()-time_start
        print("{0}s: Finished exploring screen angle.".format(time_current))
        
    def compute_c_vel_angle(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        #file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        #file_delay = os.path.join(path_results,"delay.npy")
        #SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        #f_nu = np.load(file_delay)
                    
        #define files to compute
        file_csum = os.path.join(path_computation,"csum.npy")
        file_thangle = os.path.join(path_computation,"thangle.npy")
        
        #create containers
        N_a = 100
        Combo = np.zeros((self.N_t,self.N_nu),dtype=np.complex)
        csum = np.zeros(N_a,dtype=float)
        thangle = np.linspace(0.,2.*math.pi,num=N_a,endpoint=True)
        
        #perform the computation
        time_start = time.time()
        print("Computing velocity orientation from combined spectra ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - perform the combination
        f_doppler = 2.*math.pi*self.nu_half/self.LightSpeed
        # - start the loop
        bar = progressbar.ProgressBar(maxval=N_a, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_a in xrange(N_a):
            bar.update(i_a)
            # - set angles
            ang = thangle[i_a]
            mag = 5.*self.mas
#            theta00 = mag*np.cos(ang)
#            theta01 = mag*np.sin(ang)
            theta00 = 0.
            theta01 = 0.
            theta10 = mag*np.cos(ang)
            theta11 = mag*np.sin(ang)
            # - compute combined spectrum
            Combo = np.sum(np.sum( DynSpec*np.exp(1j*f_doppler*(theta00*p[:,na,:,na,0]+theta01*p[:,na,:,na,1]-theta10*p[na,:,:,na,0]-theta11*p[na,:,:,na,1])) ,axis=0),axis=0)/self.N_p**2
            # - compute the power spectrum
            A_DynSpec = np.fft.fft2(Combo,axes=(0,1))
            A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(0,1))
            cpower = np.abs(A_DynSpec)**2
            # sum the spectrum (upper left part)
            # - safely remove zeros if there are any
            min_nonzero = np.min(cpower[np.nonzero(cpower)])
            cpower[cpower == 0] = min_nonzero
            # - apply logarithmic scale
            cpower = np.log10(cpower)
            # - compute f_t barycenter
            csum[i_a] =  np.sum(cpower*f_t[:,na])/np.sum(cpower) #/np.sum(cpower[self.N_t/2+1:,self.N_nu/2+1:])
            # down left: np.sum(cpower[:self.N_t/2-1,:self.N_nu/2-1]) up left: np.sum(cpower[:self.N_t/2-1,self.N_nu/2+1:]) down right: np.sum(cpower[self.N_t/2+1:,:self.N_nu/2-1])
            # - compare parts separated by 180 degrees
#        comp = np.copy(csum)
#        for i_a in xrange(N_a):
#            csum[i_a] -= comp[(i_a+N_a/2)%N_a]
        bar.finish()
            
        #save the results
        np.save(file_csum,csum)
        np.save(file_thangle,thangle)
        
        time_current = time.time()-time_start
        print("{0}s: Finished exploring screen angle.".format(time_current))
        
    def plot_c_screen_angle(self,dict_plot,figure,ax,tag):
        #load and check data
        # - screen angle
        path_results = os.path.join(self.path_computations,"c_screen_angle")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute c_screen_angle' first!")
            return 0
        file_csum = os.path.join(path_results,"csum.npy")
        file_thangle = os.path.join(path_results,"thangle.npy")
        csum = np.load(file_csum)
        thangle = np.load(file_thangle)
        
        #load specifications
        key_sepcs = "plot_c_screen_angle{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"sum_min":None,"sum_max":None,
                            "title":"summed combined secondary spectrum"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - change to degrees
        thangle = thangle/math.pi*180.
        # - inform about result
        result = thangle[csum==np.max(csum)]
        print(result)
        print(csum[0])
        
        #draw the plot
        ax.set_xlim([thangle[0],thangle[-1]])
        ax.set_ylim([dict_subplot["sum_min"],dict_subplot["sum_max"]])
        ax.plot(thangle,csum,label="summed power")
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel("degrees")
        ax.set_ylabel(r"$\Sigma$ Power")
        plt.grid(True)
        
        print("Successfully plotted sums of combined secondary spectra by angle of screen.")
        
    def plot_c_vel_angle(self,dict_plot,figure,ax,tag):
        #load and check data
        # - screen angle
        path_results = os.path.join(self.path_computations,"c_screen_angle")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute c_screen_angle' first!")
            return 0
        file_csum = os.path.join(path_results,"csum.npy")
        csum_s = np.load(file_csum)
        # - velocity angle
        path_results = os.path.join(self.path_computations,"c_vel_angle")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute c_vel_angle' first!")
            return 0
        file_csum = os.path.join(path_results,"csum.npy")
        file_thangle = os.path.join(path_results,"thangle.npy")
        csum = np.load(file_csum)
        thangle = np.load(file_thangle)
        
        #load specifications
        key_sepcs = "plot_c_vel_angle{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"sum_min":None,"sum_max":None,
                            "title":"summed combined secondary spectrum"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - change to degrees
        thangle = thangle/math.pi*180.
        # - norm by screen brightness curve
        #csum = csum/csum_s*np.mean(csum_s)
        # - fit cosini
#        mu_mean = np.mean(csum)
#        csum -= mu_mean
#        def cosini(theta,am_s,am_v,th_s,th_v,offset):
#            return am_s*np.cos(2.*math.pi/180.*(theta-th_s))+am_v*np.cos(math.pi/180.*(theta-th_v))+offset
#        popt, pcov = curve_fit(cosini,thangle,csum,p0=[2.0E+9,7.0E+9,26.,296.,-0.1E+9],bounds=([0.,0.,0.,0.,-np.inf],[np.inf,np.inf,360.,360.,np.inf]),sigma=np.std(csum)*0.01*csum/csum)
#        perr = np.sqrt(np.diag(pcov))
#        am_s = popt[0]
#        am_v = popt[1]
#        th_s = popt[2]
#        th_v = popt[3]
#        offset = popt[4]
#        scurve = am_s*np.cos(2.*math.pi/180.*(thangle-th_s)) + mu_mean
#        vcurve = am_v*np.cos(math.pi/180.*(thangle-th_v)) + mu_mean
#        bestfit = am_s*np.cos(2.*math.pi/180.*(thangle-th_s))+am_v*np.cos(math.pi/180.*(thangle-th_v))+offset+mu_mean
#        csum += mu_mean
#        # - inform about result
##        result = thangle[csum==np.max(csum)]
##        print(result,np.max(csum))
#        print(popt[2],popt[3])
        
        #draw the plot
        ax.set_xlim([thangle[0],thangle[-1]])
        ax.set_ylim([dict_subplot["sum_min"],dict_subplot["sum_max"]])
        ax.plot(thangle,csum,label=r"$f_t$ barycenter") #"summed power"
#        ax.plot(thangle,scurve,label="screen curve")
#        ax.plot(thangle,vcurve,label="velocity curve")
#        ax.plot(thangle,bestfit,label="bestfit")
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel("degrees")
        ax.set_ylabel(r"$f_t$ barycenter") #r"$\Sigma$ Power"
        plt.grid(True)
        #plt.legend(loc='best')
        
        print("Successfully plotted sums of combined secondary spectra by angle of velocity.")
        
    def compute_c_contrast(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
                    
        #define files to compute
        file_ccpower = os.path.join(path_computation,"ccpower.npy")
        
        #create containers
        Combo = np.zeros((self.N_t,self.N_nu),dtype=np.complex)
        ccpower = np.zeros((self.N_t,self.N_nu),dtype=float)
        
        #perform the computation
        time_start = time.time()
        print("Computing contrast increased secondary spectrum ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - perform the combination
        f_doppler = 2.*math.pi*self.nu_half/self.LightSpeed
        # - start the loop
        N_update = 101
        ones = np.ones((self.N_t,self.N_nu),dtype=int)
        while N_update>100:
            # - throw dices
            th_max = 25.
            theta00 = rd.uniform(-th_max,th_max)*self.mas
            theta01 = rd.uniform(-th_max,th_max)*self.mas
            theta10 = rd.uniform(-th_max,th_max)*self.mas
            theta11 = rd.uniform(-th_max,th_max)*self.mas
            # - compute combined spectrum
            Combo = np.sum(np.sum( DynSpec*np.exp(1j*f_doppler*(theta00*p[:,na,:,na,0]+theta01*p[:,na,:,na,1]-theta10*p[na,:,:,na,0]-theta11*p[na,:,:,na,1])) ,axis=0),axis=0)/self.N_p**2
            # - compute the power spectrum
            A_DynSpec = np.fft.fft2(Combo,axes=(0,1))
            A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(0,1))
            cpower = np.abs(A_DynSpec)**2
            # - correct shift of spectrum
            max_cpower = np.amax(cpower)
            location = np.where(np.abs(cpower-max_cpower)==0.)
            i_t,i_nu = list(zip(location[0],location[1]))[0]
            shift_t = i_t-self.N_t/2
            shift_nu = i_nu-self.N_nu/2
            cpower = np.roll(np.roll(cpower,-shift_t,axis=0),-shift_nu,axis=1)
            # - update mode power if greater
            N_update = 0
            N_update = np.sum(ones[ccpower<cpower])
            print(N_update)
            ccpower[ccpower<cpower] = cpower[ccpower<cpower]
            
        #save the results
        np.save(file_ccpower,ccpower)
        
        time_current = time.time()-time_start
        print("{0}s: Finished increasing contrast.".format(time_current))
        
    def plot_secondary_spectrum_cc(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - combined contrast increased secondary spectrum
        path_results = os.path.join(self.path_computations,"c_contrast")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute c_contrast' first!")
            return 0
        file_ccpower = os.path.join(path_results,"ccpower.npy")
        ccpower = np.load(file_ccpower)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum_cc{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":-float(math.pi/self.delta_nu),"f_nu_max":float(math.pi/self.delta_nu),
                            "title":"contrasted secondary spectrum $P$ (log10)"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        ccpower = ccpower[min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(ccpower[np.nonzero(ccpower)])
        ccpower[ccpower == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(ccpower)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [$1/s$]")
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]")
        
        print("Successfully plotted combined secondary spectrum with increased contrast")
        
    def plot_theta_positions(self,dict_plot,figure,ax,tag):
        #load and check data
        positions = np.load(os.path.join(self.path_data,"positions.npy"))
        positions[:,:,0] *= -1.
        # - combined spectrum
        path_results = os.path.join(self.path_computations,"combined_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute combined_spectrum' first!")
            return 0
        #file_combo = os.path.join(path_results,"combined_spectrum.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        #Combo = np.load(file_combo)
        theta = np.load(file_theta)
        #N_th = len(theta[:,0,0])
        
        #load specifications
        key_sepcs = "plot_theta_positions{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"title":"$\\theta$ values and incomplete delta","th_min":0.,"th_max":1.,"theta_combination":0,"arrow_head_width":1.,"markercolor":"red",
                            "beam": True, "background":None}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        comb = dict_subplot["theta_combination"]
        # - compute incomplete delta distribution
        thetas = np.linspace(dict_subplot["th_min"]*self.mas,dict_subplot["th_max"]*self.mas,num=100,dtype=float)
#        mpos = np.mean(positions,axis=1)
#        exponential = np.exp(-1j*2.*math.pi*self.nu_half/self.LightSpeed*((thetas[na,:,na]-theta[comb,0,0])*mpos[:,na,na,0]+(thetas[na,na,:]-theta[comb,0,1])*mpos[:,na,na,1]))
#        kernel = np.abs(1./self.N_p*np.sum(exponential,axis=0))
        exponential = np.exp(-1j*2.*math.pi*self.nu_half/self.LightSpeed*((thetas[na,:,na,na]-theta[comb,0,0])*positions[:,na,na,:,0]+(thetas[na,na,:,na]-theta[comb,0,1])*positions[:,na,na,:,1]))
        kernel = np.mean(np.abs(1./self.N_p*np.sum(exponential,axis=0)),axis=2)
        if dict_subplot["beam"]:
            alpha = 1.
        else:
            alpha = 0.
            
        #load background data
        background = dict_subplot["background"]
        if not background == None:
            file_background = os.path.join("measurements",background)
            points = np.load(file_background)
            #points[:,0] *= -1.
            points[:,1] *= -1.
            
        #draw the plot
        ax.set_xlim([dict_subplot["th_min"],dict_subplot["th_max"]])
        ax.set_ylim([dict_subplot["th_min"],dict_subplot["th_max"]])
        im = ax.pcolormesh(thetas/self.mas,thetas/self.mas,map(list, zip(*kernel)),cmap=dict_subplot["cmap"],vmin=0.,vmax=1.,alpha=alpha)
        ax.scatter(theta[comb,:,0]/self.mas,theta[comb,:,1]/self.mas,c=dict_subplot["markercolor"])
        if not background == None:
            ax.scatter(points[:,0],points[:,1],c="black",marker="o",facecolors='none',alpha=0.3)
        figure.colorbar(im, ax=ax)
        ar_x = theta[comb,0,0]/self.mas
        ar_y = theta[comb,0,1]/self.mas
        ar_dx = theta[comb,1,0]/self.mas - ar_x
        ar_dy = theta[comb,1,1]/self.mas - ar_y
        if not ar_dx==ar_dy==0.:
            ax.arrow(ar_x,ar_y,ar_dx,ar_dy,length_includes_head=True,head_width=dict_subplot["arrow_head_width"],facecolor=dict_subplot["markercolor"],edgecolor=dict_subplot["markercolor"])
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta_x$ [mas]")
        ax.set_ylabel(r"$\theta_y$ [mas]")
        
        print("Successfully plotted positions of thetas used for transformation.")
        
#-----THETA THETA DIAGRAM-----
    
    def compute_thth_diagram(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_thth_diagram.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"N_th":100,"th_min":-25.,"th_max":25.,"D_eff":1000.,"v_eff":300.,"v_angle":0.,"source":"ss_0_0", "eta":0.5797, "from_curvature":True}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        N_th = dict_specs["N_th"]
        th_min = dict_specs["th_min"]*self.mas
        th_max = dict_specs["th_max"]*self.mas
        v_eff = dict_specs["v_eff"]*1000.
        v_angle = dict_specs["v_angle"]*math.pi/180.
        vv_eff = [v_eff*np.cos(v_angle),v_eff*np.sin(v_angle)]
        if dict_specs["from_curvature"]:
            D_eff = dict_specs["eta"]/self.LightSpeed*2.*self.nu_half**2*vv_eff[0]**2
            print(D_eff/self.pc)
        else:
            D_eff = dict_specs["D_eff"]*self.pc
        source = dict_specs["source"]
        
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - check the data source
        source = source.split("_")
        if source[0]=="ss":
            i_p1 = int(source[1])
            i_p2 = int(source[2])
            SecSpec = SecSpec[i_p1,i_p2]
        elif source[0]=="cc":
            # - combined contrast increased secondary spectrum
            path_results = os.path.join(self.path_computations,"c_contrast")
            if not os.path.exists(path_results):
                warnings.warn("You need to run 'compute c_contrast' first!")
                return 0
            file_ccpower = os.path.join(path_results,"ccpower.npy")
            ccpower = np.load(file_ccpower)
            SecSpec = ccpower
        
        #define files to compute
        file_thth = os.path.join(path_computation,"thth.npy")
        file_theta = os.path.join(path_computation,"theta.npy")
        
        #create containers
        thth = np.zeros((N_th,N_th),dtype=float)
        theta = np.linspace(th_min,th_max,num=N_th,endpoint=True)
        
        #perform the computation
        time_start = time.time()
        print("Computing theta theta diagram ...")
        # - clean secondary spectrum
        for i_nu in xrange(self.N_nu):
            SecSpec[:,i_nu] -= np.median(SecSpec[:,i_nu])
        SecSpec[SecSpec<0.] = 0.
        # - interpolate
        ip_SecSpec = interpolate.interp2d(f_nu,f_t,SecSpec,kind='linear',fill_value=0.)
        # - compute flux conserving norm
        norm = 2.*math.pi/self.LightSpeed*D_eff
        # - compute conversion factors
        t_factor = 2.*math.pi*self.nu_half/self.LightSpeed*vv_eff[0]
        nu_factor = -math.pi/self.LightSpeed*D_eff
        # - perform the combination
        bar = progressbar.ProgressBar(maxval=N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th1,v_th1 in enumerate(theta):
            bar.update(i_th1)
            for i_th2,v_th2 in enumerate(theta):
                f_t = t_factor*(v_th1-v_th2)
                f_nu = nu_factor*(v_th1+v_th2)*(v_th1-v_th2)
                thth[i_th1,i_th2] = ip_SecSpec(f_nu,f_t)*np.abs(norm*f_t)
        bar.finish()
        
        #save the results
        np.save(file_thth,thth)
        np.save(file_theta,theta)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the theta-theta diagram.".format(time_current))
        
    def plot_thth_diagram(self,dict_plot,figure,ax,tag):
        #load and check data
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        thth = np.load(file_thth)
        theta = np.load(file_theta)
        
        #load specifications
        key_sepcs = "plot_thth_diagram{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"th_min":None,"th_max":None,
                            "title":"secondary spectrum $P$ (log10)"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - compute offsets to center pccolormesh
        offset = (theta[1]-theta[0])/2./self.mas
        # - safely remove zeros if there are any
        min_nonzero = np.min(thth[np.nonzero(thth)])
        thth[thth == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(thth)
        
        #draw the plot
        ax.set_xlim([dict_subplot["th_min"],dict_subplot["th_max"]])
        ax.set_ylim([dict_subplot["th_min"],dict_subplot["th_max"]])
        im = ax.pcolormesh(theta/self.mas-offset,theta/self.mas-offset,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta_x$ [mas]")
        ax.set_ylabel(r"$\theta_y$ [mas]")
        
        print("Successfully plotted theta-theta diagram.")
        
    def compute_thth_root(self,path_computation):
        #load and check data
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        thth = np.load(file_thth)
        theta = np.load(file_theta)
        N_th = len(theta)
        
        #load specifications
        file_specs = os.path.join(path_computation,"specs_thth_root.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"epochs":100000,"dlur":3.,"uldr":1.}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        epochs = dict_specs["epochs"]
        dlur = dict_specs["dlur"]*self.mas
        uldr = dict_specs["uldr"]*self.mas
        
        #define files to compute
        file_brightness = os.path.join(path_computation,"brightness.npy")
        file_thth_clean = os.path.join(path_computation,"thth_clean.npy")
        
        #create containers
        mu = np.zeros(N_th,dtype=float)
        thth_clean = np.empty((N_th,N_th),dtype=float)
        
        #perform the computation
        time_start = time.time()
        print("Computing brightness distribution ...")
#        # - take the double median of the differences between columns and lines to get a first guess
#        for i_th in range(N_th):
#            mu[i_th] = np.median(np.median(thth[:,i_th,na]-thth[:,:],axis=0))
        # - or start from eigenvector decomposition
        eigenvalues, eigenvectors = np.linalg.eig(thth)
        amplitude = np.sqrt(np.max(np.abs(eigenvalues)))
        i_amp = np.where(eigenvalues==np.max(np.abs(eigenvalues)))[0][0]
        mu = amplitude*np.abs(eigenvectors[:,i_amp])
        # - perform the maschine learning
        bar = progressbar.ProgressBar(maxval=epochs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in xrange(epochs):
            bar.update(i)
            for i1 in range(N_th):
                i2 = rd.randint(0,N_th-1)
                if (abs(theta[i1]+theta[i2])>uldr) & (abs(theta[i1]-theta[i2])>dlur):
                    if not mu[i2]==0.:
                        max_step = mu[i2]*0.01*(1.-float(i)/epochs)
                        grad = thth[i1,i2]/mu[i2]-mu[i1]
                        grad *= 0.2
                        if abs(grad)>max_step:
                            grad = max_step*np.sign(grad)
                        mu[i1] += grad
                        if mu[i1]<0.:
                            mu[i1] = 0.
        bar.finish()
        # - compute cleaned theta-theta diagram
        thth_clean = mu[:,na]*mu[na,:]
        
        #save the results
        np.save(file_brightness,mu)
        np.save(file_thth_clean,thth_clean)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the brightness distribution by taking the root of the theta-theta diagram.".format(time_current))
        
    def compute_thth_eig_decomp(self,path_computation):
        #load and check data
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        thth = np.load(file_thth)
        theta = np.load(file_theta)
        N_th = len(theta)
        
        #define files to compute
        file_brightness = os.path.join(path_computation,"brightness.npy")
        file_thth_clean = os.path.join(path_computation,"thth_clean.npy")
        
        #create containers
        mu = np.zeros(N_th,dtype=float)
        thth_clean = np.empty((N_th,N_th),dtype=float)
        
        #perform the computation
        time_start = time.time()
        print("Performing eigenvector decomposition ...")
#        # - flag noisy parts to zero
#        uldr = 1.
#        dlur = 3.
#        for i1 in range(N_th):
#            for i2 in range(N_th):
#                if (abs(theta[i1]+theta[i2])>uldr) & (abs(theta[i1]-theta[i2])>dlur):
#                    thth[i1,i2] = 0.
        # - perform the decomposition
        eigenvalues, eigenvectors = np.linalg.eig(thth)
        amplitude = np.sqrt(np.max(np.abs(eigenvalues)))
        i_amp = np.where(eigenvalues==np.max(np.abs(eigenvalues)))[0][0]
        mu = amplitude*np.abs(eigenvectors[:,i_amp])
        thth_clean = mu[:,na]*mu[na,:]
        
        #save the results
        np.save(file_brightness,mu)
        np.save(file_thth_clean,thth_clean)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the brightness distribution by performing the eigenvectordecomposition of the theta-theta diagram.".format(time_current))
        
        
    def plot_thth_brightness(self,dict_plot,figure,ax,tag):
        #load and check data
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        thth = np.load(file_thth)
        theta = np.load(file_theta)
        N_th = len(theta)
        
        #load specifications
        key_sepcs = "plot_thth_brightness{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"th_min":-25.,"th_max":25.,"mu_min":0.,"mu_max":1.,"color":"black",
                            "title":"brightness distribution (log10)","cmap":'viridis',"vmin":0.,"vmax":20.,"alpha":0.3, "source":"thth_eig_decomp"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        # - brightness
        path_results = os.path.join(self.path_computations,dict_subplot["source"])
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute {0}' first!".format(dict_subplot["source"]))
            return 0
        file_mu = os.path.join(path_results,"brightness.npy")
        #file_thth_clean = os.path.join(path_results,"thth_clean.npy")
        mu = np.load(file_mu)
        #thth_clean = np.load(file_thth_clean)
            
        #Preprocess data
        # - compute common y-axis
        min_index_th = 0
        max_index_th = N_th-1
        for index_th in range(N_th):
            if theta[index_th]<dict_subplot["th_min"]:
                min_index_th = index_th
            elif theta[index_th]>dict_subplot["th_max"]:
                max_index_th = index_th
                break
        th_mu = np.linspace(dict_subplot["mu_min"],dict_subplot["mu_max"],num=max_index_th-min_index_th+1)
        thth = thth[:,min_index_th:max_index_th+1]
        # - compute offsets to center pccolormesh
        offset = (theta[1]-theta[0])/2./self.mas
        #offset_mu = (th_mu[1]-th_mu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(thth[np.nonzero(thth)])
        thth[thth == 0] = min_nonzero
        min_nonzero = np.min(mu[np.nonzero(mu)])
        mu[mu == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(thth)
        mu_log10 = np.log10(mu)
        #print(np.nanmin(mu_log10),np.nanmax(mu_log10))
        
        
        #draw the plot
        ax.set_xlim([dict_subplot["th_min"],dict_subplot["th_max"]])
        ax.set_ylim([dict_subplot["mu_min"],dict_subplot["mu_max"]])
        im = ax.pcolormesh(theta/self.mas-offset,th_mu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"],alpha=dict_subplot["alpha"])
        figure.colorbar(im, ax=ax)
        ax.plot(theta/self.mas,mu_log10,label="brightness",color=dict_subplot["color"])
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta$ [mas]")
        ax.set_ylabel(r"$\mu$ (log10)")
        plt.grid(True)
        
        print("Successfully plotted brightness distribution from theta-theta diagram.")
        
    def plot_thth_diagram_clean(self,dict_plot,figure,ax,tag):
        #load and check data
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        #file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        #thth = np.load(file_thth)
        theta = np.load(file_theta)
        #N_th = len(theta)
        
        #load specifications
        key_sepcs = "plot_thth_diagram_clean{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"th_min":None,"th_max":None,
                            "title":"cleaned secondary spectrum (log10)", "source":"thth_eig_decomp"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        # - theta theta root
        path_results = os.path.join(self.path_computations,dict_subplot["source"])
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute {0}' first!".format(dict_subplot["source"]))
            return 0
        #file_mu = os.path.join(path_results,"brightness.npy")
        file_thth_clean = os.path.join(path_results,"thth_clean.npy")
        #mu = np.load(file_mu)
        thth_clean = np.load(file_thth_clean)
        
        #Preprocess data
        # - compute offsets to center pccolormesh
        offset = (theta[1]-theta[0])/2./self.mas
        # - safely remove zeros if there are any
        min_nonzero = np.min(thth_clean[np.nonzero(thth_clean)])
        thth_clean[thth_clean == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(thth_clean)
        
        #draw the plot
        ax.set_xlim([dict_subplot["th_min"],dict_subplot["th_max"]])
        ax.set_ylim([dict_subplot["th_min"],dict_subplot["th_max"]])
        im = ax.pcolormesh(theta/self.mas-offset,theta/self.mas-offset,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta_x$ [mas]")
        ax.set_ylabel(r"$\theta_y$ [mas]")
        
        print("Successfully plotted cleaned theta-theta diagram.")
        
    def compute_SecSpec_from_thth(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_thth_diagram.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"D_eff":1000.,"v_eff":100.,"v_angle":0.,"source":"thth_eig_decomp"}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        D_eff = dict_specs["D_eff"]*self.pc
        v_eff = dict_specs["v_eff"]*1000.
        v_angle = dict_specs["v_angle"]*math.pi/180.
        vv_eff = [v_eff*np.cos(v_angle),v_eff*np.sin(v_angle)]
        source = dict_specs["source"]
        
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - check the data source
        # - theta theta diagram
        path_results = os.path.join(self.path_computations,"thth_diagram")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute thth_diagram' first!")
            return 0
        file_thth = os.path.join(path_results,"thth.npy")
        file_theta = os.path.join(path_results,"theta.npy")
        theta = np.load(file_theta)
        #N_th = len(theta)
        if source == "thth_diagram":
            thth = np.load(file_thth)
        elif source == "thth_eig_decomp":
            # - theta theta root
            path_results = os.path.join(self.path_computations,"thth_eig_decomp")
            if not os.path.exists(path_results):
                warnings.warn("You need to run 'compute {0}' first!".format("thth_eig_decomp"))
                return 0
            #file_mu = os.path.join(path_results,"brightness.npy")
            file_thth_clean = os.path.join(path_results,"thth_clean.npy")
            #mu = np.load(file_mu)
            thth = np.load(file_thth_clean)
        elif source == "thth_root":
            # - theta theta root
            path_results = os.path.join(self.path_computations,"thth_root")
            if not os.path.exists(path_results):
                warnings.warn("You need to run 'compute {0}' first!".format("thth_root"))
                return 0
            #file_mu = os.path.join(path_results,"brightness.npy")
            file_thth_clean = os.path.join(path_results,"thth_clean.npy")
            #mu = np.load(file_mu)
            thth = np.load(file_thth_clean)
        
        #define files to compute
        file_SSthth = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        
        #create containers
        SSthth = np.zeros((self.N_t,self.N_nu),dtype=float)
        
        #perform the computation
        time_start = time.time()
        print("Computing secondary spectrum from theta theta diagram ...")
        # - interpolate
        ip_thth = interpolate.interp2d(theta,theta,thth,kind='linear',fill_value=0.)
        # - compute flux conserving norm
        norm = 2.*math.pi/self.LightSpeed*D_eff
        # - compute conversion factors
        a_factor = -self.nu_half*vv_eff[0]/D_eff
        b_factor = self.LightSpeed/(4.*math.pi*self.nu_half*vv_eff[0])
        # - perform the combination
        bar = progressbar.ProgressBar(maxval=self.N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_t,v_t in enumerate(f_t):
            bar.update(i_t)
            for i_nu,v_nu in enumerate(f_nu):
                if not v_t==0.:
                    th1 = a_factor*v_nu/v_t + b_factor*v_t
                    th2 = a_factor*v_nu/v_t - b_factor*v_t
                    SSthth[i_t,i_nu] = ip_thth(th1,th2)/np.abs(norm*v_t)
                else:
                    SSthth[i_t,i_nu] = 0.
        bar.finish()
        
        #save the results
        np.save(file_SSthth,SSthth)
        np.save(file_doppler,f_t)
        np.save(file_delay,f_nu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed secondary spectrum from the theta-theta diagram.".format(time_current))
        
    def plot_SecSpec_from_thth(self,dict_plot,figure,ax,tag):
        #load and check data
        # - SecSpec_from_thth
        path_computation = os.path.join(self.path_computations,"SecSpec_from_thth")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute SecSpec_from_thth' first!")
            return 0
        file_power = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #load specifications
        key_sepcs = "plot_secondary_spectrum{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":float(np.max(np.real(SecSpec))),"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "title":"secondary spectrum (log10)","f_t_sampling":1,"f_nu_sampling":1} # $P$ [$(W/m^2)^2$]
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        SecSpec = SecSpec[min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        f_t_sampling = dict_subplot["f_t_sampling"]
        f_nu_sampling = dict_subplot["f_nu_sampling"]
        SecSpec = block_reduce(SecSpec, block_size=(f_t_sampling,f_nu_sampling), func=np.mean)
        coordinates = np.array([f_t,f_t])
        coordinates = block_reduce(coordinates, block_size=(1,f_t_sampling), func=np.mean, cval=f_t[-1])
        f_t = coordinates[0,:]
        coordinates = np.array([f_nu,f_nu])
        coordinates = block_reduce(coordinates, block_size=(1,f_nu_sampling), func=np.mean, cval=f_nu[-1])
        f_nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(SecSpec[np.nonzero(SecSpec)])
        SecSpec[SecSpec == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(SecSpec)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [Hz]") #(r"$f_x$ [au]") #
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]") #(r"$f_y$ [au]") #
        
        print("Successfully plotted secondary spectrum.")
        
    def compute_reconstructed_DS(self,path_computation):
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_phase = os.path.join(path_results,"Fourier_phase.npy")
#        file_doppler = os.path.join(path_computation,"doppler.npy")
#        file_delay = os.path.join(path_computation,"delay.npy")
        phase = np.load(file_phase)
#        f_t = np.load(file_doppler)
#        f_nu = np.load(file_delay)
        # - SecSpec_from_thth
        path_results = os.path.join(self.path_computations,"SecSpec_from_thth")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute SecSpec_from_thth' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
#        file_doppler = os.path.join(path_computation,"doppler.npy")
#        file_delay = os.path.join(path_computation,"delay.npy")
        SecSpec = np.load(file_power)
#        f_t = np.load(file_doppler)
#        f_nu = np.load(file_delay)
        
        # telescopes (adapt this to be modifyable!!!)
        tel1 = 0
        tel2 = 0
        
        #define files to compute
        file_rds = os.path.join(path_computation,"rds.npy")
        
        #create containers
        rds = np.zeros((self.N_t,self.N_nu),dtype=float)
        A_theo = np.zeros((self.N_t,self.N_nu),dtype=complex)
        
        #perform the computation
        time_start = time.time()
        print("Computing reconstructed dynamic spectrum ...")
        # - reconstruct Fourier transform using original phase
        A_theo = np.sqrt(SecSpec)/(self.delta_t*self.delta_nu)*np.exp(1.j*phase[tel1,tel2,:,:])
        # - backtronsform to dynamic spectrum
        rds = np.fft.ifft2(A_theo)

        #save the results
        np.save(file_rds,rds)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the reconstructed dynamic spectrum.".format(time_current))
        
    def plot_reconstructed_DS(self,dict_plot,figure,ax,tag):
        #load and check data
        # - cleaned dynamic spectrum
        path_results = os.path.join(self.path_computations,"reconstructed_DS")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute reconstructed_DS first!")
            return 0
        file_rds = os.path.join(path_results,"rds.npy")
        DynSpec = np.load(file_rds)
        
        #load specifications
        key_sepcs = "plot_reconstructed_dynamic_spectrum{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":-1.,"vmax":float(np.max(np.real(DynSpec))),"t_min":float(self.t_min),"t_max":float(self.t_max),"nu_min":float(self.nu_min),"nu_max":float(self.nu_max),
                            "title":"reconstructed dynamic spectrum","t_sampling":1,"nu_sampling":1}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(self.t)-1
        for index_t in range(len(self.t)):
            if self.t[index_t]<dict_subplot["t_min"]:
                min_index_t = index_t
            elif self.t[index_t]>dict_subplot["t_max"]:
                max_index_t = index_t
                break
        t = self.t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(self.nu)-1
        for index_nu in range(len(self.nu)):
            if self.nu[index_nu]<dict_subplot["nu_min"]:
                min_index_nu = index_nu
            elif self.nu[index_nu]>dict_subplot["nu_max"]:
                max_index_nu = index_nu
                break
        nu = self.nu[min_index_nu:max_index_nu+2]
        DynSpectrum = np.real(DynSpec[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        # - downsampling
        t_sampling = dict_subplot["t_sampling"]
        nu_sampling = dict_subplot["nu_sampling"]
        DynSpectrum = block_reduce(DynSpectrum, block_size=(t_sampling,nu_sampling), func=np.mean)
        coordinates = np.array([t,t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=t[-1])
        t = coordinates[0,:]
        coordinates = np.array([nu,nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=nu[-1])
        nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_t = (t[1]-t[0])/2.
        offset_nu = (nu[1]-nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["t_min"],dict_subplot["t_max"]])
        ax.set_ylim([dict_subplot["nu_min"],dict_subplot["nu_max"]])
        im = ax.pcolormesh(t-offset_t,nu-offset_nu,map(list, zip(*DynSpectrum)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"time $t$ [$s$]")
        ax.set_ylabel(r"frequency $\nu$ [Hz]")
        
        print("Successfully plotted reconstructed dynamic spectrum.")
        
        
#-----LINEARIZED SECONDARY SPECTRUM
        
    def plot_lin_sec_spec(self,dict_plot,figure,ax,tag):
        #load and check data
        # - linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"linear_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute linear_secondary_spectrum' first!")
            return 0
        file_LinSecSpec = os.path.join(path_results,"LinSecSpec.npy")
        file_psip = os.path.join(path_results,"psip.npy")
        file_psin = os.path.join(path_results,"psin.npy")
        LinSecSpec = np.load(file_LinSecSpec)
        psip = np.load(file_psip)
        psin = np.load(file_psin)
        
        #load specifications
        key_sepcs = "plot_lin_sec_spec{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"psip_min":None,"psip_max":None,"psin_min":None,"psin_max":None,
                            "title":"linearized secondary spectrum (log10)"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - compute offsets to center pccolormesh
        offset_p = (psip[1]-psip[0])/2.
        offset_n = (psin[1]-psin[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(LinSecSpec[np.nonzero(LinSecSpec)])
        LinSecSpec[LinSecSpec == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(LinSecSpec)
        #SecSpec_log10 = skimage.exposure.equalize_hist(SecSpec_log10,nbins=256)
        #selem = np.ones((30,30),dtype=int)
        SecSpec_log10 = cv2.normalize(SecSpec_log10, dst=SecSpec_log10, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        selem = morphology.disk(30)
        SecSpec_log10 = skimage.filters.rank.equalize(SecSpec_log10,selem=selem)
        # - inform about scales
        print(psin[0],psin[-1],psip[0],psip[-1],np.min(SecSpec_log10),np.max(SecSpec_log10))
        
        #draw the plot
        ax.set_xlim([dict_subplot["psin_min"],dict_subplot["psin_max"]])
        ax.set_ylim([dict_subplot["psip_min"],dict_subplot["psip_max"]])
        im = ax.pcolormesh(psin-offset_n,psip-offset_p,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\Psi_- = f_t/\nu_0 \propto \theta_1-\theta_2$")
        ax.set_ylabel(r"$\Psi_+ = f_\nu\nu_0^2/f_t \propto \theta_1+\theta_2$")
        
        print("Successfully plotted linearized secondary spectrum.")
        
    def compute_fit_LinSecSpec(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_fitLinSecSpec.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"psip_bounds":[0.1e+15,2.5e+15],"psin_bounds":[0.15e-09,0.8e-09],"psin_fiducial":-0.25e-09}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        psip_bounds = dict_specs["psip_bounds"]
        psin_bounds = dict_specs["psin_bounds"]
        psin_fiducial = dict_specs["psin_fiducial"]
        
        #load and check data
        # - linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"linear_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute linear_secondary_spectrum' first!")
            return 0
        file_LinSecSpec = os.path.join(path_results,"LinSecSpec.npy")
        file_psip = os.path.join(path_results,"psip.npy")
        file_psin = os.path.join(path_results,"psin.npy")
        LinSecSpec = np.load(file_LinSecSpec)
        psip = np.load(file_psip)
        psin = np.load(file_psin)
        
        #define files to compute
        file_slope = os.path.join(path_computation,"slope.npy")
        file_scale = os.path.join(path_computation,"scale.npy")
        
        #create containers
        # - cut parts that are not fitted
        psip_mask = np.where((psip>-psip_bounds[1]) & (psip<psip_bounds[1]))[0]
        psin_mask = np.where((psin>psin_bounds[0]) & (psin<psin_bounds[1]))[0]
        Spec = LinSecSpec[np.ix_(psin_mask,psip_mask)]
        psip = psip[psip_mask]
        psin = psin[psin_mask]
        psip_flags = np.where((psip>-psip_bounds[0]) & (psip<psip_bounds[0]))[0]
        idx_fiducial = (np.abs(psin - psin_fiducial)).argmin()
        mu_fid = Spec[idx_fiducial,:]
        mu_fid[psip_flags] = np.median(mu_fid)
        # - prepare arrays of results
        slope = np.zeros(psin.shape,float)
        scale = np.zeros(psin.shape,float)
        
        # define fit function
        factor_a = (psip_bounds[1]-psip_bounds[0])/10.
        def fitfunc(in_mu,in_a,in_b):
            result = np.copy(in_mu)
            result[psip_flags] = np.median(result)
            filler = np.median(result)
            ip_mu = interpolate.interp1d(psip,result,kind='linear',fill_value='extrapolate')
            result = ip_mu(psip+in_a*factor_a)*in_b
            result[psip_flags] = np.median(result)
            return result
        
        #perform the computation
        time_start = time.time()
        print("Fitting linearized secondary spectrum ...")
        # - perform the fit
        a_max = 10.
        b_max = np.max(Spec)/np.min(Spec)
        for i_psin,v_psin in enumerate(psin):
            if not v_psin==psin_fiducial:
                mu = Spec[i_psin,:]
                mu[psip_flags] = np.median(mu)
                popt, pcov = curve_fit(fitfunc,mu,mu_fid,p0=[2.5e+24*(v_psin-psin_fiducial)/factor_a,1.],bounds=([1.0e+24*(v_psin-psin_fiducial)/factor_a,0.],[4.0e+24*(v_psin-psin_fiducial)/factor_a,b_max]),maxfev=1000)
                #perr = np.sqrt(np.diag(pcov))
                slope[i_psin] = popt[0]/(v_psin-psin_fiducial)*factor_a
                scale[i_psin] = popt[1]
            else:
                slope[i_psin] = 0.
                scale[i_psin] = 1.
#        bar = progressbar.ProgressBar(maxval=self.N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#        bar.start()
#        for i_f_t,v_f_t in enumerate(f_t):
#            bar.update(i_f_t)
#            for i_psip,v_psip in enumerate(psip):
#                v_f_nu = v_psip*v_f_t/self.nu_half**2
#                LinSecSpec[i_f_t,i_psip] = ip_SecSpec(v_f_nu,v_f_t)
#        bar.finish()
        
        #save the results
        np.save(file_slope,slope)
        np.save(file_scale,scale)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully fitted the linearized secondary spectrum.".format(time_current))
        
    def plot_Hist_fitLinSecSpec(self,dict_plot,figure,ax,tag):
        #load and check data
        # - Hough transformed linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"fit_LinSecSpec")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute fit_LinSecSpec' first!")
            return 0
        file_slope = os.path.join(path_results,"slope.npy")
        file_scale = os.path.join(path_results,"scale.npy")
        slope = np.load(file_slope)
        scale = np.load(file_scale)
        # inform about scales
        print("slope [{0},{1}], scale [{2},{3}]".format(np.min(slope),np.max(slope),np.min(scale),np.max(scale)))
        print(np.median(slope),4.*math.pi/self.LightSpeed/self.nu_half*(319.0*1000.*np.cos(np.deg2rad(-16.0)))**2/self.pc*np.median(slope))
        
        #load specifications
        key_sepcs = "plot_Hist_fitLinSecSpec{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"slope_min":float(np.min(slope)),"slope_max":float(np.max(slope)),"nbins":10,
                            "title":"curvature"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #draw the plot
        ax.hist(slope,bins=dict_subplot["nbins"],range=(dict_subplot["slope_min"],dict_subplot["slope_max"]))
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"slope [au]")
        ax.set_ylabel(r"counts")
        
        print("Successfully plotted fitted slope of linearized secondary spectrum.")
        
    def compute_AC_LinSecSpec(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_AC_LinSecSpec.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"psip_max":3.5e+15,"psin_max":1.0e-09,"psip_sampling":10,"psin_sampling":10,"vmin":12.0,"vmax":15.5}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        psip_max = dict_specs["psip_max"]
        psin_max = dict_specs["psin_max"]
        psip_sampling = dict_specs["psip_sampling"]
        psin_sampling = dict_specs["psin_sampling"]
        vmin = dict_specs["vmin"]
        vmax = dict_specs["vmax"]
        
        #load and check data
        # - linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"linear_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute linear_secondary_spectrum' first!")
            return 0
        file_LinSecSpec = os.path.join(path_results,"LinSecSpec.npy")
        file_psip = os.path.join(path_results,"psip.npy")
        file_psin = os.path.join(path_results,"psin.npy")
        LinSecSpec = np.load(file_LinSecSpec)
        psip = np.load(file_psip)
        psin = np.load(file_psin)
#        N_psip = len(psip)
#        N_psin = len(psin)
        
        #define files to compute
        file_autocorr = os.path.join(path_computation,"autocorr.npy")
        file_sepp = os.path.join(path_computation,"separation_psip.npy")
        file_sepn = os.path.join(path_computation,"separation_psin.npy")
        
        #create containers
#        autocorr = np.zeros((2*N_psin-1,2*N_psip-1),dtype=float)
#        N_sepn = 2*N_psin-1
#        sepn = np.linspace(psin[0]-psin[-1],psin[-1]-psin[0],num=N_sepn)
#        N_sepp = 2*N_psip-1
#        sepp = np.linspace(psip[0]-psip[-1],psip[-1]-psip[0],num=N_sepp)
        psip_mask = np.where((psip>-psip_max) & (psip<psip_max))[0]
        psin_mask = np.where((psin>-psin_max) & (psin<psin_max))[0]
        LinSecSpec = LinSecSpec[np.ix_(psin_mask,psip_mask)]
        psip = psip[psip_mask]
        psin = psin[psin_mask]
        autocorr = np.zeros(LinSecSpec.shape,dtype=float)
        sepn = psin-np.mean(psin)
        sepp = psip-np.mean(psip)
        # - downsampling
        LinSecSpec = block_reduce(LinSecSpec, block_size=(psin_sampling,psip_sampling), func=np.mean)
        coordinates = np.array([sepn,sepn])
        coordinates = block_reduce(coordinates, block_size=(1,psin_sampling), func=np.mean, cval=sepn[-1])
        sepn = coordinates[0,:]
        coordinates = np.array([sepp,sepp])
        coordinates = block_reduce(coordinates, block_size=(1,psip_sampling), func=np.mean, cval=sepp[-1])
        sepp = coordinates[0,:]
        
        #perform the computation
        time_start = time.time()
        print("Computing autocorrelation ...")
        # - apply logarithmic scale
        min_nonzero = np.min(LinSecSpec[np.nonzero(LinSecSpec)])
        LinSecSpec[LinSecSpec == 0] = min_nonzero
        Spec = np.log10(LinSecSpec)
        # - normalize spectra
#        sortened = np.sort(Spec,axis=None)
#        Num = len(sortened)
#        Lowest = sortened[int(0.1*Num)]
#        Highest = sortened[int(0.9*Num)]
        Lowest = vmin
        Highest = vmax
        Spec[Spec<Lowest] = Lowest
        Spec[Spec>Highest] = Highest
        Spec = (Spec-np.min(Spec))/np.std(Spec)
        # - compute autocorrelation
        autocorr = signal.correlate2d(Spec,Spec,mode="same")
        
        #save the results
        np.save(file_autocorr,autocorr)
        np.save(file_sepp,sepp)
        np.save(file_sepn,sepn)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the autocorrelation of the linearized secondary spectrum.".format(time_current))
        
    def plot_AC_LinSecSpec(self,dict_plot,figure,ax,tag):
        #load and check data
        # - Hough transformed linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"AC_LinSecSpec")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute AC_LinSecSpec' first!")
            return 0
        file_autocorr = os.path.join(path_results,"autocorr.npy")
        file_sepp = os.path.join(path_results,"separation_psip.npy")
        file_sepn = os.path.join(path_results,"separation_psin.npy")
        autocorr = np.load(file_autocorr)
        sepp = np.load(file_sepp)
        sepn = np.load(file_sepn)
        # inform about scales
        print("sepp [{0},{1}], sepn [{2},{3}], autocorr [{4},{5}]".format(sepp[0],sepp[-1],sepn[0],sepn[-1],np.min(autocorr),np.max(autocorr)))
        
        #load specifications
        key_sepcs = "plot_AC_LinSecSpec{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":20.,"sepp_min":None,"sepp_max":None,"sepn_min":None,"sepn_max":None,
                            "title":"AC of linearized secondary spectrum"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - compute offsets to center pccolormesh
        offset_p = (sepp[1]-sepp[0])/2.
        offset_n = (sepn[1]-sepn[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["sepn_min"],dict_subplot["sepn_max"]])
        ax.set_ylim([dict_subplot["sepp_min"],dict_subplot["sepp_max"]])
        im = ax.pcolormesh(sepn-offset_n,sepp-offset_p,map(list, zip(*autocorr)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\Delta \Psi_-$")
        ax.set_ylabel(r"$\Delta \Psi_+$")
        
        print("Successfully plotted autocorrelation of linearized secondary spectrum.")
        
    def compute_FFT_linsecspec(self,path_computation):
        #load and check data
        # - linearized secondary spectrum
        path_results = os.path.join(self.path_computations,"linear_secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute linear_secondary_spectrum' first!")
            return 0
        file_LinSecSpec = os.path.join(path_results,"LinSecSpec.npy")
        file_psip = os.path.join(path_results,"psip.npy")
        file_psin = os.path.join(path_results,"psin.npy")
        LinSecSpec = np.load(file_LinSecSpec)
        psip = np.load(file_psip)
        psin = np.load(file_psin)
        
        #crop data
        psip_min = -3.5e+15 #0.4e+15
        psip_max = 3.5e+15
        psin_min = 0.1e-09 #-1.0e+09
        psin_max = 1.0e+09
        psip_mask = np.where((psip>psip_min) & (psip<psip_max))[0]
        psin_mask = np.where((psin>psin_min) & (psin<psin_max))[0]
        LinSecSpec = LinSecSpec[np.ix_(psin_mask,psip_mask)]
        psip = psip[psip_mask]
        psin = psin[psin_mask]
        N_psip = len(psip)
        N_psin = len(psin)
        
        #define files to compute
        file_power = os.path.join(path_computation,"power.npy")
        file_fp = os.path.join(path_computation,"fp.npy")
        file_fn = os.path.join(path_computation,"fn.npy")
        
        #create containers
        delta_p = psip[1]-psip[0]
        delta_n = psin[1]-psin[0]
        power = np.zeros((N_psin,N_psip),dtype=float)
        fp = np.linspace(-math.pi/delta_p,math.pi/delta_p,num=N_psip,endpoint=False)
        fn = np.linspace(-math.pi/delta_n,math.pi/delta_n,num=N_psin,endpoint=False)
        
        #perform the computation
        time_start = time.time()
        print("Computing power spectrum ...")
        # - apply logarithmic scale
        min_nonzero = np.min(LinSecSpec[np.nonzero(LinSecSpec)])
        LinSecSpec[LinSecSpec == 0] = min_nonzero
        Spec = np.log10(LinSecSpec)
        # - normalize spectrum
        #Spec[Spec>15.5] = 15.5
        Spec = cv2.normalize(Spec, dst=Spec, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        selem = morphology.disk(30)
        Spec = skimage.filters.rank.equalize(Spec,selem=selem)
        # - compute the power spectrum
        A_Spec = np.fft.fft2(Spec)
        A_Spec = np.fft.fftshift(A_Spec)
        power = np.abs(A_Spec)**2
        #phase = np.angle(A_DynSpec)
        
        #save the results
        np.save(file_power,power)
        np.save(file_fp,fp)
        np.save(file_fn,fn)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the power spectrum of the linearized secondary spectrum.".format(time_current))
        
    def plot_FFT_linsecspec(self,dict_plot,figure,ax,tag):
        #load and check data
        # - FFT_linsecspec
        path_computation = os.path.join(self.path_computations,"FFT_linsecspec")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute FFT_linsecspec' first!")
            return 0
        file_power = os.path.join(path_computation,"power.npy")
        file_fn = os.path.join(path_computation,"fn.npy")
        file_fp = os.path.join(path_computation,"fp.npy")
        power = np.load(file_power)
        fn = np.load(file_fn)
        fp = np.load(file_fp)
        
        #load specifications
        key_sepcs = "plot_FFT_linsecspec{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":float(np.nanmin(np.log10(power))),"vmax":float(np.nanmax(np.log10(power))),"fn_min":float(fn[0]),"fn_max":float(fn[-1]),"fp_min":float(fp[0]),"fp_max":float(fp[-1]),
                            "title":"FFT Linear Spectrum","fn_sampling":1,"fp_sampling":1}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
#        # - sparate 2pi from the Fourier frequency
#        fn /= 2.*math.pi
#        fp /= 2.*math.pi
        # - read in specifications
        min_index_t = 0
        max_index_t = len(fn)-1
        for index_t in range(len(fn)):
            if fn[index_t]<dict_subplot["fn_min"]:
                min_index_t = index_t
            elif fn[index_t]>dict_subplot["fn_max"]:
                max_index_t = index_t
                break
        fn = fn[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(fp)-1
        for index_nu in range(len(fp)):
            if fp[index_nu]<dict_subplot["fp_min"]:
                min_index_nu = index_nu
            elif fp[index_nu]>dict_subplot["fp_max"]:
                max_index_nu = index_nu
                break
        fp = fp[min_index_nu:max_index_nu+1]
        Spec = power[min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        fn_sampling = dict_subplot["fn_sampling"]
        fp_sampling = dict_subplot["fp_sampling"]
        Spec = block_reduce(Spec, block_size=(fn_sampling,fp_sampling), func=np.mean)
        coordinates = np.array([fn,fn])
        coordinates = block_reduce(coordinates, block_size=(1,fn_sampling), func=np.mean, cval=fn[-1])
        fn = coordinates[0,:]
        coordinates = np.array([fp,fp])
        coordinates = block_reduce(coordinates, block_size=(1,fp_sampling), func=np.mean, cval=fp[-1])
        fp = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_fn = (fn[1]-fn[0])/2.
        offset_fp = (fp[1]-fp[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(Spec[np.nonzero(Spec)])
        Spec[Spec == 0] = min_nonzero
        # - apply logarithmic scale
        Spec_log10 = np.log10(Spec)
        
        #draw the plot
        ax.set_xlim([dict_subplot["fn_min"],dict_subplot["fn_max"]])
        ax.set_ylim([dict_subplot["fp_min"],dict_subplot["fp_max"]])
        im = ax.pcolormesh(fn-offset_fn,fp-offset_fp,map(list, zip(*Spec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$f_-$")
        ax.set_ylabel(r"$f_+$")
        
        print("Successfully plotted the power spectrum of the linearized secondary spectrum.")
        
    def compute_FFT_linsecspec_weights(self,path_computation):
        #load and check data
        # - FFT_linsecspec
        path_results = os.path.join(self.path_computations,"FFT_linsecspec")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute FFT_linsecspec' first!")
            return 0
        file_power = os.path.join(path_results,"power.npy")
        file_fn = os.path.join(path_results,"fn.npy")
        file_fp = os.path.join(path_results,"fp.npy")
        power = np.load(file_power)
        fn = np.load(file_fn)
        fp = np.load(file_fp)
        
        #define files to compute
        file_weights = os.path.join(path_computation,"weights.npy")
        file_slopes = os.path.join(path_computation,"slopes.npy")
        
        #crop data
        min_fn = 0.6e+11
        min_fp = np.min(fp)
        fn_mask = np.where((fn>=min_fn))[0]
        fp_mask = np.where((fp>=min_fp))[0]
        power = power[np.ix_(fn_mask,fp_mask)]
        fp = fp[fp_mask]
        fn = fn[fn_mask]
        N_fp = len(fp)
        N_fn = len(fn)
        
        #perform the computation
        time_start = time.time()
        print("Summing pixels ...")
        slopes = np.zeros(fp.shape,dtype=float)
        for i_fp,v_fp in enumerate(fp):
            slopes[i_fp] = v_fp/fn[-1]
        weight = np.zeros(slopes.shape,dtype=float)
        bar = progressbar.ProgressBar(maxval=N_fn, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_fn,v_fn in enumerate(fn):
            bar.update(i_fn)
            for i_slope,v_slope in enumerate(slopes):
                i_fp = int((v_slope*v_fn-fp[0])/(fp[-1]-fp[0])*N_fp-0.5)
                weight[i_slope] += power[i_fn,i_fp]
        bar.finish()
        
        #save the results
        np.save(file_weights,weight)
        np.save(file_slopes,slopes)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the weights of slopes.".format(time_current))
        
    def plot_FFT_linsecspec_weights(self,dict_plot,figure,ax,tag):
        #load and check data
        # - FFT weights
        path_results = os.path.join(self.path_computations,"FFT_linsecspec_weights")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute FFT_linsecspec_weights' first!")
            return 0
        file_weights = os.path.join(path_results,"weights.npy")
        file_slopes = os.path.join(path_results,"slopes.npy")
        weights = np.load(file_weights)
        slopes = np.load(file_slopes)
        #inform about scales
        print("weights:[{0},{1}], slopes:[{2},{3}]".format(np.min(weights),np.max(weights),slopes[0],slopes[-1]))
        
        #load specifications
        key_sepcs = "plot_FFT_linsecspec_weights{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"sum_min":None,"sum_max":None, "cutoff":0.,
                            "title":"weights of FFT slopes"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #find peaks
        cutoff = dict_subplot["cutoff"]
        mpeak = np.max(weights[slopes<-cutoff])
        ppeak = np.max(weights[slopes>cutoff])
        mpeak = slopes[np.where(weights==mpeak)[0]][0]
        ppeak = slopes[np.where(weights==ppeak)[0]][0]
        list_peak = [np.abs(mpeak),np.abs(ppeak)]
        print(mpeak,ppeak,np.mean(list_peak),np.std(list_peak))
        eta = 1./(np.mean(list_peak)*self.nu_half**3)*2.*math.pi
        sig_eta = np.std(list_peak)/(np.mean(list_peak)**2*self.nu_half**3)*2.*math.pi
        print("--> slope eta={0} +/- {1} s^3".format(eta,sig_eta))
        
        #draw the plot
        ax.set_xlim([slopes[0],slopes[-1]])
        ax.set_ylim([dict_subplot["sum_min"],dict_subplot["sum_max"]])
        ax.plot(slopes,weights,label=r"FFT slope weight")
        ax.axvline(dict_subplot["cutoff"],color="red",linestyle='dashed')
        ax.axvline(-dict_subplot["cutoff"],color="red",linestyle='dashed')
        ax.axvline(mpeak,color="blue",linestyle='dashed')
        ax.axvline(ppeak,color="blue",linestyle='dashed')
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel("slope $f_+/f_-$")
        ax.set_ylabel(r"weight")
        plt.grid(True)
        
        print("Successfully plotted the weights of slopes.")
        
#-----AUTOCORRELATION-----
        
    def compute_autocorrelation(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        
        #define files to compute
        file_autocorr = os.path.join(path_computation,"autocorr.npy")
        file_sept = os.path.join(path_computation,"separation_t.npy")
        file_sepnu = os.path.join(path_computation,"separation_nu.npy")
        
        #create containers
        autocorr = np.zeros((self.N_p,self.N_p,2*self.N_t-1,2*self.N_nu-1),dtype=float)
        N_sept = 2*self.N_t-1
        sept = np.linspace(self.t[0]-self.t[-1],self.t[-1]-self.t[0],num=N_sept)
        N_sepnu = 2*self.N_nu-1
        sepnu = np.linspace(self.nu[0]-self.nu[-1],self.nu[-1]-self.nu[0],num=N_sepnu)
        #counts = np.zeros((N_sept,N_sepnu),dtype=int)
        
        #perform the computation
        time_start = time.time()
        print("Computing autocorrelation ...")
        for i_p0 in range(self.N_p):
            for i_p1 in range(self.N_p):
                print("For telescope combination ({0},{1})".format(i_p0,i_p1))
                # - normalize spectra
                Spec = np.abs(DynSpec[i_p0,i_p1,:,:])
                Spec[flags] = np.nan
                Spec = (Spec-np.nanmean(Spec))/np.nanstd(Spec)
                # - perform the multiplication
                Spec[flags] = 0.
                autocorr[i_p0,i_p1,:,:] = signal.correlate2d(Spec,Spec)
#                bar = progressbar.ProgressBar(maxval=N_sept, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#                bar.start()
#                for i_sept in xrange(N_sept):
#                    bar.update(i_sept)
#                    for i_sepnu in xrange(N_sepnu):
#                        # - produce a shifted copy of the spectrum
#                        tshift = i_sept-self.N_t+1
#                        nushift = i_sepnu-self.N_nu+1
#                        #print(tshift,nushift)
#                        #shiftspec = shift(Spec,(tshift,nushift),cval=np.nan)
#                        
##                        shiftspec = np.empty_like(Spec)
##                        if tshift==0:
##                            if nushift > 0:
##                                shiftspec[:,:nushift] = np.nan
##                                shiftspec[:,nushift:] = Spec[:,:-nushift]
##                            elif nushift < 0:
##                                shiftspec[:,nushift:] = np.nan
##                                shiftspec[:,:nushift] = Spec[:,-nushift:]
##                            else:
##                                shiftspec[:,:] = Spec
##                        elif nushift==0:
##                            if tshift > 0:
##                                shiftspec[:tshift,:] = np.nan
##                                shiftspec[tshift:,:] = Spec[:-tshift,:]
##                            elif tshift < 0:
##                                shiftspec[tshift:,:] = np.nan
##                                shiftspec[:tshift,:] = Spec[-tshift:,:]
##                        elif tshift > 0 and nushift > 0:
##                            shiftspec[:tshift,:nushift] = np.nan
##                            shiftspec[tshift:,nushift:] = Spec[:-tshift,:-nushift]
##                        elif tshift < 0 and nushift < 0:
##                            shiftspec[tshift:,nushift:] = np.nan
##                            shiftspec[:tshift,:nushift] = Spec[-tshift:,-nushift:]
##                        elif tshift > 0 and nushift < 0:
##                            shiftspec[:tshift,nushift:] = np.nan
##                            shiftspec[tshift:,:nushift] = Spec[:-tshift,-nushift:]
##                        elif tshift < 0 and nushift > 0:
##                            shiftspec[tshift:,:nushift] = np.nan
##                            shiftspec[:tshift,nushift:] = Spec[-tshift:,:-nushift]
##                        # - take the mean of their products
##                        autocorr[i_p0,i_p1,i_sept,i_sepnu] = np.nanmean(Spec*shiftspec)
#                bar.finish()
        
        #save the results
        np.save(file_autocorr,autocorr)
        np.save(file_sept,sept)
        np.save(file_sepnu,sepnu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the autocorrelation.".format(time_current))
        
    def plot_autocorrelation(self,dict_plot,figure,ax,tag):
        #load specifications
        key_sepcs = "plot_autocorrelation{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":None,"vmax":None,"sept_min":-float(self.t[-1]-self.t[0]),"sept_max":float(self.t[-1]-self.t[0]),"sepnu_min":-float(self.nu[-1]-self.nu[0]),"sepnu_max":float(self.nu[-1]-self.nu[0]),
                            "telescope1":0,"telescope2":0,"title":"autocorrelation","source":"autocorrelation"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #load and check data
        # - set source
        source = dict_subplot["source"] # "autocorrelation","autocorr_from_secspec"
        # - autocorrelation
        path_results = os.path.join(self.path_computations,source)
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute {0}' first!".format(source))
            return 0
        file_autocorr = os.path.join(path_results,"autocorr.npy")
        file_sept = os.path.join(path_results,"separation_t.npy")
        file_sepnu = os.path.join(path_results,"separation_nu.npy")
        autocorr = np.load(file_autocorr)
        sept = np.load(file_sept)
        sepnu = np.load(file_sepnu)
        
        #Preprocess data
        # - enter choice of telescopes
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        spec = autocorr[tel1,tel2,:,:]
        # - compute offsets to center pccolormesh
        offset_t = (sept[1]-sept[0])/2.
        offset_nu = (sepnu[1]-sepnu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["sept_min"],dict_subplot["sept_max"]])
        ax.set_ylim([dict_subplot["sepnu_min"],dict_subplot["sepnu_max"]])
        im = ax.pcolormesh(sept-offset_t,sepnu-offset_nu,map(list, zip(*spec)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(r"$\Delta \nu$")
        
        print("Successfully plotted the autocorrelation.")
        
    def compute_secspec_from_autocorr(self,path_computation):
        #load and check data
        # - autocorrelation
        path_results = os.path.join(self.path_computations,"autocorrelation")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute autocorrelation' first!")
            return 0
        file_autocorr = os.path.join(path_results,"autocorr.npy")
        file_sept = os.path.join(path_results,"separation_t.npy")
        file_sepnu = os.path.join(path_results,"separation_nu.npy")
        autocorr = np.load(file_autocorr)
        sept = np.load(file_sept)
        sepnu = np.load(file_sepnu)
        N_sept = len(sept)
        N_sepnu = len(sepnu)
        
        #define files to compute
        file_power = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        
        #create containers
        SecSpec = np.zeros((self.N_p,self.N_p,N_sept,N_sepnu),dtype=float)
        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=N_sept,endpoint=False)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=N_sepnu,endpoint=False)
        
        #perform the computation
        # - main computation
        time_start = time.time()
        print("Computing secondary spectrum ...")
        # - compute the power spectrum
        A_DynSpec = np.fft.fft2(autocorr,axes=(2,3))
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(2,3))
        SecSpec = self.delta_nu**2*self.delta_t**2*np.abs(A_DynSpec)/4.
        
        #save the results
        np.save(file_power,SecSpec)
        np.save(file_doppler,f_t)
        np.save(file_delay,f_nu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the secondary spectrum.".format(time_current))
        
    def plot_secspec_from_autocorr(self,dict_plot,figure,ax,tag):
        #load specifications
        key_sepcs = "plot_secspec_from_autocorr{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":float(np.max(np.real(SecSpec))),"f_t_min":float(-math.pi/self.delta_t),"f_t_max":float(math.pi/self.delta_t),"f_nu_min":0.,"f_nu_max":float(math.pi/self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"secondary spectrum (log10) from autocorrelation","source":"secspec_from_autocorr"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #load and check data
        # - set source
        source = dict_subplot["source"] # "secspec_from_autocorr","autocorr_secspec_mean"
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,source)
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute {0}' first!".format(source))
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #Preprocess data
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        SecSpec = SecSpec[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(SecSpec[np.nonzero(SecSpec)])
        SecSpec[SecSpec == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(SecSpec)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [Hz]") #(r"$f_x$ [au]") #
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]") #(r"$f_y$ [au]") #
        
        print("Successfully plotted secondary spectrum computed from autocorrelation.")
        
    def compute_autocorr_from_secspec(self,path_computation):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #define files to compute
        file_autocorr = os.path.join(path_computation,"autocorr.npy")
        file_sept = os.path.join(path_computation,"separation_t.npy")
        file_sepnu = os.path.join(path_computation,"separation_nu.npy")
        
        #create containers
        autocorr = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        delta_ft = f_t[1]-f_t[0]
        delta_fnu = f_nu[1]-f_nu[0]
        sept = np.linspace(-math.pi/delta_ft,math.pi/delta_ft,num=self.N_t,endpoint=False)
        sepnu = np.linspace(-math.pi/delta_fnu,math.pi/delta_fnu,num=self.N_nu,endpoint=False)
        
        #perform the computation
        time_start = time.time()
        print("Computing autocorrelation ...")
        # - perform Fourier transform
        A_SecSpec = np.conj(np.fft.fft2(SecSpec,axes=(2,3)))
        A_SecSpec = np.fft.fftshift(A_SecSpec,axes=(2,3))
        autocorr = np.abs(A_SecSpec)/(2.*math.pi)**2/self.delta_nu**2/self.delta_t**2
        
        #save the results
        np.save(file_autocorr,autocorr)
        np.save(file_sept,sept)
        np.save(file_sepnu,sepnu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the autocorrelation.".format(time_current))
        
    def compute_autocorr_secspec_mean(self,path_computation):
        #load and check data
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        # - secondary spectrum from autocorrelation
        path_results = os.path.join(self.path_computations,"secspec_from_autocorr")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secspec_from_autocorr' first!")
            return 0
        file_power_a = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler_a = os.path.join(path_results,"doppler.npy")
        file_delay_a = os.path.join(path_results,"delay.npy")
        SecSpec_a = np.load(file_power_a)
        f_t_a = np.load(file_doppler_a)
        f_nu_a = np.load(file_delay_a)
        
        #define files to compute
        file_power_mean = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        
        #create containers
#        SecSpec_mean = np.copy(SecSpec)
#        counts = np.ones((self.N_t,self.N_nu),dtype=int)
        SecSpec_mean = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        counts = np.zeros((self.N_t,self.N_nu),dtype=int)
        
        #perform the computation
        time_start = time.time()
        print("Computing averaged secondary spectrum ...")
        # - normalize spectrum from autocorrelation
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                SecSpec_a[i_p1,i_p2,:,:] = SecSpec_a[i_p1,i_p2,:,:]*np.mean(SecSpec[i_p1,i_p2,:,:])/np.mean(SecSpec_a[i_p1,i_p2,:,:])
        # binning
        d_ft = (f_t[1]-f_t[0])/2.
        d_fnu = (f_nu[1]-f_nu[0])/2.
        i_fnu = 0
        v_fnu = f_nu[i_fnu]
        for i_fnua,v_fnua in enumerate(f_nu_a):
            while abs(v_fnua-v_fnu)>d_fnu:
                i_fnu += 1
                v_fnu = f_nu[i_fnu]
            i_ft = 0
            v_ft = f_t[i_ft]
            for i_fta,v_fta in enumerate(f_t_a):
                while abs(v_fta-v_ft)>d_ft:
                    i_ft += 1
                    v_ft = f_t[i_ft]
                SecSpec_mean[:,:,i_ft,i_fnu] += SecSpec_a[:,:,i_fta,i_fnua]
                counts[i_ft,i_fnu] += 1
        counts[counts==0] = 1
        SecSpec_mean[:,:,:,:] /= counts[na,na,:,:]            
        
        #save the results
        np.save(file_power_mean,SecSpec_mean)
        np.save(file_doppler,f_t)
        np.save(file_delay,f_nu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the averaged secondary spectrum (classic + from autocorrelation).".format(time_current))
        
#-----IMAGING-----
        
    def compute_dirty_image(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_dirty_image.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"N_thx":50,"thx_max":30.,"thx_min":-30.,"N_thy":50,"thy_max":30.,"thy_min":-30.,"veff":305.,"veffa":86.,"Deff":1170.}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        N_thx = dict_specs["N_thx"]
        thx_max = dict_specs["thx_max"]*self.mas
        thx_min = dict_specs["thx_min"]*self.mas
        N_thy = dict_specs["N_thy"]
        thy_max = dict_specs["thy_max"]*self.mas
        thy_min = dict_specs["thy_min"]*self.mas
        Deff = dict_specs["Deff"]*self.pc
        veffx = dict_specs["veff"]*1000.*np.cos(math.pi/180.*dict_specs["veffa"])
        veffy = dict_specs["veff"]*1000.*np.sin(math.pi/180.*dict_specs["veffa"])
        
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
        
        #define files to compute
        file_dirty_image = os.path.join(path_computation,"dirty_image.npy")
        file_xaxis_theta = os.path.join(path_computation,"xaxis_theta.npy")
        file_yaxis_theta = os.path.join(path_computation,"yaxis_theta.npy")
        
        #create containers
        t = np.copy(self.t)
        dirty_image = np.zeros((N_thx,N_thy),dtype=float)
        thx = np.linspace(thx_min,thx_max,num=N_thx,endpoint=True)
        thy = np.linspace(thy_min,thy_max,num=N_thy,endpoint=True)
        a_thx = -2.*math.pi*self.nu_half/self.LightSpeed*thx
        a_thy = -2.*math.pi*self.nu_half/self.LightSpeed*thy
        rate = -2.*math.pi*self.nu_half/self.LightSpeed*(thx[:,na]*veffx+thy[na,:]*veffy)
        delay = math.pi/self.LightSpeed*Deff*(thx[:,na]**2+thy[na,:]**2)
        # - save fixed results
        np.save(file_xaxis_theta,thx)
        np.save(file_yaxis_theta,thy)
        
        #Remove flagged data (full time axes)
        N_t = self.N_t
        try:
            for i_t in xrange(self.N_t):
                if flags[i_t,0]:
                    np.delete(t,i_t)
                    np.delete(DynSpec,i_t,axis=2)
                    np.delete(p,i_t,axis=1)
                    N_t -= 1
        except:
            # - if old format
            np.delete(t,flags)
            np.delete(DynSpec,flags,axis=2)
            np.delete(p,flags,axis=1)
            N_t -= len(flags)
        
        #perform the computation
        time_start = time.time()
        print("Computing a dirty image of the screen ...")
        # - preparations
        I_ptn = np.zeros((self.N_p,N_t,self.N_nu),dtype=np.complex)
        # - main loop
        bar = progressbar.ProgressBar(maxval=N_thx, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_thx1 in xrange(N_thx):
            bar.update(i_thx1)
            for i_thy1 in xrange(N_thy):
                I_ptn = np.sum(DynSpec*np.exp(-1j*(a_thx[i_thx1]*p[:,na,:,na,0] + a_thy[i_thy1]*p[:,na,:,na,1])),axis=0)
                dummy_image = np.zeros((N_thx,N_thy),dtype=np.complex)
                for i_thx2 in xrange(N_thx):
                    for i_thy2 in xrange(N_thy):
                        dummy_image[i_thx2,i_thy2] = np.sum(I_ptn*np.exp(-1j*( (rate[i_thx1,i_thy1]-rate[i_thx2,i_thy2])*t[na,:,na] + (delay[i_thx1,i_thy1]-delay[i_thx2,i_thy2])*self.nu[na,na,:] - (a_thx[i_thx2]*p[:,:,na,0] + a_thy[i_thy2]*p[:,:,na,1]) )))
                dirty_image[i_thx1,i_thy1] = np.sum(np.abs(dummy_image))
                # - save image pointwise to control output in real time
                np.save(file_dirty_image,dirty_image)
        bar.finish()
        
        #save the results
        np.save(file_dirty_image,dirty_image)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed a dirty image of the screen.".format(time_current))
        
    def plot_dirty_image(self,dict_plot,figure,ax,tag):
        #load and check data
        # - dirty image
        source = "dirty_image"
        path_results = os.path.join(self.path_computations,source)
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute {0}' first!".format(source))
            return 0
        file_dirty_image = os.path.join(path_results,"dirty_image.npy")
        file_xaxis_theta = os.path.join(path_results,"xaxis_theta.npy")
        file_yaxis_theta = os.path.join(path_results,"yaxis_theta.npy")
        dirty_image = np.load(file_dirty_image)
        thx = np.load(file_xaxis_theta)/self.mas
        thy = np.load(file_yaxis_theta)/self.mas
        
        #load specifications
        key_sepcs = "plot_dirty_image{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":float(np.max(dirty_image)),
                            "title":"dirty image (log10)"}
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
            
        #Preprocess data
        # - compute offsets to center pccolormesh
        offset_thx = (thx[1]-thx[0])/2.
        offset_thy = (thy[1]-thy[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(dirty_image[np.nonzero(dirty_image)])
        dirty_image[dirty_image == 0] = min_nonzero
        # - apply logarithmic scale
        dirty_image_log10 = np.log10(dirty_image)
        # - inform about variation scale
        print("image std = {0}".format(np.std(dirty_image_log10)))
        
        #draw the plot
        ax.set_xlim([thx[0],thx[-1]-offset_thx])
        ax.set_ylim([thy[0],thx[-1]-offset_thy])
        im = ax.pcolormesh(thx-offset_thx,thy-offset_thy,map(list, zip(*dirty_image_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"$\theta_x$ [mas]")
        ax.set_ylabel(r"$\theta_y$ [mas]")
        
        print("Successfully plotted dirty image.")
        
    def compute_theta_transform(self,path_computation):
        #load specifications
        file_specs = os.path.join(path_computation,"specs_theta_transform.yaml")
        if not os.path.exists(file_specs):
            dict_specs = {"N_thx":50,"thx_max":30.,"thx_min":-30.,"N_thy":50,"thy_max":30.,"thy_min":-30.,"veff":305.,"veffa":86.,"Deff":1170.}
            with open(file_specs,'w') as writefile:
                self.yaml.dump(dict_specs,writefile)
        else:
            with open(file_specs,'r') as readfile:
                dict_specs =self.yaml.load(readfile)
        N_thx = dict_specs["N_thx"]
        thx_max = dict_specs["thx_max"]*self.mas
        thx_min = dict_specs["thx_min"]*self.mas
        N_thy = dict_specs["N_thy"]
        thy_max = dict_specs["thy_max"]*self.mas
        thy_min = dict_specs["thy_min"]*self.mas
        Deff = dict_specs["Deff"]*self.pc
        veffx = dict_specs["veff"]*1000.*np.cos(math.pi/180.*dict_specs["veffa"])
        veffy = dict_specs["veff"]*1000.*np.sin(math.pi/180.*dict_specs["veffa"])
        
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        p = np.load(os.path.join(self.path_data,"positions.npy"))
        p[:,:,0] *= -1.
        # - secondary spectrum
        path_results = os.path.join(self.path_computations,"secondary_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute secondary_spectrum' first!")
            return 0
        #file_power = os.path.join(path_results,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_results,"doppler.npy")
        file_delay = os.path.join(path_results,"delay.npy")
        #SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #define files to compute
        file_theta_transform = os.path.join(path_computation,"theta_transform.npy")
        file_ar_theta = os.path.join(path_computation,"ar_theta.npy")
        
        #create containers
        theta_transform = np.zeros((N_thx,N_thy,N_thx,N_thy),dtype=complex)
        theta = np.zeros((N_thx,N_thy,2),dtype=float)
        i_ft = np.zeros((N_thx,N_thy,N_thx,N_thy),dtype=int)
        i_fnu = np.zeros((N_thx,N_thy,N_thx,N_thy),dtype=int)
        
        #perform the computation
        time_start = time.time()
        print("Computing the theta transform ...")
        # - preparations
        #  - compute the theta array
        N_flat = N_thx**2*N_thy**2
        thx = np.linspace(thx_min,thx_max,num=N_thy,endpoint=True)
        thy = np.linspace(thy_min,thy_max,num=N_thy,endpoint=True)
        theta[:,:,0] = thx[:,na]
        theta[:,:,1] = thy[na,:]
        np.save(file_ar_theta,theta)
        #  - compute the corresponding Fourier conjugates
#        alpha = -2.*math.pi*self.nu_half/self.LightSpeed*theta
        rate = -2.*math.pi*self.nu_half/self.LightSpeed*(theta[:,:,0]*veffx+theta[:,:,1]*veffy)
        delay = math.pi/self.LightSpeed*Deff*(theta[:,:,0]**2+theta[:,:,1]**2)
        #  - prepare evaluation of FFT
        print("{0}s: Preparing FFT ...".format(time.time()-time_start))
        bar = progressbar.ProgressBar(maxval=N_thx, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_thx1 in xrange(N_thx):
            bar.update(i_thx1)
            for i_thy1 in xrange(N_thy):
                for i_thx2 in xrange(N_thx):
                    for i_thy2 in xrange(N_thy):
                        eft = -(rate[i_thx1,i_thy1]-rate[i_thx2,i_thy2])
                        efnu = -(delay[i_thx1,i_thy1]-delay[i_thx2,i_thy2])
                        idx = np.searchsorted(f_t, eft, side="left")
                        if idx == len(f_t):
                            idx -= 1
                        elif idx > 0 and (idx == len(f_t) or math.fabs(eft - f_t[idx-1]) < math.fabs(eft - f_t[idx])):
                            idx -= 1
                        i_ft[i_thx1,i_thy1,i_thx2,i_thy2] = idx
                        idx = np.searchsorted(f_nu, efnu, side="left")
                        if idx == len(f_nu):
                            idx -= 1
                        elif idx > 0 and (idx == len(f_nu) or math.fabs(efnu - f_nu[idx-1]) < math.fabs(efnu - f_nu[idx])):
                            idx -= 1
                        i_fnu[i_thx1,i_thy1,i_thx2,i_thy2] = idx
        bar.finish()
        #  - flatten arrays
        print("{0}s: Performing numpy operations ...".format(time.time()-time_start))
        i_ft = i_ft.flatten()
        i_fnu = i_fnu.flatten()
        dummy_ones = np.ones((N_thx,N_thy),dtype=float)
        theta1 = np.zeros((N_flat,2),dtype=float)
        theta2 = np.zeros((N_flat,2),dtype=float)
        theta1[:,0] = (theta[:,:,na,na,0]*dummy_ones[na,na,:,:]).flatten()
        theta1[:,1] = (theta[:,:,na,na,1]*dummy_ones[na,na,:,:]).flatten()
        theta2[:,0] = (theta[na,na,:,:,0]*dummy_ones[:,:,na,na]).flatten()
        theta2[:,1] = (theta[na,na,:,:,1]*dummy_ones[:,:,na,na]).flatten()
        falpha1 = -2.*math.pi*self.nu_half/self.LightSpeed*theta1
        falpha2 = -2.*math.pi*self.nu_half/self.LightSpeed*theta2
#        #  - already perform FFT over nu
#        hFFT_DynSpec = np.fft.fft(DynSpec,axis=3)
#        hFFT_DynSpec = np.fft.fftshift(hFFT_DynSpec,axes=3)
        #  - perform combination without a loop (requires a lot of memory but loop takes weeks!)
        comb_factor = np.exp(-1j*(falpha1[:,na,na,na,0]*p[na,:,na,:,0]+falpha1[:,na,na,na,1]*p[na,:,na,:,1])+1j*(falpha2[:,na,na,na,0]*p[na,na,:,:,0]+falpha2[:,na,na,na,1]*p[na,na,:,:,1]))
        print("{0}s: Performing combination ...".format(time.time()-time_start))
        bar = progressbar.ProgressBar(maxval=self.N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        #  - free memory
        rate = None
        delay = None
        theta = None
        theta1 = None
        theta2 = None
        falpha1 = None
        falpha2 = None
        dummy_ones = None
        thx = None
        thy = None
        #  - perform combination
        S_DynSpec = np.zeros((N_flat,self.N_t,self.N_nu),dtype=complex)
        for i_nu in xrange(self.N_nu):
            bar.update(i_nu)
            S_DynSpec[:,:,i_nu] = np.sum(DynSpec[na,:,:,:,i_nu]*comb_factor[:,:,:,:],axis=(1,2))
        bar.finish()
        comb_factor = None
#        c_DynSpec = DynSpec[na,:,:,:,:]*comb_factor[:,:,:,:,na]
#        S_DynSpec = np.sum(c_DynSpec,axis=(1,2))
        FFT_DynSpec = np.fft.fft2(S_DynSpec,axes=(1,2))
        FFT_DynSpec = np.fft.fftshift(FFT_DynSpec,axes=(1,2))
#        print("{0}s: Performing combination...".format(time.time()-time_start))
#        bar = progressbar.ProgressBar(maxval=N_thx**2*N_thy, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#        bar.start()
#        for i_thx1 in xrange(N_thx):
#            for i_thy1 in xrange(N_thy):
#                for i_thx2 in xrange(N_thx):
#                    bar.update(i_thx1*N_thy*N_thx+i_thy1*N_thx+i_thx2)
#                    for i_thy2 in xrange(N_thy):
#                        S_DynSpec = np.sum(hFFT_DynSpec*np.exp(-1j*(alpha[i_thx1,i_thy1,0]*p[:,na,:,na,0]+alpha[i_thx1,i_thy1,1]*p[:,na,:,na,1])+1j*(alpha[i_thx2,i_thy2,0]*p[na,:,:,na,0]+alpha[i_thx2,i_thy2,1]*p[na,:,:,na,1])),axis=(0,1))       
#                        FFT_DynSpec = np.fft.fft(S_DynSpec,axis=0)
#                        FFT_DynSpec = np.fft.fftshift(FFT_DynSpec,axes=0)
#                        theta_transform[i_thx1,i_thy1,i_thx2,i_thy2] = FFT_DynSpec[i_ft[i_thx1,i_thy1,i_thx2,i_thy2],i_fnu[i_thx1,i_thy1,i_thx2,i_thy2]]
#        bar.finish()
        print("{0}s: Performing sampling ...".format(time.time()-time_start))
        bar = progressbar.ProgressBar(maxval=N_flat, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        ftheta_transform = np.zeros((N_flat),dtype=float)
        for i_fl in xrange(N_flat):
            bar.update(i_fl)
            ftheta_transform[i_fl] = FFT_DynSpec[i_fl,i_ft[i_fl],i_fnu[i_fl]]
        bar.finish()
        theta_transform = np.reshape(ftheta_transform,theta_transform.shape)
        
        #save the results
        np.save(file_theta_transform,theta_transform)
        
        print("{0}s: Successfully computed the theta transform of the data.".format(time.time()-time_start))
        
#-----Fourier analysis-----    
        
    def compute_cleaned_dynamic_spectrum(self,path_computation):
        #load and check data
        DynSpec = np.load(os.path.join(self.path_data,"DynSpec.npy"))
        flags = np.load(os.path.join(self.path_data,"flags.npy"))
        
        #define files to compute
        file_cds = os.path.join(path_computation,"cds.npy")
#        file_doppler = os.path.join(path_computation,"doppler.npy")
#        file_delay = os.path.join(path_computation,"delay.npy")
        
        #create containers
        SecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=complex)
#        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
#        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=self.N_nu,endpoint=False)
        
        #perform the computation
        time_start = time.time()
        print("Computing cleaned dynamic spectrum ...")
        # - remove flagged data
        DynSpec[:,:,flags] = 0.
        # - compute the 2d Fourier transform
        SecSpec = np.fft.fft2(DynSpec,axes=(2,3))
        # - reduce noise
#        SecSpec -= np.median(SecSpec,axis=3)[:,:,:,na]
        # - delete x and y axis
#        center = 0.
#        center = SecSpec[:,:,0,0]
        SecSpec[:,:,0,:] = 0.
#        SecSpec[:,:,1,:] = 0.
#        SecSpec[:,:,2,:] = 0.
#        SecSpec[:,:,3,:] = 0.
#        SecSpec[:,:,-1,:] = 0.
#        SecSpec[:,:,-2,:] = 0.
#        SecSpec[:,:,-3,:] = 0.
        SecSpec[:,:,:,0] = 0.
#        SecSpec[:,:,:,1] = 0.
#        SecSpec[:,:,:,-1] = 0.
#        SecSpec[:,:,0,0] = center
#        SecSpec = np.abs(SecSpec)
        threshold = np.median(np.abs(SecSpec),axis=(2,3))
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                spec = SecSpec[i_p1,i_p2,:,:]
                spec[spec<threshold[i_p1,i_p2]] = 0.
                SecSpec[i_p1,i_p2,:,:] = spec
        # - backtronsform to dynamic spectrum
        cds = np.fft.ifft2(SecSpec,axes=(2,3))
        # - normalize columns
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                DySpe = np.abs(DynSpec[i_p1,i_p2,:,:])
                median = np.median(DySpe[np.nonzero(DySpe)])
                DySpe[DySpe==0.] = median
                orig_upper = np.sort(DySpe,axis=None)[int(0.9*self.N_t*self.N_nu)]
                orig_lower = np.sort(DySpe,axis=None)[int(0.1*self.N_t*self.N_nu)]
                orig_range = orig_upper - orig_lower
                for i_t in xrange(self.N_t):
                    upper = np.sort(np.abs(cds[i_p1,i_p2,i_t,:]),axis=None)[int(0.9*self.N_nu)]
                    lower = np.sort(np.abs(cds[i_p1,i_p2,i_t,:]),axis=None)[int(0.1*self.N_nu)]
                    trange = upper - lower
                    # - will not work for complex numbers
                    cds[i_p1,i_p2,i_t,:] = (cds[i_p1,i_p2,i_t,:]-lower)/trange*orig_range+orig_lower
        #save the results
        np.save(file_cds,cds)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the cleaned dynamic spectrum.".format(time_current))
        
    def plot_cleaned_dynamic_spectrum(self,dict_plot,figure,ax,tag):
        #load and check data
        # - cleaned dynamic spectrum
        path_results = os.path.join(self.path_computations,"cleaned_dynamic_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute cleaned_dynamic_spectrum' first!")
            return 0
        file_cds = os.path.join(path_results,"cds.npy")
        DynSpec = np.load(file_cds)
        
        #load specifications
        key_sepcs = "plot_cleaned_dynamic_spectrum{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":-1.,"vmax":float(np.max(np.real(DynSpec))),"t_min":float(self.t_min),"t_max":float(self.t_max),"nu_min":float(self.nu_min),"nu_max":float(self.nu_max),
                            "telescope1":0,"telescope2":0,"title":"cleaned dynamic spectrum","t_sampling":1,"nu_sampling":1} # $I$ [$W/m^2$]
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(self.t)-1
        for index_t in range(len(self.t)):
            if self.t[index_t]<dict_subplot["t_min"]:
                min_index_t = index_t
            elif self.t[index_t]>dict_subplot["t_max"]:
                max_index_t = index_t
                break
        t = self.t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(self.nu)-1
        for index_nu in range(len(self.nu)):
            if self.nu[index_nu]<dict_subplot["nu_min"]:
                min_index_nu = index_nu
            elif self.nu[index_nu]>dict_subplot["nu_max"]:
                max_index_nu = index_nu
                break
        nu = self.nu[min_index_nu:max_index_nu+2]
        DynSpectrum = np.real(DynSpec[tel1,tel2,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        # - downsampling
        t_sampling = dict_subplot["t_sampling"]
        nu_sampling = dict_subplot["nu_sampling"]
        DynSpectrum = block_reduce(DynSpectrum, block_size=(t_sampling,nu_sampling), func=np.mean)
        coordinates = np.array([t,t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=t[-1])
        t = coordinates[0,:]
        coordinates = np.array([nu,nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=nu[-1])
        nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_t = (t[1]-t[0])/2.
        offset_nu = (nu[1]-nu[0])/2.
        
        #draw the plot
        ax.set_xlim([dict_subplot["t_min"],dict_subplot["t_max"]])
        ax.set_ylim([dict_subplot["nu_min"],dict_subplot["nu_max"]])
        im = ax.pcolormesh(t-offset_t,nu-offset_nu,map(list, zip(*DynSpectrum)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"time $t$ [$s$]")
        ax.set_ylabel(r"frequency $\nu$ [Hz]")
        
        print("Successfully plotted cleaned dynamic spectrum.")
        
    def compute_SecSpec_from_cds(self,path_computation):
        #load and check data
        # - cleaned dynamic spectrum
        path_results = os.path.join(self.path_computations,"cleaned_dynamic_spectrum")
        if not os.path.exists(path_results):
            warnings.warn("You need to run 'compute cleaned_dynamic_spectrum' first!")
            return 0
        file_cds = os.path.join(path_results,"cds.npy")
        DynSpec = np.load(file_cds)
        
        #define files to compute
        file_power = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        
        #create containers
        SecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=self.N_nu,endpoint=False)
        
        #perform the computation
        time_start = time.time()
        print("Computing secondary spectrum from cleaned dynamic spectrum ...")
        # - compute the power spectrum
        A_DynSpec = np.fft.fft2(DynSpec,axes=(2,3))
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(2,3))
        SecSpec = self.delta_nu**2*self.delta_t**2*np.abs(A_DynSpec)**2
        
        #save the results
        np.save(file_power,SecSpec)
        np.save(file_doppler,f_t)
        np.save(file_delay,f_nu)
        
        time_current = time.time()-time_start
        print("{0}s: Successfully computed the secondary spectrum from cleaned dynamic spectrum.".format(time_current))
        
    def plot_SecSpec_from_cds(self,dict_plot,figure,ax,tag):
        #load and check data
        # - secondary spectrum from cleaned dynamic spectrum
        path_computation = os.path.join(self.path_computations,"SecSpec_from_cds")
        if not os.path.exists(path_computation):
            warnings.warn("You need to run 'compute SecSpec_from_cds' first!")
            return 0
        file_power = os.path.join(path_computation,"secondary_spectrum.npy")
        file_doppler = os.path.join(path_computation,"doppler.npy")
        file_delay = os.path.join(path_computation,"delay.npy")
        SecSpec = np.load(file_power)
        f_t = np.load(file_doppler)
        f_nu = np.load(file_delay)
        
        #load specifications
        key_sepcs = "plot_SecSpec_from_cds{0}".format(tag)
        if not key_sepcs in dict_plot:
            dict_subplot = {"cmap":'viridis',"vmin":0.,"vmax":float(np.max(np.real(SecSpec))),"f_t_min":float(-1./2./self.delta_t),"f_t_max":float(1./2./self.delta_t),"f_nu_min":0.,"f_nu_max":float(1./2./self.delta_nu),
                            "telescope1":0,"telescope2":0,"title":"cleaned secondary spectrum (log10)","f_t_sampling":1,"f_nu_sampling":1} # $P$ [$(W/m^2)^2$]
            dict_plot.update({key_sepcs:dict_subplot})
        else:
            dict_subplot = dict_plot[key_sepcs]
        
        #Preprocess data
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - read in specifications
        tel1 = dict_subplot["telescope1"]
        tel2 = dict_subplot["telescope2"]
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<dict_subplot["f_t_min"]:
                min_index_t = index_t
            elif f_t[index_t]>dict_subplot["f_t_max"]:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<dict_subplot["f_nu_min"]:
                min_index_nu = index_nu
            elif f_nu[index_nu]>dict_subplot["f_nu_max"]:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+1]
        SecSpec = SecSpec[tel1,tel2,min_index_t:max_index_t+1,min_index_nu:max_index_nu+1]
        # - downsampling
        f_t_sampling = dict_subplot["f_t_sampling"]
        f_nu_sampling = dict_subplot["f_nu_sampling"]
        SecSpec = block_reduce(SecSpec, block_size=(f_t_sampling,f_nu_sampling), func=np.mean)
        coordinates = np.array([f_t,f_t])
        coordinates = block_reduce(coordinates, block_size=(1,f_t_sampling), func=np.mean, cval=f_t[-1])
        f_t = coordinates[0,:]
        coordinates = np.array([f_nu,f_nu])
        coordinates = block_reduce(coordinates, block_size=(1,f_nu_sampling), func=np.mean, cval=f_nu[-1])
        f_nu = coordinates[0,:]
        # - compute offsets to center pccolormesh
        offset_f_t = (f_t[1]-f_t[0])/2.
        offset_f_nu = (f_nu[1]-f_nu[0])/2.
        # - safely remove zeros if there are any
        min_nonzero = np.min(SecSpec[np.nonzero(SecSpec)])
        SecSpec[SecSpec == 0] = min_nonzero
        # - apply logarithmic scale
        SecSpec_log10 = np.log10(SecSpec)
        
        #draw the plot
        ax.set_xlim([dict_subplot["f_t_min"],dict_subplot["f_t_max"]])
        ax.set_ylim([dict_subplot["f_nu_min"],dict_subplot["f_nu_max"]])
        im = ax.pcolormesh(f_t-offset_f_t,f_nu-offset_f_nu,map(list, zip(*SecSpec_log10)),cmap=dict_subplot["cmap"],vmin=dict_subplot["vmin"],vmax=dict_subplot["vmax"])
        figure.colorbar(im, ax=ax)
        ax.set_title(dict_subplot["title"])
        ax.set_xlabel(r"Doppler $f_t$ [Hz]") #(r"$f_x$ [au]") #
        ax.set_ylabel(r"Delay $f_{\nu}$ [$s$]") #(r"$f_y$ [au]") #
        
        print("Successfully plotted secondary spectrum.")
        
#-----------------------OBSOLETE FUNCTIONS------------------------------------------------------------------------------------------------------------------------------------
        