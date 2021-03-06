import numpy as np
from numpy import newaxis as na
import math

from scinter.computation import computation as scinter_computation

class defined_analyses:
    
    def dynamic_spectrum(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_DynSpec = self._add_specification("source_DynSpec",[None,None,None],dict_subplot)
    
        #load and check data
        if source_DynSpec[0]==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            source = scinter_computation(self.dict_paths,source_DynSpec[0])
            DynSpec, = source.load_result([source_DynSpec[2]])
        if source_DynSpec[0]=="FFT_phi_evolution":
            source_nus = self._add_specification("source_nus",["FFT_phi_evolution",None,"nus"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_nus[0])
            nus, = source.load_result([source_nus[2]])
            source_midnus = scinter_computation(self.dict_paths,"FFT_phi_evolution")
            midnus, = source_midnus.load_result(["midnus"])
            i_slice = self._add_specification("i_slice",0,dict_subplot)
            nu = nus[i_slice,:]
            t = self.t
        elif source_DynSpec[0]=="DS_downsampled": #[DS_downsampled, null, new_DS]
            source_t = self._add_specification("source_t",["DS_downsampled",None,"new_t"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_t[0])
            t, = source.load_result([source_t[2]])
            source_nu = self._add_specification("source_nu",["DS_downsampled",None,"new_nu"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_nu[0])
            nu, = source.load_result([source_nu[2]])
        elif source_DynSpec[0]=="Gini_deconvolution": #[Gini_deconvolution, null, DS_deconv]
            source_t = self._add_specification("source_t",["Gini_deconvolution",None,"t"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_t[0])
            t, = source.load_result([source_t[2]])
            source_nu = self._add_specification("source_nu",["Gini_deconvolution",None,"nu"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_nu[0])
            nu, = source.load_result([source_nu[2]])
        else:
            nu = self.nu
            t = self.t
            
        #load specifications
        t_scale = self._add_specification("t_scale",60.,dict_subplot)
        nu_scale = self._add_specification("nu_scale",1.0e+06,dict_subplot)
        t_lcut = self._add_specification("t_lcut",float(self.t[0]/t_scale),dict_subplot)
        t_ucut = self._add_specification("t_ucut",float(self.t[-1]/t_scale),dict_subplot)
        nu_lcut = self._add_specification("nu_lcut",float(self.nu[0]/nu_scale),dict_subplot)
        nu_ucut = self._add_specification("nu_ucut",float(self.nu[-1]/nu_scale),dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Dynamic Spectrum",dict_subplot)
        self._add_specification("xlabel",r"$t$ [min]",dict_subplot)
        self._add_specification("ylabel",r"$\nu$ [MHz]",dict_subplot)
        
        #refine data
        # - apply scale
        t = t/t_scale
        nu = nu/nu_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(t)-1
        for index_t in range(len(t)):
            if t[index_t]<t_lcut:
                min_index_t = index_t
            elif t[index_t]>t_ucut:
                max_index_t = index_t
                break
        t = t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(nu)-1
        for index_nu in range(len(nu)):
            if nu[index_nu]<nu_lcut:
                min_index_nu = index_nu
            elif nu[index_nu]>nu_ucut:
                max_index_nu = index_nu
                break
        nu = nu[min_index_nu:max_index_nu+2]
        if source_DynSpec[0]=="FFT_phi_evolution":
            print("Plotting dynamic spectrum for slice centered at {0}...".format(midnus[i_slice]))
            DynSpec = DynSpec[i_slice,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        elif source_DynSpec[0]=="Gini_deconvolution":
            DynSpec = DynSpec[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        elif source_DynSpec[0] in ["deconvolve_SecSpec","SecSpec_fit_stripes"]:
            DynSpec = np.abs(DynSpec[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        else:
            tel1 = self._add_specification("telescope1",0,dict_subplot)
            tel2 = self._add_specification("telescope2",0,dict_subplot)
            DynSpec = np.real(DynSpec[tel1,tel2,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        # - define data
        data = {'x':t,'y':nu,'f_xy':DynSpec}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def secondary_spectrum(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_SecSpec = self._add_specification("source_SecSpec",["secondary_spectrum",None,"secondary_spectrum"],dict_subplot)
        source_doppler = self._add_specification("source_doppler",["secondary_spectrum",None,"doppler"],dict_subplot)
        source_delay = self._add_specification("source_delay",["secondary_spectrum",None,"delay"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_SecSpec[0])
        SecSpec, = source.load_result([source_SecSpec[2]])
        source = scinter_computation(self.dict_paths,source_doppler[0])
        f_t, = source.load_result([source_doppler[2]])
        source = scinter_computation(self.dict_paths,source_delay[0])
        f_nu, = source.load_result([source_delay[2]])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        doppler_lcut = self._add_specification("doppler_lcut",float(f_t[0]/doppler_scale),dict_subplot)
        doppler_ucut = self._add_specification("doppler_ucut",float(f_t[-1]/doppler_scale),dict_subplot)
        delay_lcut = self._add_specification("delay_lcut",float(f_nu[0]/delay_scale),dict_subplot)
        delay_ucut = self._add_specification("delay_ucut",float(f_nu[-1]/delay_scale),dict_subplot)
        doppler_weight_reference = self._add_specification("doppler_weight_reference",None,dict_subplot)
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Secondary Spectrum",dict_subplot)
        self._add_specification("xlabel",r"$f_D$ [mHz]",dict_subplot)
        self._add_specification("ylabel",r"$\tau$ [$\mu$s]",dict_subplot)
        
        #refine data
        # - apply scale
        f_t = f_t/doppler_scale
        f_nu = f_nu/delay_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<doppler_lcut:
                min_index_t = index_t
            elif f_t[index_t]>doppler_ucut:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<delay_lcut:
                min_index_nu = index_nu
            elif f_nu[index_nu]>delay_ucut:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+2]
        # - load parameters and spectrum specific to chosen data source
        if source_SecSpec[0] in ["secondary_spectrum","nut_SecSpec","iterative_SecSpec","SecSpec_divPulseVar","SecSpec_scale_cut"]:
            tel1 = self._add_specification("telescope1",0,dict_subplot)
            tel2 = self._add_specification("telescope2",0,dict_subplot)
            SecSpec = SecSpec[tel1,tel2,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        elif source_SecSpec[0]=="extracted_SSpec":
            SecSpec = SecSpec[0,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        elif source_SecSpec[0] in ["clean_SecSpec","LombScargle_SecSpec","deconvolve_SecSpec","SecSpec_fit_stripes","SecSpec_clean_deconvolution"]:
            # print(SecSpec.shape,np.std(np.abs(SecSpec)))
            # print(SecSpec)
            # print(np.isnan(np.std(SecSpec)))
            SecSpec = np.abs(SecSpec[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])
        elif source_SecSpec[0] in ["FFT_phi_evolution"]:
            i_slice = self._add_specification("i_slice",0,dict_subplot)
            SecSpec = SecSpec[i_slice,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        else:
            SecSpec = SecSpec[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        if not doppler_weight_reference==None:
            # - weight secondary spectrum by density of corresponding 1D images
            SecSpec = SecSpec*np.abs(f_t[:,na])/doppler_weight_reference
        if flag_logarithmic:
            # - safely remove zeros if there are any
            min_nonzero = np.min(SecSpec[np.nonzero(SecSpec)])
            SecSpec[SecSpec == 0] = min_nonzero
            # - apply logarithmic scale
            SecSpec = np.log10(SecSpec)
        # - define data
        data = {'x':f_t,'y':f_nu,'f_xy':SecSpec}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def secondary_phase(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_phase = self._add_specification("source_SecSpec",["secondary_spectrum",None,"Fourier_phase"],dict_subplot)
        source_doppler = self._add_specification("source_doppler",["secondary_spectrum",None,"doppler"],dict_subplot)
        source_delay = self._add_specification("source_delay",["secondary_spectrum",None,"delay"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_phase[0])
        phase, = source.load_result([source_phase[2]])
        source = scinter_computation(self.dict_paths,source_doppler[0])
        f_t, = source.load_result([source_doppler[2]])
        source = scinter_computation(self.dict_paths,source_delay[0])
        f_nu, = source.load_result([source_delay[2]])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        doppler_lcut = self._add_specification("doppler_lcut",float(f_t[0]/doppler_scale),dict_subplot)
        doppler_ucut = self._add_specification("doppler_ucut",float(f_t[-1]/doppler_scale),dict_subplot)
        delay_lcut = self._add_specification("delay_lcut",float(f_nu[0]/delay_scale),dict_subplot)
        delay_ucut = self._add_specification("delay_ucut",float(f_nu[-1]/delay_scale),dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Secondary Spectrum Phase",dict_subplot)
        self._add_specification("xlabel",r"$f_D$ [mHz]",dict_subplot)
        self._add_specification("ylabel",r"$\tau$ [$\mu$s]",dict_subplot)
        self._add_specification("cmap",'bwr',dict_subplot)
        
        #refine data
        # - apply scale
        f_t = f_t/doppler_scale
        f_nu = f_nu/delay_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<doppler_lcut:
                min_index_t = index_t
            elif f_t[index_t]>doppler_ucut:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<delay_lcut:
                min_index_nu = index_nu
            elif f_nu[index_nu]>delay_ucut:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+2]
        # - load parameters and spectrum specific to chosen data source
        if source_phase[0] in ["secondary_spectrum"]:
            tel1 = self._add_specification("telescope1",0,dict_subplot)
            tel2 = self._add_specification("telescope2",0,dict_subplot)
            phase = phase[tel1,tel2,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        elif source_phase[0] in ["FFT_phi_evolution"]:
            index_nu = self._add_specification("index_nu",0,dict_subplot)
            index_nu_cal = self._add_specification("index_nu_cal",None,dict_subplot)
            if index_nu_cal==None:
                phase = phase[index_nu,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
            else:
                phase = np.angle(np.exp(1.0j*phase[index_nu,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2])*np.exp(-1.0j*phase[0,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]))
            # - uncomment to print available frequencies
            #source = scinter_computation(self.dict_paths,source_phase[0])
            #midnus, = source.load_result(["midnus"])
            #print(midnus/1.0e+06)
        else:
            phase = phase[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        # - define data
        data = {'x':f_t,'y':f_nu,'f_xy':phase}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def secondary_measure(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_measure = self._add_specification("source_measure",["secondary_spectrum",None,"secondary_spectrum"],dict_subplot)
        source_doppler = self._add_specification("source_doppler",["secondary_spectrum",None,"doppler"],dict_subplot)
        source_delay = self._add_specification("source_delay",["secondary_spectrum",None,"delay"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_measure[0])
        measure, = source.load_result([source_measure[2]])
        source = scinter_computation(self.dict_paths,source_doppler[0])
        f_t, = source.load_result([source_doppler[2]])
        source = scinter_computation(self.dict_paths,source_delay[0])
        f_nu, = source.load_result([source_delay[2]])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        doppler_lcut = self._add_specification("doppler_lcut",float(f_t[0]/doppler_scale),dict_subplot)
        doppler_ucut = self._add_specification("doppler_ucut",float(f_t[-1]/doppler_scale),dict_subplot)
        delay_lcut = self._add_specification("delay_lcut",float(f_nu[0]/delay_scale),dict_subplot)
        delay_ucut = self._add_specification("delay_ucut",float(f_nu[-1]/delay_scale),dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Secondary Measure",dict_subplot)
        self._add_specification("xlabel",r"$f_D$ [mHz]",dict_subplot)
        self._add_specification("ylabel",r"$\tau$ [$\mu$s]",dict_subplot)
        
        #refine data
        # - apply scale
        f_t = f_t/doppler_scale
        f_nu = f_nu/delay_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(f_t)-1
        for index_t in range(len(f_t)):
            if f_t[index_t]<doppler_lcut:
                min_index_t = index_t
            elif f_t[index_t]>doppler_ucut:
                max_index_t = index_t
                break
        f_t = f_t[min_index_t:max_index_t+2]
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<delay_lcut:
                min_index_nu = index_nu
            elif f_nu[index_nu]>delay_ucut:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+2]
        # - load parameters and spectrum specific to chosen data source
        measure = measure[min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        # - define data
        data = {'x':f_t,'y':f_nu,'f_xy':measure}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def FFT_phi_evolution(self,dict_subplot,list_category,list_data):
        #load and check data
        source = scinter_computation(self.dict_paths,"FFT_phi_evolution")
        phase,cphase,midnus,f_t,f_nu = source.load_result(["phase","cphase","midnus","doppler","delay"])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        nu_scale = self._add_specification("nu_scale",1.0e+06,dict_subplot)
        f_D = self._add_specification("f_D",0.,dict_subplot)
        tau = self._add_specification("tau",0.,dict_subplot)
        # - specifications for plotting
        self._add_specification("title","phase evolution at $f_D = {0}$ mHz and $\\tau = {1} \mu$s".format(f_D,tau),dict_subplot)
        self._add_specification("xlabel",r"$\nu$ [MHz]",dict_subplot)
        self._add_specification("ylabel","$\phi$ [rad]",dict_subplot)
        self._add_specification("curve_label0","raw",dict_subplot)
        self._add_specification("curve_label1","corrected",dict_subplot)
        
        #refine data
        # - apply scale
        f_t = f_t/doppler_scale
        f_nu = f_nu/delay_scale
        midnus = midnus/nu_scale
        # - draw pixel
        i_t = (np.abs(f_t-f_D)).argmin()
        i_nu = (np.abs(f_nu-tau)).argmin()
        print(f_t[i_t],f_nu[i_nu])
        # - define data
        data = {'N':2,'x0':midnus,'y0':phase[:,i_t,i_nu],'x1':midnus,'y1':cphase[:,i_t,i_nu]}
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        
    def pulse_variation(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_PV = self._add_specification("source_PV",["pulse_variation",None,"pulse_variation"],dict_subplot)
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_PV[0])
        pulse_variation, = source.load_result([source_PV[2]])
        
        #load specifications
        tel1 = self._add_specification("telescope1",0,dict_subplot)
        tel2 = self._add_specification("telescope2",0,dict_subplot)
        t_scale = self._add_specification("t_scale",60.,dict_subplot)
        t_lcut = self._add_specification("t_lcut",float(self.t[0]/t_scale),dict_subplot)
        t_ucut = self._add_specification("t_ucut",float(self.t[-1]/t_scale),dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Pulse Variation",dict_subplot)
        self._add_specification("xlabel",r"$t$ [min]",dict_subplot)
        self._add_specification("ylabel",r"strength [au]",dict_subplot)
        
        #refine data
        # - apply scale
        t = self.t/t_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(t)-1
        for index_t in range(len(t)):
            if t[index_t]<t_lcut:
                min_index_t = index_t
            elif t[index_t]>t_ucut:
                max_index_t = index_t
                break
        t = t[min_index_t:max_index_t+2]
        pulse_variation = pulse_variation[tel1,tel2,min_index_t:max_index_t+2]
        # - define data
        data = {'N':1,'x0':t,'y0':pulse_variation}
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        
    def halfSecSpec(self,dict_subplot,list_category,list_data):
        #load specifications
        source_halfSecSpec = self._add_specification("source_halfSecSpec",["halfSecSpec",None,"halfSecSpec"],dict_subplot)
        source_delay = self._add_specification("source_delay",["secondary_spectrum",None,"delay"],dict_subplot)
    
        #load and check data
        source = scinter_computation(self.dict_paths,source_halfSecSpec[0])
        halfSecSpec, = source.load_result([source_halfSecSpec[2]])
        source = scinter_computation(self.dict_paths,source_delay[0])
        f_nu, = source.load_result([source_delay[2]])
        
        # - sparate 2pi from the Fourier frequency
        f_nu /= 2.*math.pi
        
        #load specifications
        tel1 = self._add_specification("telescope1",0,dict_subplot)
        tel2 = self._add_specification("telescope2",0,dict_subplot)
        t_scale = self._add_specification("t_scale",60.,dict_subplot)
        t_lcut = self._add_specification("t_lcut",float(self.t[0]/t_scale),dict_subplot)
        t_ucut = self._add_specification("t_ucut",float(self.t[-1]/t_scale),dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        delay_lcut = self._add_specification("delay_lcut",float(f_nu[0]/delay_scale),dict_subplot)
        delay_ucut = self._add_specification("delay_ucut",float(f_nu[-1]/delay_scale),dict_subplot)
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"half Secondary Spectrum",dict_subplot)
        self._add_specification("xlabel",r"$t$ [min]",dict_subplot)
        self._add_specification("ylabel",r"$\tau$ [$\mu$s]",dict_subplot)
        
        #refine data
        # - apply scale
        t = self.t/t_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(t)-1
        for index_t in range(len(t)):
            if t[index_t]<t_lcut:
                min_index_t = index_t
            elif t[index_t]>t_ucut:
                max_index_t = index_t
                break
        t = t[min_index_t:max_index_t+2]
        # - apply scale
        f_nu = f_nu/delay_scale
        # - cut off out of range data
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<delay_lcut:
                min_index_nu = index_nu
            elif f_nu[index_nu]>delay_ucut:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+2]
        halfSecSpec = halfSecSpec[tel1,tel2,min_index_t:max_index_t+2,min_index_nu:max_index_nu+2]
        if flag_logarithmic:
            # - safely remove zeros if there are any
            min_nonzero = np.min(halfSecSpec[np.nonzero(halfSecSpec)])
            halfSecSpec[halfSecSpec == 0] = min_nonzero
            # - apply logarithmic scale
            halfSecSpec = np.log10(halfSecSpec)
        # - define data
        data = {'x':t,'y':f_nu,'f_xy':halfSecSpec}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
################### thth diagram ###################
        
    def thth(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_thth = self._add_specification("source_thth",["thth_real",None,"thth"],dict_subplot)
        source_thetas = self._add_specification("source_thetas",["thth_real",None,"thetas"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_thth[0])
        thth, = source.load_result([source_thth[2]])
        source = scinter_computation(self.dict_paths,source_thetas[0])
        thetas, = source.load_result([source_thetas[2]])
        
        #load specifications
        veff = self._add_specification("veff",305000.,dict_subplot) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas,dict_subplot) #radians
        flag_logarithmic = self._add_specification("flag_logarithmic",False,dict_subplot)
        flag_mask = self._add_specification("flag_mask",False,dict_subplot)
        flag_plot_lines = self._add_specification("flag_plot_lines",False,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"$\theta$-$\theta$ diagram",dict_subplot)
        self._add_specification("xlabel",r"$\theta_1$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"$\theta_2$ [mas]",dict_subplot)
        # - inform about parameters
        print("Current nu_0={0}".format(self.nu_half))
        
        #refine data
        # - apply scale and velocity translation
        theta1 = (-self.LightSpeed/self.nu_half/veff*thetas)/theta_scale
        theta2 = np.copy(theta1)
        # - apply mask
        if flag_mask:
            source_mask = self._add_specification("source_mask",["thth_eigenvector",None,"weights"],dict_subplot)
            source = scinter_computation(self.dict_paths,source_mask[0])
            mask, = source.load_result([source_mask[2]])
            thth *= mask
        # - convert to log10 scale
        if flag_logarithmic:
            # - flip negative values
            thth = np.abs(thth)
            # - safely remove zeros if there are any
            min_nonzero = np.min(thth[np.nonzero(thth)])
            thth[thth == 0] = min_nonzero
            # - apply logarithmic scale
            thth = np.log10(thth)
        # - define data
        data = {'x':theta1,'y':theta2,'f_xy':thth}
        # - overplot lines if requested
        if flag_plot_lines:
            source_lines = self._add_specification("source_lines","import_lines",dict_subplot)
            source = scinter_computation(self.dict_paths,source_lines)
            th1,th2 = source.load_result(["th1","th2"])
            th1 *= -self.LightSpeed/self.nu_half/veff/theta_scale
            th2 *= -self.LightSpeed/self.nu_half/veff/theta_scale
            N_lines = len(th1)
            line_data = {'N':N_lines}
            for i_line in range(N_lines):
                line_data.update({'x{0}'.format(i_line):th1[i_line,:],'y{0}'.format(i_line):th2[i_line,:]})
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        if flag_plot_lines:
            list_category.append("curve")
            list_data.append(line_data)
        
    def FFT_thth(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_FFT_thth = self._add_specification("source_FFT_thth",["FFT_thth",None,"FFT_thth"],dict_subplot)
        source_f_theta = self._add_specification("source_f_theta",["FFT_thth",None,"f_theta"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_FFT_thth[0])
        FFT_thth, = source.load_result([source_FFT_thth[2]])
        source = scinter_computation(self.dict_paths,source_f_theta[0])
        f_theta, = source.load_result([source_f_theta[2]])
        
        #load specifications
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"FFT of $\theta$-$\theta$ diagram",dict_subplot)
        self._add_specification("xlabel",r"FFT($\theta_1$)",dict_subplot)
        self._add_specification("ylabel",r"FFT($\theta_2$)",dict_subplot)
        # - inform about parameters
        print("Current nu_0={0}".format(self.nu_half))
        
        #refine data
        if flag_logarithmic:
            # - safely remove zeros if there are any
            min_nonzero = np.min(FFT_thth[np.nonzero(FFT_thth)])
            FFT_thth[FFT_thth == 0] = min_nonzero
            # - apply logarithmic scale
            FFT_thth = np.log10(FFT_thth)
        # - define data
        data = {'x':f_theta,'y':np.copy(f_theta),'f_xy':FFT_thth}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def linSS(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_linSS = self._add_specification("source_linSS",["linSS_real",None,"linSS"],dict_subplot)
        source_doppler = self._add_specification("source_doppler",["linSS_real",None,"doppler"],dict_subplot)
        source_ddd = self._add_specification("source_ddd",["linSS_real",None,"ddd"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_linSS[0])
        linSS, = source.load_result([source_linSS[2]])
        source = scinter_computation(self.dict_paths,source_doppler[0])
        doppler, = source.load_result([source_doppler[2]])
        source = scinter_computation(self.dict_paths,source_ddd[0])
        ddd, = source.load_result([source_ddd[2]])
        
        #load specifications
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        delay_scale = self._add_specification("delay_scale",1.0e-06,dict_subplot)
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"Linearized Secondary Spectrum",dict_subplot)
        self._add_specification("xlabel",r"$f_D$ [mHz]",dict_subplot)
        self._add_specification("ylabel",r"$\tau/f_D$ [$\mu$s/mHz]",dict_subplot)
        # - inform about parameters
        print("Current nu_0={0}".format(self.nu_half))
        
        #refine data
        # - apply scale and velocity translation
        doppler = doppler/doppler_scale
        ddd = ddd*doppler_scale/delay_scale
        if flag_logarithmic:
            # - flip negative values
            linSS = np.abs(linSS)
            # - safely remove zeros if there are any
            min_nonzero = np.min(linSS[np.nonzero(linSS)])
            linSS[linSS == 0] = min_nonzero
            # - apply logarithmic scale
            linSS = np.log10(linSS)
        # - define data
        data = {'x':doppler,'y':ddd,'f_xy':linSS}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def FFT_linSS(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_FFT_linSS = self._add_specification("source_FFT_linSS",["FFT_linSS",None,"FFT_linSS"],dict_subplot)
        source_f_dop = self._add_specification("source_f_dop",["FFT_linSS",None,"f_dop"],dict_subplot)
        source_f_ddd = self._add_specification("source_f_ddd",["FFT_linSS",None,"f_ddd"],dict_subplot)
        
        #TODO implement usage of archive folders
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_FFT_linSS[0])
        FFT_linSS, = source.load_result([source_FFT_linSS[2]])
        source = scinter_computation(self.dict_paths,source_f_dop[0])
        f_dop, = source.load_result([source_f_dop[2]])
        source = scinter_computation(self.dict_paths,source_f_ddd[0])
        f_ddd, = source.load_result([source_f_ddd[2]])
        
        #load specifications
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"FFT of Linearized Secondary Spectrum",dict_subplot)
        self._add_specification("xlabel",r"FFT($f_D$) [s]",dict_subplot)
        self._add_specification("ylabel",r"FFT($\tau/f_D$) [1/s$^2$]",dict_subplot)
        # - inform about parameters
        print("Current nu_0={0}".format(self.nu_half))
        
        #refine data
        if flag_logarithmic:
            # - safely remove zeros if there are any
            min_nonzero = np.min(FFT_linSS[np.nonzero(FFT_linSS)])
            FFT_linSS[FFT_linSS == 0] = min_nonzero
            # - apply logarithmic scale
            FFT_linSS = np.log10(FFT_linSS)
        # - define data
        data = {'x':f_dop,'y':f_ddd,'f_xy':FFT_linSS}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def curvature_search(self,dict_subplot,list_category,list_data):
        #load and check data
        source = scinter_computation(self.dict_paths,"FFT_linSS_sum")
        eta,eta_sum,xdata,fit_result,fit_error = source.load_result(["curv","curv_sum","xdata","fit_result","fit_error"])
        
        #load specifications
        # - specifications for plotting
        self._add_specification("title","curvature search at $\\nu_0 = {0}$ Hz".format(self.nu_half),dict_subplot)
        self._add_specification("xlabel",r"$\eta$ [s$^3$]",dict_subplot)
        self._add_specification("ylabel","weight [au]",dict_subplot)
        self._add_specification("curve_label0","weight",dict_subplot)
        self._add_specification("curve_label1","fit $\eta=${0}({1}) s$^3$".format(fit_result[0],fit_error[0]),dict_subplot)
        
        #refine data
        # - define parabolic fit
        ydata = fit_result[2]*(xdata-fit_result[0])**2+fit_result[1]
        # - define data
        data = {'N':2,'x0':eta,'y0':eta_sum,'x1':xdata,'y1':ydata}
        
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        
    def thth_eigenvector(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_eigenvector = self._add_specification("source_eigenvector",["thth_eigenvector",None,"eigenvector"],dict_subplot)
        source_thetas = self._add_specification("source_thetas",["thth_real",None,"thetas"],dict_subplot)
    
        #load and check data
        source = scinter_computation(self.dict_paths,source_eigenvector[0])
        eigenvector, = source.load_result([source_eigenvector[2]])
        source = scinter_computation(self.dict_paths,source_thetas[0])
        thetas, = source.load_result([source_thetas[2]])
        
        #load specifications
        veff = self._add_specification("veff",305000.,dict_subplot) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas,dict_subplot) #radians
        flag_logarithmic = self._add_specification("flag_logarithmic",False,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"eigenvector $\mu(\theta)$ [log10]",dict_subplot)
        self._add_specification("xlabel",r"$\theta$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"$\mu$",dict_subplot)
        
        #refine data
        # - apply scale and velocity translation
        thetas = (-self.LightSpeed/self.nu_half/veff*thetas)/theta_scale
        if flag_logarithmic:
            # - flip negative values
            eigenvector = np.abs(eigenvector)
            # - safely remove zeros if there are any
            min_nonzero = np.min(eigenvector[np.nonzero(eigenvector)])
            eigenvector[eigenvector == 0] = min_nonzero
            # - apply logarithmic scale
            eigenvector = np.log10(eigenvector)
        # - define data
        data = {'N':1,'x0':thetas,'y0':eigenvector}
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        
    def discrete_screen(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_screen = self._add_specification("source_screen",["render_screen",None,"screen"],dict_subplot)
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_screen[0])
        screen, = source.load_result([source_screen[2]])
        
        #load specifications
        veff = self._add_specification("veff",305000.,dict_subplot) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas,dict_subplot) #radians
        flag_logarithmic = self._add_specification("flag_logarithmic",False,dict_subplot)
        flag_show_veff = self._add_specification("flag_show_veff",False,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"screen",dict_subplot)
        self._add_specification("xlabel",r"$\theta_\parallel$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"$\theta_\perp$ [mas]",dict_subplot)
        
        #refine data
        theta_par = screen[:,0]
        theta_perp = screen[:,1]
        mu = screen[:,2]
        # - apply scale and velocity translation
        theta_par = (-self.LightSpeed/self.nu_half/veff*theta_par)/theta_scale
        theta_perp = (-self.LightSpeed/self.nu_half/veff*theta_perp)/theta_scale
        if flag_logarithmic:
            # - flip negative values
            mu = np.abs(mu)
            # - safely remove zeros if there are any
            min_nonzero = np.min(mu[np.nonzero(mu)])
            mu[mu == 0] = min_nonzero
            # - apply logarithmic scale
            mu = np.log10(mu)
        if flag_show_veff:
            beta = self._add_specification("beta",-28.6,dict_subplot) #degrees
            if -90.<beta<90.:
                x_veff = np.array([0.,np.nanmax(theta_par)])
                y_veff = np.array([0.,np.nanmax(theta_par)*np.tan(np.deg2rad(beta))])
            else:
                x_veff = np.array([0.,np.nanmin(theta_par)])
                y_veff = np.array([0.,np.nanmin(theta_par)*np.tan(np.deg2rad(beta))])
        # - define data
        data = {'x':theta_par,'y':theta_perp,'f_xy':mu}
        if flag_show_veff:
            data_veff = {'N':1,'x0':x_veff,'y0':y_veff}
        
        #define list of data and the categories of plots
        list_category.append("scatter")
        list_data.append(data)
        if flag_show_veff:
            list_category.append("curve")
            list_data.append(data_veff)
            
    def thfD(self,dict_subplot,list_category,list_data):
        #check which data to use
        source_thfD = self._add_specification("source_thfD",["thfD_diagram",None,"thfD"],dict_subplot)
        source_thetas = self._add_specification("source_thetas",["thfD_diagram",None,"thetas"],dict_subplot)
        source_fDs = self._add_specification("source_fDs",["thfD_diagram",None,"fDs"],dict_subplot)
        
        #load and check data
        source = scinter_computation(self.dict_paths,source_thfD[0])
        thfD, = source.load_result([source_thfD[2]])
        source = scinter_computation(self.dict_paths,source_thetas[0])
        thetas, = source.load_result([source_thetas[2]])
        source = scinter_computation(self.dict_paths,source_fDs[0])
        fDs, = source.load_result([source_fDs[2]])
        
        #load specifications
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        theta_scale = self._add_specification("theta_scale",self.mas,dict_subplot) #radians
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"$\theta$-$f_\mathrm{D}$ diagram",dict_subplot)
        self._add_specification("xlabel",r"$\theta$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"$f_\mathrm{D}$ [mHz]",dict_subplot)
        
        #refine data
        # - apply scale
        thetas /= theta_scale
        fDs /= doppler_scale*2.*np.pi
        # - convert to log10 scale
        if flag_logarithmic:
            # - flip negative values
            thfD = np.abs(thfD)
            # - safely remove zeros if there are any
            min_nonzero = np.min(thfD[np.nonzero(thfD)])
            thfD[thfD == 0] = min_nonzero
            # - apply logarithmic scale
            thfD = np.log10(thfD)
        # - define data
        data = {'x':thetas,'y':fDs,'f_xy':thfD}
        
        #define list of data and the categories of plots
        list_category.append("colormesh")
        list_data.append(data)
        
    def thfD_line(self,dict_subplot,list_category,list_data):
        #load and check data
        thetas = self._load_add_specification("source_thetas",["thfD_diagram",None,"thetas"],dict_subplot)
        profile = self._load_add_specification("source_profile",["thfD_line",None,"profile"],dict_subplot)
        line = self._load_add_specification("source_line",["thfD_line",None,"line"],dict_subplot)
        
        #load specifications
        flag_logarithmic = self._add_specification("flag_logarithmic",True,dict_subplot)
        theta_scale = self._add_specification("theta_scale",self.mas,dict_subplot) #radians
        # - specifications for plotting
        self._add_specification("title",r"Image Distribution",dict_subplot)
        self._add_specification("xlabel",r"$\theta$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"power [au]",dict_subplot)
        
        #refine data
        # - apply scale
        thetas /= theta_scale
        # - convert to log10 scale
        if flag_logarithmic:
            line = self._log10_safe(line)
        # - define data
        data = {'N':1,'x0':thetas,'y0':line}
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        
    def thfD_profile(self,dict_subplot,list_category,list_data):
        #load and check data
        fDs = self._load_add_specification("source_thetas",["thfD_diagram",None,"fDs"],dict_subplot)
        profile = self._load_add_specification("source_profile",["thfD_line",None,"profile"],dict_subplot)
        line = self._load_add_specification("source_line",["thfD_line",None,"line"],dict_subplot)
        
        #load specifications
        flag_logarithmic = self._add_specification("flag_logarithmic",False,dict_subplot)
        doppler_scale = self._add_specification("doppler_scale",1.0e-03,dict_subplot)
        # - specifications for plotting
        self._add_specification("title",r"doppler profile",dict_subplot)
        self._add_specification("xlabel",r"$f_\mathrm{D}$ [mas]",dict_subplot)
        self._add_specification("ylabel",r"power [au]",dict_subplot)
        
        #refine data
        # - apply scale
        fDs /= doppler_scale*2.*np.pi
        # - convert to log10 scale
        if flag_logarithmic:
            profile = self._log10_safe(profile)
        # - define data
        data = {'N':1,'x0':fDs,'y0':profile}
        
        #define list of data and the categories of plots
        list_category.append("curve")
        list_data.append(data)
        