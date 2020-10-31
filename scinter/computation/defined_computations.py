import numpy as np
from numpy import newaxis as na
from numpy.ctypeslib import ndpointer
import scipy
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d
from scipy.optimize import curve_fit
import math
import progressbar
import ctypes
import cv2
import skimage
import skimage.morphology as morphology
import time
import skimage
from skimage.measure import block_reduce
from astropy.timeseries import LombScargle #does not work for python 2 !

from .result_source import result_source as scinter_computation

class defined_computations():
    #Useful constants
    ly =  9460730472580800. #m
    au =      149597870700. #m
    pc = 648000./math.pi*au #m
    LightSpeed = 299792458. #m/s
    r_terra = 6371.*1000. #m
    mas = 1./1000.*math.pi/648000. #radians
    
################### internal ###################

    def _find_nearest(self,array,value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
################### dynamic spectrum ###################

    def DS_downsampled(self):
        #load specifications
        DS_source = self._add_specification("DS_source",None)
        new_N_t = self._add_specification("new_N_t",100)
        new_N_nu = self._add_specification("new_N_nu",100)
        
        #load and check data
        if DS_source==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            self._warning("Using {0} instead of unprocessed dynamic spectrum.".format(DS_source))
            source = scinter_computation(self.dict_paths,DS_source[0])
            DynSpec, = source.load_result([DS_source[2]])
            
        #preparations
        DynSpec = np.abs(DynSpec)
        new_t = np.linspace(self.t[0],self.t[-1],num=new_N_t,endpoint=True)
        new_nu = np.linspace(self.nu[0],self.nu[-1],num=new_N_nu,endpoint=True)
        new_dt = new_t[1]-new_t[0]
        new_dnu = new_nu[1]-new_nu[0]
        
        #create containers
        new_DS = np.zeros((self.N_p,self.N_p,new_N_t,new_N_nu),dtype=float)
        
        #perform the computation
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                for i_t in range(new_N_t):
                    # - compute indices in range
                    i_t_l = np.argmax(self.t>new_t[i_t]-new_dt/2.)
                    i_t_u = np.argmax(self.t>new_t[i_t]+new_dt/2.)
                    for i_nu in range(new_N_nu):
                        i_nu_l = np.argmax(self.nu>new_nu[i_nu]-new_dnu/2.)
                        i_nu_u = np.argmax(self.nu>new_nu[i_nu]+new_dnu/2.)
                        sample = DynSpec[i_p1,i_p2,i_t_l:i_t_u,i_nu_l:i_nu_u]
                        good_ones = []
                        for entry in sample.flatten():
                            if entry>0.:
                                good_ones.append(entry)
                        if not len(good_ones)==0:
                            new_DS[i_p1,i_p2,i_t,i_nu] = np.mean(good_ones)
                         
        #save the results
        self._add_result(new_DS,"new_DS")
        self._add_result(new_t,"new_t")
        self._add_result(new_nu,"new_nu")
    
################### secondary spectrum ###################
        
    def secondary_spectrum(self):
        #load specifications
        DS_source = self._add_specification("DS_source",None) #["DynSpec_connected", None, "DynSpec_connected"]
        
        #load and check data
        if DS_source==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            self._warning("Using {0} instead of unprocessed dynamic spectrum.".format(DS_source))
            source = scinter_computation(self.dict_paths,DS_source[0])
            DynSpec, = source.load_result([DS_source[2]])
        
        #create containers
        SecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        SecCrossSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=np.complex)
        phase = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=self.N_nu,endpoint=False)
        
        #perform the computation
        # - compute the power spectrum
        A_DynSpec = np.fft.fft2(DynSpec,axes=(2,3))
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=(2,3))
        #SecSpec = self.delta_nu**2*self.delta_t**2*np.abs(A_DynSpec)**2
        #SecSpec = np.abs(A_DynSpec)**2
        SecSpec = np.abs(A_DynSpec)
        phase = np.angle(A_DynSpec)
        # - compute the cross spectrum
        A_flipped = np.roll(np.flip(A_DynSpec,axis=(2,3)),1,axis=(2,3))
        #SecCrossSpec = self.delta_nu**2*self.delta_t**2*A_DynSpec*A_flipped
        SecCrossSpec = A_DynSpec*A_flipped
        
        #save the results
        self._add_result(SecSpec,"secondary_spectrum")
        self._add_result(SecCrossSpec,"secondary_cross_spectrum")
        self._add_result(phase,"Fourier_phase")
        self._add_result(f_t,"doppler")
        self._add_result(f_nu,"delay")
        
    def SecSpec_slow_t(self):
        #load specifications
        DS_source = self._add_specification("DS_source",None)
        delay_max = self._add_specification("delay_max",6.0)
        tel1 = self._add_specification("tel1",0)
        tel2 = self._add_specification("tel2",0)
        N_t = self._add_specification("N_t",100)
        doppler_max = self._add_specification("doppler_max",10.0)
        #apply scales
        doppler_max *= 2.*math.pi*1.0e-03
        delay_max *= 2.*math.pi*1.0e-06
        
        #load and check data
        if DS_source==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            self._warning("Using {0} instead of unprocessed dynamic spectrum.".format(DS_source))
            source = scinter_computation(self.dict_paths,DS_source[0])
            DynSpec, = source.load_result([DS_source[2]])
        DynSpec = DynSpec[tel1,tel2,:,:]
            
        #create containers
        #SecSpec = np.zeros((N_t,self.N_nu),dtype=float)
        f_t = np.linspace(-doppler_max,doppler_max,num=N_t,endpoint=True)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=self.N_nu,endpoint=False)
        
        #perform the computation
        #FFT over nu
        A_DynSpec = np.fft.fft(DynSpec,axis=1)
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=1)
        #cut discarded delays
        min_index_nu = 0
        max_index_nu = len(f_nu)-1
        for index_nu in range(len(f_nu)):
            if f_nu[index_nu]<0.:
                min_index_nu = index_nu
            elif f_nu[index_nu]>delay_max:
                max_index_nu = index_nu
                break
        f_nu = f_nu[min_index_nu:max_index_nu+2]
        A_DynSpec = A_DynSpec[:,min_index_nu:max_index_nu+2]
        N_nu = len(f_nu)
        #SFT over t
        A_SecSpec = np.zeros((N_t,N_nu),dtype=complex)
        # #test dimensions
        # print(N_nu,2*N_nu,2*N_nu-1)
        # SecSpec = np.zeros((N_t,2*N_nu-1),dtype=float)
        # print(SecSpec[:,N_nu-1:2*N_nu].shape,A_SecSpec.shape)
        # SecSpec[:,N_nu-1:2*N_nu] = np.abs(A_SecSpec[:,:])
        # SecSpec[:,N_nu-1::-1] = np.abs(A_SecSpec[:,:])
        bar = progressbar.ProgressBar(maxval=N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_nu in range(N_nu):
            bar.update(i_nu)
            for i_t in range(N_t):
                A_SecSpec[i_t,i_nu] = np.sum( A_DynSpec[:,i_nu]*np.exp(-1.j*f_t[i_t]*self.t[:]) )
        bar.finish()
        #Use symmetry for other halfSecSpec
        SecSpec = np.zeros((N_t,2*N_nu-1),dtype=float)
        SecSpec[:,N_nu-1:2*N_nu] = np.abs(A_SecSpec[:,:])
        SecSpec[:,N_nu-1::-1] = np.abs(A_SecSpec[::-1,:])
        hf_nu = np.copy(f_nu)
        f_nu = np.empty(2*N_nu-1,dtype=float)
        f_nu[N_nu-1:2*N_nu] = hf_nu[:]
        f_nu[N_nu-1::-1] = -hf_nu[:]
                
        #save the results
        self._add_result(SecSpec,"secondary_spectrum")
        self._add_result(f_t,"doppler")
        self._add_result(f_nu,"delay")
        
    def nut_SecSpec(self):
        #load specifications
        DS_source = self._add_specification("DS_source",None) #["DynSpec_connected", None, "DynSpec_connected"]
        
        #load and check data
        if DS_source==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            self._warning("Using {0} instead of unprocessed dynamic spectrum.".format(DS_source))
            source = scinter_computation(self.dict_paths,DS_source[0])
            DynSpec, = source.load_result([DS_source[2]])
        
        #import c code for discrete Fourier transform
        lib = self._load_c_lib("nut_transform")
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
        SecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
#        SecCrossSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=np.complex)
#        phase = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
        f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=self.N_nu,endpoint=False)
        
        #preparations
        ntime = self.N_t
        nfreq = self.N_nu
        r0 = np.fft.fftfreq(ntime)
        delta_r = r0[1] - r0[0]
        src = np.linspace(0, 1, ntime).astype('float64')
        src = np.arange(ntime).astype('float64')
    
        # Reference freq. to middle of band, should change this
        midf = len(self.nu)//2
        fref = self.nu[midf]
        fscale = self.nu / fref
        fscale = fscale.astype('float64')
    
        #perform the computation
        A_DynSpec = np.zeros(DynSpec.shape,dtype=np.complex)
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                # - declare the empty result array:
                SS = np.empty((ntime, nfreq), dtype=np.complex128)
                # - call the DFT:
                lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale, src, DynSpec[i_p1,i_p2,:,:].astype('float64'), SS)
                # - flip along time
                SS = SS[::-1]
                # - correct zero point
                SS = np.roll(SS,1,axis=0)
                # - Still need to FFT y axis, should change to pyfftw for memory and
                #   speed improvement
                SS = np.fft.fft(SS, axis=1)
                SS = np.fft.fftshift(SS, axes=1)
                A_DynSpec[i_p1,i_p2] = SS
        #SecSpec = self.delta_nu**2*self.delta_t**2*np.abs(A_DynSpec)**2
        SecSpec = np.abs(A_DynSpec)
#        phase = np.angle(A_DynSpec)
#        # - compute the cross spectrum
#        A_flipped = np.roll(np.flip(A_DynSpec,axis=(2,3)),1,axis=(2,3))
#        SecCrossSpec = self.delta_nu**2*self.delta_t**2*A_DynSpec*A_flipped
        
        #save the results
        self._add_result(SecSpec,"secondary_spectrum")
#        self._add_result(SecCrossSpec,"secondary_cross_spectrum")
#        self._add_result(phase,"Fourier_phase")
        self._add_result(f_t,"doppler")
        self._add_result(f_nu,"delay")
        
    def clean_SecSpec(self):
        #load and check data
        DSpec = self._load_base_file("DynSpec")
        flags = self._load_base_file("flags")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        
        #load specifications
        N_iter = self._add_specification("N_iter",10)
        step_iter = self._add_specification("step_iter",0.1)
        flag_continue = self._add_specification("flag_continue",False)
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        
        #create containers
        if flag_continue:
            source = scinter_computation(self.dict_paths,"clean_SecSpec")
            cSecSpec,SSpec,psf = source.load_result(["cSecSpec","residuals","psf"])
        else:
            DSpec = DSpec[pos1,pos2,:,:]
            SSpec = np.fft.fft2(DSpec,axes=(0,1))
            SSpec[:,0] = 0.
            SSpec = np.fft.fftshift(SSpec,axes=(0,1))
            cSecSpec = np.zeros((self.N_t,self.N_nu),dtype=complex)
            # - FFT the mask and use result as psf
            mask = np.ones(SSpec.shape,dtype=float)
            mask[flags] = 0.
            psf = np.fft.fft2(mask,axes=(0,1))
            psf = np.fft.fftshift(psf,axes=(0,1))
            midpoint = np.array([len(f_t)/2,len(f_nu)/2])
            psf = psf/np.abs(psf[midpoint[0],midpoint[1]])
        
        #perform the computation
        bar = progressbar.ProgressBar(maxval=N_iter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_iter in xrange(N_iter):
            bar.update(i_iter)
            max_cpower = np.amax(np.abs(SSpec))
            location = np.where(np.abs(np.abs(SSpec)-max_cpower)==0.)
            i_t,i_nu = list(zip(location[0],location[1]))[0]
            cSecSpec[i_t,i_nu] = cSecSpec[i_t,i_nu]+step_iter*SSpec[i_t,i_nu]
            shift_t = midpoint[0]-i_t
            shift_nu = midpoint[1]-i_nu
            #shifted_psf = np.empty_like(psf) #,(shift_t,shift_nu),axis=(0,1)
            shifted_psf = scipy.ndimage.interpolation.shift(np.real(psf),[shift_t,shift_nu],cval=0.) + 1.0j*scipy.ndimage.interpolation.shift(np.imag(psf),[shift_t,shift_nu],cval=0.)
            SSpec = SSpec - step_iter*SSpec[i_t,i_nu]*shifted_psf
        bar.finish()
        
        #save the results
        self._add_result(psf,"psf")
        self._add_result(cSecSpec,"cSecSpec")
        self._add_result(SSpec,"residuals")
        
    def LombScargle_SecSpec(self):
        #load and check data
        DSpec = self._load_base_file("DynSpec")
        flags = self._load_base_file("flags")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        f_t,f_nu = source.load_result(["doppler","delay"])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        
        #create containers
        DSpec = DSpec[pos1,pos2,:,:]
        #SSpec = SSpec[pos1,pos2,:,:]
        
        LS_SSpec = np.zeros((self.N_t,self.N_nu),dtype=complex)
        t = np.delete(self.t,flags)
        DSpec = np.real(np.delete(DSpec,flags,axis=0))
        DSpec -= np.median(DSpec)
        
        #perform the computation
        amplitude = np.zeros(f_t.shape,dtype = float)
        angle = np.zeros(f_t.shape,dtype = float)
        posf_t = f_t[int(self.N_t/2):]
        bar = progressbar.ProgressBar(maxval=self.N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_nu in range(self.N_nu):
            bar.update(i_nu)
            if not (np.std(DSpec[:,i_nu])==0. or np.isnan(np.std(DSpec[:,i_nu])).any()):
                LS = LombScargle(t, DSpec[:,i_nu])
                # if i_nu==0:
                    # print("LS: type={0}".format(type(LS)))
                if self.N_t % 2 != 0:
                    is1_t = int(self.N_t/2)+1
                    is2_t = int(self.N_t/2)
                else:
                    is1_t = int(self.N_t/2)
                    is2_t = int(self.N_t/2)
                # if i_nu==0:
                    # print("N_t={0}, is1_t={1}, is2_t={2}".format(self.N_t,is1_t,is2_t))
                amplitude[is2_t:] = np.sqrt(LS.power(posf_t,method='cython'))
                amplitude[1:is1_t] = amplitude[self.N_t+1:is2_t:-1]
                # if i_nu==0:
                    # print("amplitude: type={0}, shape={1}, std={2}, dtype={3}".format(type(amplitude),amplitude.shape,np.std(amplitude),amplitude.dtype))
                if np.isnan(np.std(amplitude)):
                    print("{0}: amplitude contains nan".format(i_nu))
                    print("DSpec.std={0}".format(np.std(DSpec[:,i_nu])))
                    for i,v in enumerate(amplitude):
                        if np.isnan(v):
                            print("at {0}".format(i))
                for i_t in range(int(self.N_t/2)+1,self.N_t):
                    angle[i_t] = np.angle(LS.model_parameters(f_t[i_t])[2]+1.0j*LS.model_parameters(f_t[i_t])[1])
                angle[1:is1_t] = -angle[self.N_t+1:is2_t:-1]
                # if i_nu==0:
                    # print("angle: type={0}, shape={1}, std={2}, dtype={3}".format(type(angle),angle.shape,np.std(angle),angle.dtype))
                # if np.isnan(np.std(angle)):
                    # print("{0}: angle contains nan".format(i_nu))
                    # for i,v in enumerate(angle):
                        # if np.isnan(v):
                            # print("at {0}".format(i))
                LS_SSpec[:,i_nu] = amplitude*np.exp(1.0j*angle)
                # if i_nu==0:
                    # print("LS_SSpec: type={0}, shape={1}, std={2}, dtype={3}".format(type(LS_SSpec),LS_SSpec.shape,np.std(LS_SSpec),LS_SSpec.dtype))
                # if np.isnan(np.std(LS_SSpec[:,i_nu])):
                    # print("{0}: LS_SSpec contains nan".format(i_nu))
                    # for i,v in enumerate(LS_SSpec[:,i_nu]):
                        # if np.isnan(v):
                            # print("at {0}".format(i))
        bar.finish()
        # - perform FFT along remaining axis
        LS_SSpec = np.fft.fft(LS_SSpec, axis=1)
        LS_SSpec = np.fft.fftshift(LS_SSpec, axes=1)
        LS_SSpec = LS_SSpec[::-1,:]
        LS_SSpec = np.roll(LS_SSpec,1,axis=0)
        print("LS_SSpec: type={0}, shape={1}, std={2}, dtype={3}".format(type(LS_SSpec),LS_SSpec.shape,np.std(LS_SSpec),LS_SSpec.dtype))
        
        #save the results
        self._add_result(LS_SSpec,"LombScargle_SecSpec")
        
    def iterative_SecSpec(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        flags = self._load_base_file("flags")
        
        #create containers
        SecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        phase = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        
        #perform the computation
        A_DynSpec = np.zeros((self.N_t,self.N_nu),dtype=complex)
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                bar = progressbar.ProgressBar(maxval=self.N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for i_nu in range(self.N_nu):
                    bar.update(i_nu)
                    row = np.copy(DynSpec[i_p1,i_p2,:,i_nu])
                    test = np.fft.fft(row)
                    test = np.fft.fftshift(test)
                    for i_t in range(self.N_t):
                        wtest = np.copy(test)
                        wtest[i_t] = 0.
                        wrow = row - np.fft.ifft(np.fft.ifftshift(wtest))
                        wrow[flags] = 0.
                        wresult = np.fft.fftshift(np.fft.fft(wrow))
                        A_DynSpec[i_t,i_nu] = wresult[i_t]
                bar.finish()
                A_DynSpec = np.fft.fftshift(np.fft.fft(A_DynSpec,axis=1),axes=1)
                SecSpec[i_p1,i_p2,:,:] = np.abs(A_DynSpec)
                phase[i_p1,i_p2,:,:] = np.angle(A_DynSpec)
        
        #save the results
        self._add_result(SecSpec,"secondary_spectrum")
        self._add_result(phase,"Fourier_phase")
        
    def pulse_variation(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SSpec,f_t,f_nu,phase = source.load_result(["secondary_spectrum","doppler","delay","Fourier_phase"])
        
        #load specifications
        min_doppler = self._add_specification("min_doppler",0.0)
        
        #create containers
        pulse_variation = np.zeros((self.N_p,self.N_p,self.N_t),dtype=float)
        line0 = np.zeros((self.N_p,self.N_p,self.N_t),dtype=complex)
        
        #perform the computation
        i_nu0 = np.argwhere(f_nu==0.)[0][0]
        line0 = SSpec[:,:,:,i_nu0]*np.exp(1.j*phase[:,:,:,i_nu0])
        is_dop0 = np.argwhere(np.abs(f_t)<=min_doppler)
        line0[:,:,is_dop0] = 0.
        pulse_variation = np.abs(np.fft.ifft(line0,axis=2))
        
        #save the results
        self._add_result(pulse_variation,"pulse_variation")
        
    def DynSpec_noPulseVar(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"pulse_variation")
        pulse_variation, = source.load_result(["pulse_variation"])
        
        #load specifications
        min_DS = self._add_specification("min_amplitude",None)
        fraction = self._add_specification("fraction",0.9)
        
        #create containers
        DynSpec_noPulseVar = np.real(DynSpec)
        
        #perform the computation
        PulseVar = pulse_variation/self.N_nu
        DynSpec_noPulseVar = DynSpec_noPulseVar[:,:,:,:]-PulseVar[:,:,:,na]
        
        
        # if not min_DS==None:
            # DynSpec_noPulseVar[DynSpec_noPulseVar<min_DS] = 0.
        # N_low = int((1-fraction)*pulse_variation.shape[2])
        # for i_p1 in range(self.N_p):
            # for i_p2 in range(self.N_p):
                # is_low = np.argpartition(np.abs(pulse_variation[i_p1,i_p2,:]), N_low)[:N_low]
                
                # scale = pulse_variation[i_p1,i_p2,:] - np.min(pulse_variation[i_p1,i_p2,:])
                # median_DS = np.median(DynSpec_noPulseVar[i_p1,i_p2,:,:],axis=1)
                # median_DS -= np.min(median_DS)
                # DynSpec_noPulseVar[i_p1,i_p2,:,:] = DynSpec_noPulseVar[i_p1,i_p2,:,:]-np.min(median_DS)
                # max_pulse = np.max(scale)
                # max_DS = np.max(median_DS)
                # scale *= max_DS/max_pulse
                
                # #DynSpec_noPulseVar[i_p1,i_p2,:,:] = DynSpec_noPulseVar[i_p1,i_p2,:,:]/(scale[:,na]+0.5*max_DS)
                # DynSpec_noPulseVar[i_p1,i_p2,:,:] = DynSpec_noPulseVar[i_p1,i_p2,:,:]-scale[:,na]
                
                # DynSpec_noPulseVar[i_p1,i_p2,is_low,:] = 0.
                # #pulse_variation[i_p1,i_p2,is_low] = 0.
                
                
        #DynSpec_noPulseVar = DynSpec_noPulseVar[:,:,:,:]/pulse_variation[:,:,:,na]
        
        
        #save the results
        self._add_result(DynSpec_noPulseVar,"DynSpec_noPulseVar")
        
    def DynSpec_connected(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        
        #load specifications
        smooth_steps = self._add_specification("smooth_steps",6)
        
        #create containers
        DSc = np.real(DynSpec)
        lmean_DSC = np.empty((self.N_t),dtype=float)
        rmean_DSC = np.empty((self.N_t),dtype=float)
        
        #perform the computation
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p): 
                # - compute smoothed spectrum
                for i_t in range(self.N_t):
                    v_lmean = 0.
                    v_rmean = 0.
                    i_l = np.max([0,i_t-smooth_steps])
                    i_u = np.min([self.N_t-1,i_t+smooth_steps])
                    N_lmean = i_t-i_l+1
                    N_rmean = i_u-i_t+1
                    for i_mean in range(i_l,i_t+1):
                        v_lmean += np.mean(DSc[i_p1,i_p2,i_mean,:])
                    for i_mean in range(i_t,i_u+1):
                        v_rmean += np.mean(DSc[i_p1,i_p2,i_mean,:])
                    v_lmean /= N_lmean
                    v_rmean /= N_rmean
                    lmean_DSC[i_t] = v_lmean
                    rmean_DSC[i_t] = v_rmean
                # - cancel difference that deviates from smoothed trend
                for i_t in range(1,self.N_t):
                    delta = np.mean(DSc[i_p1,i_p2,i_t,:])-np.mean(DSc[i_p1,i_p2,i_t-1,:])
                    delta -= (rmean_DSC[i_t]-lmean_DSC[i_t-1])/(smooth_steps+1.)
                    DSc[i_p1,i_p2,i_t,:] -= delta
                    
        #save the results
        self._add_result(DSc,"DynSpec_connected")
        
    def halfSecSpec(self):
        #load specifications
        
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        
        #create containers
        halfSecSpec = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        phase = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        DS = np.copy(DynSpec)
        median_DS = np.zeros(DS.shape,dtype=float)
        
        #perform the computation
        # - fill zero flags by median of nonzero values
        DS[DS==0.] = np.nan
        median_DS[:,:,:,:] = np.nanmedian(np.real(DS),axis=3)[:,:,:,na]
        DS[np.isnan(DS)] = median_DS[np.isnan(DS)]
        # or i_p1 in range(self.N_p):
            # for i_p2 in range(self.N_p):
                # for i_t in range(self.N_t):
                    # #median_DS = np.nanmedian(DS[i_p1,i_p2,i_t,:])
                    # #is_DS0 = np.argwhere(np.isnan(DS[i_p1,i_p2,i_t,:]))
                    # for i_nu in range(self.N_nu):
                        
                    
        # - compute the power spectrum along frequency
        A_DynSpec = np.fft.fft(DS,axis=3)
        A_DynSpec = np.fft.fftshift(A_DynSpec,axes=3)
        halfSecSpec = np.abs(A_DynSpec)
        phase = np.angle(A_DynSpec)
        
        #save the results
        self._add_result(halfSecSpec,"halfSecSpec")
        self._add_result(phase,"Fourier_phase")
        
    def SecSpec_divPulseVar(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"halfSecSpec")
        hSS,hSS_phase = source.load_result(["halfSecSpec","Fourier_phase"])
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        f_t,f_nu = source.load_result(["doppler","delay"])
        
        #load specifications
        smooth_dt = self._add_specification("smooth_dt",30.) #s
        tau_0l = self._add_specification("tau_0l",4.0e-6) #s
        tau_0u = self._add_specification("tau_0u",6.0e-6) #s
        
        #create containers
        SecSpec_divPulseVar = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        DynSpec_divPulseVar = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        pulse_variation = np.zeros((self.N_p,self.N_p,self.N_t),dtype=float)
        DS_smooth = np.empty((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        PV_smooth = np.empty_like(pulse_variation)
        noise_threshold = np.empty((self.N_p,self.N_p,self.N_nu),dtype=float)
        DS = np.abs(DynSpec)
        
        #perform the computation
        f_nu /= 2.*math.pi
        # - find range in tau used to extract the pulse variation by summing over it
        for i_nu in range(self.N_nu):
            if (tau_0l<f_nu[i_nu]<tau_0u): # or (-tau_0l>f_nu[i_nu]>-tau_0u)
                #pulse_variation += np.real(hSS[:,:,:,i_nu]*np.exp(1.j*hSS_phase[:,:,:,i_nu]))
                pulse_variation += hSS[:,:,:,i_nu]
        # - create the smoothed dynamic spectrum
        N_smooth = int(smooth_dt/self.delta_t)
        DS_smooth = uniform_filter1d(DS, size=N_smooth, axis=2, mode='nearest')
        PV_smooth = uniform_filter1d(pulse_variation, size=N_smooth, axis=2, mode='nearest')
        # - fit pulse variation to points above median and subtract the result
        noise_threshold = np.median(DS_smooth,axis=2)
        for i_p1 in range(self.N_p):
            for i_p2 in range(self.N_p):
                for i_nu in range(self.N_nu):
                    ar_i_t = []
                    ar_DS = []
                    for i_t in range(self.N_t):
                        if DS_smooth[i_p1,i_p2,i_t,i_nu]>noise_threshold[i_p1,i_p2,i_nu]:
                            ar_i_t.append(i_t)
                            ar_DS.append(DS_smooth[i_p1,i_p2,i_t,i_nu])
                    if not ar_DS == []:
                        def fitfunc(in_x,in_a,in_b):
                            N_x = len(in_x)
                            result = np.empty(N_x,dtype=float)
                            for i_x in range(N_x):
                                result[i_x] = PV_smooth[i_p1,i_p2,int(in_x[i_x])]*in_a+in_b
                            return result
                        popt, pcov = curve_fit(fitfunc,ar_i_t,ar_DS)
                        #DynSpec_divPulseVar[i_p1,i_p2,:,i_nu] = np.abs(DynSpec[i_p1,i_p2,:,i_nu])-(pulse_variation[i_p1,i_p2,:]*popt[0]+popt[1])
                        DynSpec_divPulseVar[i_p1,i_p2,:,i_nu] = np.abs(DynSpec[i_p1,i_p2,:,i_nu])-(PV_smooth[i_p1,i_p2,:]*popt[0]+popt[1])
        SecSpec_divPulseVar = np.fft.fft2(DynSpec_divPulseVar,axes=(2,3))
        SecSpec_divPulseVar = np.abs(np.fft.fftshift(SecSpec_divPulseVar,axes=(2,3)))
        
        #check validity of results
        assert np.std(pulse_variation)>0.
        
        #save the results
        self._add_result(SecSpec_divPulseVar,"SecSpec_divPulseVar")
        self._add_result(DynSpec_divPulseVar,"DynSpec_divPulseVar")
        self._add_result(pulse_variation,"pulse_variation")
        self._add_result(PV_smooth,"PV_smooth")
        
    def SecSpec_scale_cut(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        
        #load specifications
        smooth_dt = self._add_specification("smooth_dt",30.) #s
        
        #create containers
        SecSpec_scale_cut = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        DynSpec_scale_cut = np.zeros((self.N_p,self.N_p,self.N_t,self.N_nu),dtype=float)
        
        #perform the computation
        # - create the smoothed dynamic spectrum
        N_smooth = int(smooth_dt/self.delta_t)
        DS_smooth = uniform_filter1d(np.abs(DynSpec), size=N_smooth, axis=2, mode='nearest')
        # - compute secondary spectrum from long range subtracted dynamic spectrum
        DynSpec_scale_cut = np.abs(DynSpec) - DS_smooth
        SecSpec_scale_cut = np.fft.fft2(DynSpec_scale_cut,axes=(2,3))
        SecSpec_scale_cut = np.abs(np.fft.fftshift(SecSpec_scale_cut,axes=(2,3)))
        # - rescale
        #DynSpec_scale_cut = DynSpec_scale_cut[:,:,:,:]*np.mean(np.abs(DynSpec),axis=(2,3))[:,:,na,na]/np.mean(DynSpec_scale_cut,axis=(2,3))[:,:,na,na]
        
        #save the results
        self._add_result(SecSpec_scale_cut,"SecSpec_scale_cut")
        self._add_result(DynSpec_scale_cut,"DynSpec_scale_cut")
        
    def deconvolve_SecSpec_1pix(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SecSpec,f_t,f_nu,SSphase = source.load_result(["secondary_spectrum","doppler","delay","Fourier_phase"])
        
        #load specifications
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        fD_min = self._add_specification("fD_min",-10.0e-3)
        fD_max = self._add_specification("fD_max",-0.5e-3)
        tau = self._add_specification("tau",0.7e-6)
        fD = self._add_specification("fD",float(np.sqrt(tau/0.08825745414070099)*np.sign(fD_min)))
        
        #preparation
        # - take single dish for simplicity
        DynSpec = DynSpec[pos1,pos2,:,:]
        SecSpec = SecSpec[pos1,pos2,:,:]*np.exp(1.0j*SSphase[pos1,pos2,:,:])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - get indices
        #dft = f_t[1]-f_t[0]
        #dfnu = f_nu[1]-f_nu[0]
        idx_tau = self._find_nearest(f_nu,tau)
        idx_fD = self._find_nearest(f_t,fD)
        idx_fD0 = self._find_nearest(f_t,0.)
        idx_tau0 = self._find_nearest(f_nu,0.)
        idx_fD_max = self._find_nearest(f_t,fD_max)
        idx_fD_min = self._find_nearest(f_t,fD_min)
        
        #create containers
        kernel = np.empty_like(DynSpec)
        DynSpec_deconv = np.empty_like(DynSpec)
        SecSpec_deconv = np.empty_like(SecSpec)
        
        #perform the computation
        # - get area to deconvolve into point
        Fkernel = np.zeros_like(SecSpec)
        Fkernel[idx_fD_min+idx_fD0-idx_fD:idx_fD_max+idx_fD0-idx_fD+1,idx_tau0] = SecSpec[idx_fD_min:idx_fD_max+1,idx_tau]
        #print([i for i in range(idx_fD_min,idx_fD_max+1)])
        #print([i for i in range(idx_fD_min+idx_fD0-idx_fD,idx_fD_max+idx_fD0-idx_fD+1)])
        #print(idx_fD_min,idx_fD,idx_fD_max,idx_fD0)
        kernel = np.fft.ifft2(np.fft.ifftshift(Fkernel,axes=(0,1)))
        kernel /= np.mean(np.abs(kernel))
        #print(np.std(Fkernel),np.std(kernel))
        DynSpec_deconv = DynSpec/kernel
        SecSpec_deconv = np.fft.fftshift(np.fft.fft2(DynSpec_deconv),axes=(0,1))
        
        #save the results
        self._add_result(kernel,"kernel")
        self._add_result(DynSpec_deconv,"DynSpec_deconv")
        self._add_result(SecSpec_deconv,"SecSpec_deconv")
        
    def deconvolve_SecSpec(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SecSpec,f_t,f_nu,SSphase = source.load_result(["secondary_spectrum","doppler","delay","Fourier_phase"])
        
        #load specifications
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        fD_min = self._add_specification("fD_min",0.5e-3)
        tau_min = self._add_specification("tau_min",0.3e-6)
        tau_max = self._add_specification("tau_max",4.0e-6)
        eta = self._add_specification("curvature",0.08825745414070099)
        #fD = self._add_specification("fD",float(np.sqrt(tau/0.08825745414070099)*np.sign(fD_min)))
        
        #preparation
        # - take single dish for simplicity
        DynSpec = DynSpec[pos1,pos2,:,:]
        SecSpec = SecSpec[pos1,pos2,:,:]*np.exp(1.0j*SSphase[pos1,pos2,:,:])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - get indices
        #dft = f_t[1]-f_t[0]
        #dfnu = f_nu[1]-f_nu[0]
        idx_tau_min = self._find_nearest(f_nu,tau_min)
        idx_tau_max = self._find_nearest(f_nu,tau_max)
        idx_fD0 = self._find_nearest(f_t,0.)
        
        #create containers
        kernel = np.empty_like(DynSpec)
        DynSpec_deconv = np.empty_like(DynSpec)
        SecSpec_deconv = np.empty_like(SecSpec)
        
        #perform the computation
        Fkernel = np.zeros_like(SecSpec)
        weights = np.zeros_like(f_t)
        bar = progressbar.ProgressBar(maxval=idx_tau_max-idx_tau_min, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_tau in range(idx_tau_min,idx_tau_max+1):
            bar.update(i_tau-idx_tau_min)
            fD_pos = np.sqrt(f_nu[i_tau]/eta)
            fD_neg = -np.sqrt(f_nu[i_tau]/eta)
            arcpix_pos = self._find_nearest(f_t,fD_pos)
            arcpix_neg = self._find_nearest(f_t,fD_neg)
            weight_pos = np.abs(SecSpec[arcpix_pos,i_tau])
            phase_pos = np.angle(SecSpec[arcpix_pos,i_tau])
            weight_neg = np.abs(SecSpec[arcpix_neg,i_tau])
            phase_neg = np.angle(SecSpec[arcpix_neg,i_tau])
            for i_fD,fD in enumerate(f_t):
                if fD<-fD_min:
                    Fkernel[i_fD-arcpix_neg+idx_fD0,0] += SecSpec[i_fD,i_tau]*np.exp(-1.0j*phase_neg)
                    weights[i_fD-arcpix_neg+idx_fD0] += weight_neg
                elif fD>fD_min:
                    Fkernel[i_fD-arcpix_pos+idx_fD0,0] += SecSpec[i_fD,i_tau]*np.exp(-1.0j*phase_pos)
                    weights[i_fD-arcpix_pos+idx_fD0] += weight_pos
        bar.finish()
        weights[weights==0.] = 1.
        Fkernel = Fkernel/weights[:,na]
        #compute kernel to deconvolve
        kernel = np.fft.ifft2(np.fft.ifftshift(Fkernel,axes=(0,1)))
        kernel = kernel/np.mean(np.abs(kernel))
        skernel = np.empty_like(kernel)
        for i_tau in range(len(f_nu)):
            skernel[:,i_tau] = np.convolve(kernel[:,i_tau], np.ones(3)/3, mode='same')
        kernel[1:-2,:] = skernel[1:-2,:]
        #deconvolve (using only phases)
        DynSpec_deconv = DynSpec/kernel*np.abs(kernel)
        SecSpec_deconv = np.fft.fftshift(np.fft.fft2(DynSpec_deconv),axes=(0,1))
        
        #save the results
        self._add_result(kernel,"kernel")
        self._add_result(DynSpec_deconv,"DynSpec_deconv")
        self._add_result(SecSpec_deconv,"SecSpec_deconv")
        
    def SecSpec_fit_stripes(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SecSpec,f_t,f_nu,SSphase = source.load_result(["secondary_spectrum","doppler","delay","Fourier_phase"])
        
        #load specifications
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        fD_min = self._add_specification("fD_min",0.5e-3)
        tau_min = self._add_specification("tau_min",0.3e-6)
        tau_max = self._add_specification("tau_max",4.0e-6)
        eta = self._add_specification("curvature",0.08825745414070099)
        flag_restart = self._add_specification("restart",False)
        fitmethod = self._add_specification("fitmethod","combined")
        #fD = self._add_specification("fD",float(np.sqrt(tau/0.08825745414070099)*np.sign(fD_min)))
        
        #preparation
        # - take single dish for simplicity
        DynSpec = np.real(DynSpec[pos1,pos2,:,:])
        SecSpec = SecSpec[pos1,pos2,:,:]*np.exp(1.0j*SSphase[pos1,pos2,:,:])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - get indices
        #dft = f_t[1]-f_t[0]
        #dfnu = f_nu[1]-f_nu[0]
        idx_tau_min = self._find_nearest(f_nu,tau_min)
        idx_tau_max = self._find_nearest(f_nu,tau_max)
        idx_fD0 = self._find_nearest(f_t,0.)
        idx_fD_min_neg = self._find_nearest(f_t,-fD_min)
        idx_fD_min_pos = self._find_nearest(f_t,fD_min)
        
        #create containers
        kernel = np.empty(self.N_t,dtype=complex)
        DynSpec_deconv = np.empty_like(DynSpec)
        DynSpec_multiplicator = np.empty_like(DynSpec)
        SecSpec_deconv = np.empty_like(SecSpec)
        
        #perform the computation
        N_im = 2*(idx_tau_max-idx_tau_min+1)
        signal = np.empty(N_im,dtype=complex)
        input = np.empty(N_im+self.N_t,dtype=complex)
        input[0:N_im] = signal
        input[N_im:] = kernel
        is_tau_neg = np.arange(idx_tau_max,idx_tau_min-1,-1,dtype=int)
        is_tau_pos = np.arange(idx_tau_min,idx_tau_max+1,1,dtype=int)
        is_tau = np.concatenate((is_tau_neg,is_tau_pos))
        arc_loc = np.empty((N_im,2),dtype=int)
        for i_im,i_tau in enumerate(is_tau):
            if i_im<N_im/2.:
                arc_loc[i_im,0] = self._find_nearest(f_t,-np.sqrt(f_nu[i_tau]/eta))
            else:
                arc_loc[i_im,0] = self._find_nearest(f_t,np.sqrt(f_nu[i_tau]/eta))
            arc_loc[i_im,1] = i_tau
        
        def func_for_min(input,SecSpec,arc_loc,N_im,idx_fD_min_neg,idx_fD_min_pos):
            signal = input[0:N_im]+1.0j*input[N_im+self.N_t:2*N_im+self.N_t]
            kernel = input[N_im:N_im+self.N_t]+1.0j*input[2*N_im+self.N_t:]
            
            fit = np.empty_like(SecSpec)
            
            #fit[arc_loc[:,0],arc_loc[:,1]] = signal
            
            for i_im in range(N_im):
                fit[arc_loc[i_im,0],arc_loc[i_im,1]] = signal[i_im]
            for i_im in range(int(N_im/2)):
                fit[:,arc_loc[i_im,1]] = np.convolve(fit[:,arc_loc[i_im,1]],kernel[::-1],mode='same')
                
            diff = np.abs(SecSpec[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])]-fit[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])])**2
            #diff = diff[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])]
            diff[idx_fD_min_neg:1+idx_fD_min_pos,:] = 0.
            chi2 = np.sum(diff)
            print(chi2)
            
            # - compute analytical gradient
            grad = np.empty_like(input)
            i_tau_min = np.min(arc_loc[:,1])
            i_tau_max = np.max(arc_loc[:,1])
            # - derivative with respect to signal
            for i_im in range(N_im):
                i_im_i = i_im+N_im+self.N_t
                result_r = 0.+0.j
                result_i = 0.+0.j
                result_r = np.sum(SecSpec[:,arc_loc[i_im,1]]-fit[:,arc_loc[i_im,1]]*np.roll(kernel,-arc_loc[i_im,0]))
                result_r += np.conj(result_r)
                grad[i_im] = np.real(result_r)
                result_i = np.sum(SecSpec[:,arc_loc[i_im,1]]-fit[:,arc_loc[i_im,1]]*np.roll(kernel,-arc_loc[i_im,0])*1.j)
                result_i += np.conj(result_i)
                grad[i_im_i] = np.real(result_i)
            # - derivative with respect to kernel
            for i_fD in range(self.N_t):
                i_in_r = i_fD+N_im
                i_in_i = i_fD+2*N_im+self.N_t
                result_r = 0.+0.j
                result_i = 0.+0.j
                for i_im in range(N_im):
                    i_fD_shifted = i_fD+arc_loc[i_im,0]
                    if 0<i_fD_shifted<self.N_t:
                        i_tau = arc_loc[i_im,1]
                        result_r += (SecSpec[i_fD_shifted,i_tau]-fit[i_fD_shifted,i_tau])*signal[i_im]
                        result_i += (SecSpec[i_fD_shifted,i_tau]-fit[i_fD_shifted,i_tau])*signal[i_im]*1.j
                result_r += np.conj(result_r)
                result_i += np.conj(result_i)
                grad[i_in_r] = np.real(result_r)
                grad[i_in_i] = np.real(result_i)
                
            return chi2,grad
            
        def func_combfit(input,SecSpec,arc_loc,signal0,idx_fD_min_neg,idx_fD_min_pos):
            kernel = input[0:self.N_t]+1.0j*input[self.N_t:2*self.N_t]
            fit = np.empty_like(SecSpec)
            N_im = len(signal0)
            signal = np.empty_like(signal0)
            N_tau = int(N_im/2)
            
            for i_tau in range(N_tau):
                data = np.concatenate((SecSpec[0:idx_fD_min_neg,arc_loc[i_tau,1]],SecSpec[idx_fD_min_pos:,arc_loc[i_tau,1]]))
                sig1 = np.pad(kernel,((0,arc_loc[i_tau,0])), mode='constant')[arc_loc[i_tau,0]:]
                fD1 = np.concatenate((sig1[0:idx_fD_min_neg],sig1[idx_fD_min_pos:]))
                sig2 = np.pad(kernel,((arc_loc[i_tau,0],0)), mode='constant')[:-arc_loc[i_tau,0]]
                fD2 = np.concatenate((sig2[0:idx_fD_min_neg],sig2[idx_fD_min_pos:]))
                init1 = signal0[i_tau]
                init2 = signal0[-i_tau-1]
                signal[i_tau],signal[-i_tau-1] = self._fitfunc_for_linefit(data,fD1,fD2,init1,init2)
            
            for i_im in range(N_im):
                fit[arc_loc[i_im,0],arc_loc[i_im,1]] = signal[i_im]
            for i_im in range(int(N_im/2)):
                fit[:,arc_loc[i_im,1]] = np.convolve(fit[:,arc_loc[i_im,1]],kernel[::-1],mode='same')
                
            diff = np.abs(SecSpec[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])]-fit[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])])**2
            diff[idx_fD_min_neg:1+idx_fD_min_pos,:] = 0.
            chi2 = np.sum(diff)
            print(chi2)
            
            if 0:
                # - compute analytical gradient
                grad = np.empty_like(input)
                # - derivative with respect to kernel
                for i_fD in range(self.N_t):
                    i_in_r = i_fD
                    i_in_i = i_fD+self.N_t
                    result_r = 0.+0.j
                    result_i = 0.+0.j
                    for i_im in range(N_im):
                        i_fD_shifted = i_fD+arc_loc[i_im,0]
                        if 0<i_fD_shifted<self.N_t:
                            i_tau = arc_loc[i_im,1]
                            result_r += (SecSpec[i_fD_shifted,i_tau]-fit[i_fD_shifted,i_tau])*signal[i_im]
                            result_i += (SecSpec[i_fD_shifted,i_tau]-fit[i_fD_shifted,i_tau])*signal[i_im]*1.j
                    result_r += np.conj(result_r)
                    result_i += np.conj(result_i)
                    grad[i_in_r] = np.real(result_r)
                    grad[i_in_i] = np.real(result_i)
                    
                return chi2,grad
            return chi2
        
        if flag_restart:
            source = scinter_computation(self.dict_paths,"SecSpec_fit_stripes")
            signal0,kernel0 = source.load_result(["signal","kernel"])
        else:
            signal0 = SecSpec[arc_loc[:,0],arc_loc[:,1]]/np.abs(SecSpec[arc_loc[:,0],arc_loc[:,1]])
            kernel0 = np.fft.fftshift(np.fft.fft(np.mean(DynSpec,axis=1)))
        if 1:
            if fitmethod == "minimize_all":
                v0 = np.concatenate((np.real(signal0),np.real(kernel0),np.imag(signal0),np.imag(kernel0)))
                res = scipy.optimize.minimize(
                    func_for_min, v0, args= (SecSpec,arc_loc,N_im,idx_fD_min_neg,idx_fD_min_pos),
                    method="TNC", jac=True, options=dict(disp=True,return_all=True) )
                signal = res.x[0:N_im]+1.0j*res.x[N_im+self.N_t:2*N_im+self.N_t]
                kernel = res.x[N_im:N_im+self.N_t]+1.0j*res.x[2*N_im+self.N_t:]
            elif fitmethod == "combined":
                v0 = np.concatenate((np.real(kernel0),np.imag(kernel0)))
                res = scipy.optimize.minimize(
                    func_combfit, v0, args= (SecSpec,arc_loc,signal0,idx_fD_min_neg,idx_fD_min_pos),
                    method="Powell", jac=False, options=dict(disp=True,return_all=True) )
                #signal = res.x[0:N_im]+1.0j*res.x[N_im+self.N_t:2*N_im+self.N_t]
                kernel = res.x[0:self.N_t]+1.0j*res.x[self.N_t:2*self.N_t]
                N_tau = int(N_im/2)
                signal = np.empty_like(signal0)
                for i_tau in range(N_tau):
                    data = np.concatenate((SecSpec[0:idx_fD_min_neg,arc_loc[i_tau,1]],SecSpec[idx_fD_min_pos:,arc_loc[i_tau,1]]))
                    sig1 = np.pad(kernel,((0,arc_loc[i_tau,0])), mode='constant')[arc_loc[i_tau,0]:]
                    fD1 = np.concatenate((sig1[0:idx_fD_min_neg],sig1[idx_fD_min_pos:]))
                    sig2 = np.pad(kernel,((arc_loc[i_tau,0],0)), mode='constant')[:-arc_loc[i_tau,0]]
                    fD2 = np.concatenate((sig2[0:idx_fD_min_neg],sig2[idx_fD_min_pos:]))
                    init1 = signal0[i_tau]
                    init2 = signal0[-i_tau-1]
                    signal[i_tau],signal[-i_tau-1] = self._fitfunc_for_linefit(data,fD1,fD2,init1,init2)
        else:
            signal = signal0
            kernel = kernel0
        #later: fD_offset = np.zeros(N_im,dtype=int)
        #for i_im in range(N_im):
        #    SecSpec_deconv[arc_loc[i_im,0],arc_loc[i_im,1]] = signal[i_im]
        DynSpec_multiplicator = np.ones((self.N_t,self.N_nu))*np.abs(np.fft.ifft(np.fft.ifftshift(kernel)))[:,na]
        fit = np.empty_like(SecSpec)
        for i_im in range(N_im):
            fit[arc_loc[i_im,0],arc_loc[i_im,1]] = signal[i_im]
        for i_im in range(int(N_im/2)):
            fit[:,arc_loc[i_im,1]] = np.convolve(fit[:,arc_loc[i_im,1]],kernel[::-1],mode='same')
        SecSpec_deconv = np.copy(SecSpec)
        SecSpec_deconv[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])] = fit[:,np.min(arc_loc[:,1]):1+np.max(arc_loc[:,1])]
        SecSpec_deconv[idx_fD_min_neg:1+idx_fD_min_pos,:] = SecSpec[idx_fD_min_neg:1+idx_fD_min_pos,:]
        DynSpec_deconv = np.abs(np.fft.ifft2(np.fft.ifftshift(SecSpec_deconv,axes=(0,1))))
        
        #save the results
        self._add_result(kernel,"kernel")
        self._add_result(signal,"signal")
        self._add_result(DynSpec_deconv,"DynSpec_deconv")
        self._add_result(DynSpec_multiplicator,"DynSpec_multiplicator")
        self._add_result(SecSpec_deconv,"SecSpec_deconv")
        
    def _fitfunc_for_linefit(self,data,fD1,fD2,init1,init2):
        #compute signal of point in tau
        v1_r = np.real(init1)
        v1_i = np.imag(init1)
        v2_r = np.real(init2)
        v2_i = np.imag(init2)
        
        fitto = np.concatenate((np.real(data),np.imag(data)))
        input = np.concatenate((np.real(fD1),np.imag(fD1),np.real(fD2),np.imag(fD2)))
        
        def fitfunc(input,v1_r,v1_i,v2_r,v2_i):
            N4 = len(input)
            N1 = int(N4 /4)
            N2 = int(2*N4 /4)
            N3 = int(3*N4 /4)
            result = (input[0:N1]+1.j*input[N1:N2])*(v1_r+1.j*v1_i) + (input[N2:N3]+1.j*input[N3:N4])*(v2_r+1.j*v2_i)
            return np.concatenate((np.real(result),np.imag(result)))
        
        #print(v1_r,v1_i,v2_r,v2_i)
        #print(input.shape,fitto.shape,v1_r.shape,v1_i.shape,v2_r.shape,v2_i.shape)
        popt, pcov = curve_fit(fitfunc,input,fitto,p0=[v1_r,v1_i,v2_r,v2_i])
        result1 = popt[0]+1.j*popt[1]
        result2 = popt[2]+1.j*popt[3]
        
        return result1,result2
        
    def SecSpec_clean_deconvolution(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SecSpec,f_t,f_nu,SSphase = source.load_result(["secondary_spectrum","doppler","delay","Fourier_phase"])
        
        #load specifications
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        fD_dev = self._add_specification("fD_deviation",0.5e-3)
        tau_max = self._add_specification("tau_max",5.0e-6)
        eta = self._add_specification("curvature",0.0926)
        gain = self._add_specification("gain",0.5)
        thresh = self._add_specification("threshold",pow(10.,3.5))
        niter = self._add_specification("niter",100)
        
        #preparations
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        # - get complex secondary spectrum
        SecSpec = SecSpec[pos1,pos2,:,:]*np.exp(1.j*SSphase[pos1,pos2,:,:])
        # - create point spread function from center brightest pixel
        psf = np.zeros_like(SecSpec)
        center_t, center_nu = np.where(np.abs(SecSpec) == np.amax(np.abs(SecSpec)))
        center_t = center_t[0]
        center_nu = center_nu[0]
        psf[:,center_nu-10:center_nu+11] = SecSpec[:,center_nu-10:center_nu+11]
        psf[center_t,:] = SecSpec[center_t,:]
        psf /= SecSpec[center_t,center_nu]
        # - get indices
        idx_tau_max = self._find_nearest(f_nu,tau_max)
        idx_tau_min = self._find_nearest(f_nu,-tau_max)
        # - create window function around arc
        window = np.zeros(SecSpec.shape,dtype=bool)
        bar = progressbar.ProgressBar(maxval=idx_tau_max-idx_tau_min, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_tau in range(idx_tau_min,idx_tau_max+1):
            bar.update(i_tau-idx_tau_min)
            for i_fD in range(self.N_t):
                fD_arc = np.sqrt(np.abs(f_nu[i_tau])/eta)*np.sign(f_t[i_fD])
                if fD_arc-fD_dev<f_t[i_fD]<fD_arc+fD_dev:
                    window[i_fD,i_tau] = True
        bar.finish()
        
        #create containers
        SecSpec_clean = np.zeros_like(SecSpec)
        
        #perform the computation
        def overlapIndices(a1, a2, shiftx, shifty):
            if shiftx >=0:
                a1xbeg=shiftx
                a2xbeg=0
                a1xend=a1.shape[0]
                a2xend=a1.shape[0]-shiftx
            else:
                a1xbeg=0
                a2xbeg=-shiftx
                a1xend=a1.shape[0]+shiftx
                a2xend=a1.shape[0]

            if shifty >=0:
                a1ybeg=shifty
                a2ybeg=0
                a1yend=a1.shape[1]
                a2yend=a1.shape[1]-shifty
            else:
                a1ybeg=0
                a2ybeg=-shifty
                a1yend=a1.shape[1]+shifty
                a2yend=a1.shape[1]
            return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)
        # - clean
        dirty = SecSpec-psf*SecSpec[center_t,center_nu]
        comps = np.zeros(dirty.shape,dtype=complex)
        comps[center_t,center_nu] = SecSpec[center_t,center_nu]
        res = np.copy(dirty)
        bar = progressbar.ProgressBar(maxval=niter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(niter):
            bar.update(i)
            mx, my = np.unravel_index(np.abs(res[window]).argmax(), res.shape)
            mval = res[mx, my]*gain
            comps[mx, my] += mval
            a1o, a2o = overlapIndices(dirty, psf, mx-int(dirty.shape[0]/2), my-int(dirty.shape[1]/2))
            res[a1o[0]:a1o[1],a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1],a2o[2]:a2o[3]]*mval
            if np.abs(res).max() < thresh:
                break
        bar.finish()
        SecSpec_clean = comps
        
        #save the results
        self._add_result(SecSpec_clean,"SecSpec_clean")
        self._add_result(res,"residuals")
        self._add_result(psf,"psf")
        
################### series of secondary spectra ###################
        
    def FFT_phi_evolution(self):
        #load and check data
        DynSpec = self._load_base_file("DynSpec")
    
        #import c code for discrete Fourier transform
        lib = self._load_c_lib("nut_transform")
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
        
        #load specifications
        N_slice = self._add_specification("N_slice",10)
        nu_fraction = self._add_specification("nu_fraction",0.11)
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        
        #create containers
        #t0 = self.t[0]
        lnu = int(self.N_nu*nu_fraction)
        nu_pos = np.linspace(0,self.N_nu-lnu,num=N_slice,endpoint=True,dtype=int)
        nus = np.zeros(nu_pos.shape,dtype=float)
        midnus = np.zeros(nu_pos.shape,dtype=float)
        for i,index in enumerate(nu_pos):
            nus[i] = self.nu[index]
            midnus[i] = self.nu[index+int(lnu/2)]
        nuref = self.nu_half
        DSpec = np.zeros((N_slice,self.N_t,lnu),dtype=float)
        v_nus = np.zeros((N_slice,lnu),dtype=float)
        for i_slice in range(N_slice):
            # - complex values are not supported by the C FT function
            DSpec[i_slice,:,:] = np.abs(DynSpec[pos1,pos2,:,nu_pos[i_slice]:lnu+nu_pos[i_slice]])
            v_nus[i_slice,:] = self.nu[nu_pos[i_slice]:lnu+nu_pos[i_slice]]
        SSpec = np.zeros(DSpec.shape,dtype=np.complex)
        #f_t = np.linspace(-math.pi/self.delta_t,math.pi/self.delta_t,num=self.N_t,endpoint=False)
        #f_nu = np.linspace(-math.pi/self.delta_nu,math.pi/self.delta_nu,num=lnu,endpoint=False)
        f_t = np.fft.fftshift(np.fft.fftfreq(self.N_t,self.delta_t))*2.*math.pi
        f_nu = np.fft.fftshift(np.fft.fftfreq(lnu,self.delta_nu))*2.*math.pi
        phase = np.zeros(SSpec.shape,dtype=float)
        cphase = np.zeros(SSpec.shape,dtype=float)
    
        #preparations
        ntime = self.N_t
        nfreq = lnu
        r0 = np.fft.fftfreq(ntime)
        delta_r = r0[1] - r0[0]
        src = np.linspace(0, 1, ntime).astype('float64')
        src = np.arange(ntime).astype('float64')
    
        # Common reference freq.
        fscale = np.zeros((N_slice,lnu),dtype=float)
        for i_slice in range(N_slice):
            fscale[i_slice,:] = (self.nu[nu_pos[i_slice]:lnu+nu_pos[i_slice]] / nuref).astype('float64')
    
        #perform the computation
        bar = progressbar.ProgressBar(maxval=N_slice, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_slice in range(N_slice):
            bar.update(i_slice)
            # - declare the empty result array:
            SS = np.empty((ntime, nfreq), dtype=np.complex128)
            # - call the DFT:
            lib.comp_dft_for_secspec(ntime, nfreq, ntime, min(r0), delta_r, fscale[i_slice], src, DSpec[i_slice,:,:].astype('float64'), SS)
            # - flip along time
            SS = SS[::-1]
            # - correct zero point
            SS = np.roll(SS,1,axis=0)
            # - Still need to FFT y axis, should change to pyfftw for memory and
            #   speed improvement
            SS = np.fft.fft(SS, axis=1)
            SS = np.fft.fftshift(SS, axes=1)
            SSpec[i_slice,:,:] = SS
        bar.finish()
        # - compute the phases
        phase = np.angle(SSpec)
        cphase = np.angle(SSpec*np.exp(-1.0j*(f_nu[na,na,:])*(nus[:,na,na]-nuref)))
        SSpec = np.abs(SSpec)
        #cphase = np.angle(SSpec*np.exp(-1.0j*(f_nu[na,na,:]+f_t[na,:,na]*t0/nuref)*nus[:,na,na]))
        
        #save the results
        self._add_result(DSpec,"DSpec")
        self._add_result(v_nus,"nus")
        self._add_result(SSpec,"SSpec")
        self._add_result(phase,"phase")
        self._add_result(cphase,"cphase")
        self._add_result(midnus,"midnus")
        self._add_result(f_t,"doppler")
        self._add_result(f_nu,"delay")
        
    def extracted_SSpec(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"FFT_phi_evolution")
        SSpec,cphase,midnus,f_t,f_nu = source.load_result(["SSpec","cphase","midnus","doppler","delay"])
    
        #preparations
        def fitfunc(in_data,par_slope,par_phase):
            result = (par_slope*in_data+par_phase+math.pi)%(2.*math.pi)-math.pi
            return result
        N_ft = len(f_t)
        N_fnu = len(f_nu)
        #N_nus = len(midnus)
        ESSpec = np.zeros((3,N_ft,N_fnu),dtype=float)
    
        #perform the computation
        ESSpec[0,:,:] = np.median(SSpec,axis=0)
        #indices = np.array(range(N_nus),dtype=float)
        #f_slope = 1./(midnus[1]-midnus[0])
        #f_phase = (self.nu_half-midnus[0])*f_slope
        f_base = (self.nu_half-midnus[0])
        bar = progressbar.ProgressBar(maxval=N_ft, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_ft in xrange(N_ft):
            bar.update(i_ft)
            for i_fnu in xrange(N_fnu):
#                try:
#                    popt, pcov = curve_fit(fitfunc,indices,cphase[:,i_ft,i_fnu],p0=[0.1,0.1],bounds=([-math.pi,0.],[math.pi,2.*math.pi]))
#                except RuntimeError:
#                    popt = [0.,0.]
#                #print(popt[0]/math.pi)
#                ESSpec[1,i_ft,i_fnu] = popt[0]*f_slope
#                ESSpec[2,i_ft,i_fnu] = (popt[1]+f_phase*popt[0]+math.pi)%(2.*math.pi)-math.pi
                fit_phase = np.angle(np.exp(1.0j*(cphase[:,i_ft,i_fnu]-cphase[0,i_ft,i_fnu])))
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(midnus,fit_phase)
                ESSpec[1,i_ft,i_fnu] = slope
                ESSpec[2,i_ft,i_fnu] = np.angle(np.exp(1.0j*(cphase[0,i_ft,i_fnu]+slope*f_base)))
        bar.finish()
        
#        diff = cphase[1:,:,:]-cphase[:-1,:,:]
#        diff = (diff + math.pi) % (2.*math.pi) - math.pi
#        slope = np.median(diff,axis=0)
#        ESSpec[1,:,:] = slope*f_slope
#        base = np.angle(np.sum(np.exp(1.0j*(cphase-slope*indices[:,na,na])),axis=0))
#        ESSpec[2,:,:] = np.angle(np.exp(1.0j*(base+ESSpec[1,:,:]*(self.nu_half-midnus[0]))))
        
        #save the results
        self._add_result(ESSpec,"ESSpec")
        
    def mean_FFT_phase(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"FFT_phi_evolution")
        SSpec,cphase,midnus,f_t,f_nu = source.load_result(["SSpec","cphase","midnus","doppler","delay"])
    
        #preparations
        N_ft = len(f_t)
        N_fnu = len(f_nu)
        N_nus = len(midnus)
        mean_phase = np.empty((N_ft,N_fnu),dtype = float)
        weight_phase = np.empty((N_ft,N_fnu),dtype = float)
    
        #perform the computation
        combination = np.sum(np.exp(1.0j*cphase),axis=0)
        mean_phase = np.angle(combination)
        weight_phase = np.abs(combination)/N_nus
        
        #save the results
        self._add_result(mean_phase,"mean_phase")
        self._add_result(weight_phase,"weight_phase")
        
################### thth diagram ###################
        
    def thth_real(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"nut_SecSpec")
        SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        N_thth = self._add_specification("N_thth",100)
        doppler_max = self._add_specification("doppler_max",60.0e-3) #Hz
        eta = self._add_specification("curvature",0.580) #s^3
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        flag_rNoise = self._add_specification("flag_remove_noise",True)
        SSpec = SSpec[pos1,pos2,:,:]
        
        #create containers
        thetas = np.linspace(-doppler_max/2.,doppler_max/2.,num=N_thth,endpoint=True)
        thth = np.zeros((N_thth,N_thth),dtype=float)
#        # - prepare default of np.argmax
        noise = np.median(SSpec)
#        SSpec[0,:] = noise
#        SSpec[:,0] = noise
        # - get pixel boundaries of thetas
        dtheta = thetas[1]-thetas[0]
        lthetas = thetas - dtheta/2.
        rthetas = thetas + dtheta/2.
        # - get pixel boundaries of SS
        dfd = f_t[1]-f_t[0]
        lfd = f_t - dfd/2.
        rfd = f_t + dfd/2.
        dtau = f_nu[1]-f_nu[0]
        ltau = f_nu - dtau/2.
        rtau = f_nu + dtau/2.
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=N_thth, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th1 in range(N_thth):
            bar.update(i_th1)
            # - compute extreme values of theta bin
            th1_l = np.min([lthetas[i_th1],rthetas[i_th1]])
            th1_u = np.max([lthetas[i_th1],rthetas[i_th1]])
            th1_2l = np.min([lthetas[i_th1]**2,rthetas[i_th1]**2])
            th1_2u = np.max([lthetas[i_th1]**2,rthetas[i_th1]**2])
            for i_th2 in range(N_thth):
                #print(i_th1,i_th2)
                th2_l = np.min([lthetas[i_th2],rthetas[i_th2]])
                th2_u = np.max([lthetas[i_th2],rthetas[i_th2]])
                th2_2l = np.min([lthetas[i_th2]**2,rthetas[i_th2]**2])
                th2_2u = np.max([lthetas[i_th2]**2,rthetas[i_th2]**2])
                # - compute bin boundaries in SS space
                fd_u = th1_u-th2_l
                fd_l = th1_l-th2_u
                tau_u = eta*(th1_2u-th2_2l)
                tau_l = eta*(th1_2l-th2_2u)
                # - determine indices in SS space
                i_fd_l = np.argmax(rfd>fd_l)
                i_fd_u = np.argmax(rfd>fd_u)
                i_tau_l = np.argmax(rtau>tau_l)
                i_tau_u = np.argmax(rtau>tau_u)
                # - index 0 means out of boundaries, except for irrelevant cases
                if not (i_fd_l==0 or i_fd_u==0 or i_tau_l==0 or i_tau_u==0):
                    for i_fd in range(i_fd_l,i_fd_u+1):
                        for i_tau in range(i_tau_l,i_tau_u+1):
                            # - compute fractional area of pixel
                            area = (np.min([fd_u,rfd[i_fd]])-np.max([fd_l,lfd[i_fd]]))*(np.min([tau_u,rtau[i_tau]])-np.max([tau_l,ltau[i_tau]]))/dfd/dtau
                            # - read in area weighted values
                            if not (f_t[i_fd]==0. or f_nu[i_tau]==0.):
                                thth[i_th1,i_th2] += SSpec[i_fd,i_tau]*area
                            else:
                                thth[i_th1,i_th2] += noise
                            if flag_rNoise:
                               thth[i_th1,i_th2] -= noise*area
        bar.finish()
        
        #save the results
        self._add_result(thetas,"thetas")
        self._add_result(thth,"thth")
        
    def FFT_thth(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"thth_real")
        thth,thetas = source.load_result(["thth","thetas"])
        # - derived quantities
        N_th = len(thetas)
        d_th = thetas[1]-thetas[0]
        
        #create containers
        FFT_thth = np.empty(thth.shape,dtype=float)
        f_theta = np.fft.fftshift(np.fft.fftfreq(N_th,d_th))
        
        #perform the computation
        FFT_thth = np.abs(np.fft.fft2(thth))
        
        #save the results
        self._add_result(FFT_thth,"FFT_thth")
        self._add_result(f_theta,"f_theta")
        
    def thth_eigenvector(self):
        #load and check data
        source_thth = self._add_specification("source_thth",["thth_real",None,"thth"])
        source_thetas = self._add_specification("source_thetas",["thth_real",None,"thetas"])
        source = scinter_computation(self.dict_paths,source_thth[0])
        thth, = source.load_result([source_thth[2]])
        source = scinter_computation(self.dict_paths,source_thetas[0])
        thetas, = source.load_result([source_thetas[2]])
        # - derived quantities
        N_th = len(thetas)
        
        #load specifications
        """
        available techniques:
            minimization
        """
        technique = self._add_specification("technique","minimization")
        initial = self._add_specification("initial","random")
        major_diag = self._add_specification("mask_major_diag",3.)
        minor_diag = self._add_specification("mask_minor_diag",1.)
        noise_onset = self._add_specification("mask_noise_onset",7.)
        veff = self._add_specification("veff",305000.) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas) #radians
        
        #create containers
        eigenvector = np.zeros(thetas.shape,dtype=float)
        thth_model = np.zeros(thth.shape,dtype=float)
        residuals = np.empty(thth.shape,dtype=float)
        weights = np.ones(thth.shape,dtype=float)
        v0 = np.zeros(thetas.shape,dtype=float)
        
        #preparations
        # for reproducible results
        np.random.seed (1234)
        def  func_and_grad_for_min (vector, weights, thth):
            """
            Compute function and gradient.
            Here we assume that the problem has been symmetrised.
            """

            # full residuals:
            resid= vector[:,None]*vector[None,:]-thth

            f= (weights*resid**2).sum ()

            grad= 4* (weights*vector[None,:]*resid).sum (axis=1)
            
            return f,grad
        def  hesse_for_min (vector, weights, thth):
            """
            Hessian. Not needed for all methods.
            """
            temp= 4*(weights*vector[None,:]**2).sum (axis=1)
            hess= np.diag (temp) + 4*weights*(2*vector[:,None]*vector[None,:]-thth)
            return hess
        
        #perform the computation
        # - scale thetas to physical units
        thetas = (-self.LightSpeed/self.nu_half/veff*thetas)/theta_scale
        # - compute the mask
        weights = (np.abs (thetas[:,na]-thetas[na,:])>major_diag)*(
            np.abs (thetas[:,na]+thetas[na,:])>minor_diag)*(
            (np.abs(thetas[:,na]+np.zeros(N_th)[na,:])<noise_onset) | (np.abs(np.zeros(N_th)[:,na]+thetas[na,:])<noise_onset))
        # - set initial values
        if initial=="random":
            v0 = np.random.randn(N_th)
        elif "median":
            for i_th in range(N_th):
               v0[i_th] = np.nanmedian(np.nanmedian(thth[:,i_th,na]/thth[:,:],axis=0))
        # - compute the eigenvector
        if technique=="minimization":
            # - choose a method
            """
            available methods:
                Nelder-Mead   very slow, no derivatives
                Powell        no derivatives
                CG            no Hessian
                BFGS          no Hessian
                Newton-CG
                L-BFGS-B
                TNC           no Hessian
                COBYLA        no derivatives
                SLSQP         no Hessian
                trust-constr
                dogleg
                trust-ncg
                trust-exact
                trust-krylov
            Not all can deal with bounds.
            """
            method = self._add_specification("method","tnc")
            # - we need the matrices to be symmetric
            if not (thth==thth.T).all():
                self._warning("Making non-symmetric thth matrix symmetric by M=(M+M.T)/2.")
                thth = .5*(thth+thth.T)
            if not (weights==weights.T).all():
                self._warning("Making non-symmetric weights matrix symmetric by M=(M+M.T)/2.")
                weights = .5*(weights+weights.T)
            # - brightness eigenvectors are positive
            bounds= scipy.optimize.Bounds (0, np.inf)
            # - perform the minimization
            res = scipy.optimize.minimize(
                func_and_grad_for_min, v0, args= (weights,thth),
                method=method, jac=True, hess=hesse_for_min, bounds=bounds,
                options=dict(disp=True) )
            eigenvector = res.x
            thth_model = eigenvector[:,na]*eigenvector[na,:]
            residuals = thth-thth_model
        
        #save the results
        self._add_result(eigenvector,"eigenvector")
        self._add_result(thth_model,"thth_model")
        self._add_result(residuals,"residuals")
        self._add_result(weights,"weights")
        
    def linSS_real(self):
        #load specifications
        N_delay = self._add_specification("N_delay",1000)
        doppler_max = self._add_specification("doppler_max",60.0e-3) #Hz
        ddd_max = self._add_specification("delay_doppler_ratio_max",1./20.) #s^2
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        source_SecSpec = self._add_specification("source_SecSpec","nut_SecSpec")
        flag_rNoise = self._add_specification("flag_remove_noise",True)
    
        #load and check data
        source = scinter_computation(self.dict_paths,source_SecSpec)
        SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #create containers
        # - cut doppler and secondary spectrum
        i_fd_min = np.argmax(f_t>-doppler_max)
        i_fd_max = np.argmax(f_t>doppler_max)
        SSpec = SSpec[pos1,pos2,i_fd_min:i_fd_max,:]
        f_t = f_t[i_fd_min:i_fd_max]
        N_doppler = len(f_t)
#        # - prepare default of np.argmax
        noise = np.median(SSpec)
#        SSpec[:,0] = noise
        ddd = np.linspace(-ddd_max,ddd_max,num=N_delay,endpoint=True)
        linSS = np.zeros((N_doppler,N_delay),dtype=float)
        # - get pixel boundaries of thetas
        dddd = ddd[1]-ddd[0]
        lddd = ddd - dddd/2.
        rddd = ddd + dddd/2.
        # - get pixel boundaries of SS
        dtau = f_nu[1]-f_nu[0]
        ltau = f_nu - dtau/2.
        rtau = f_nu + dtau/2.
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=N_delay, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_ddd in range(N_delay):
            bar.update(i_ddd)
            for i_fd in range(N_doppler):
                if not f_t[i_fd]==0.:
                    # - compute bin boundaries in SS space
                    tau_u = np.max([lddd[i_ddd]*f_t[i_fd],rddd[i_ddd]*f_t[i_fd]])
                    tau_l = np.min([lddd[i_ddd]*f_t[i_fd],rddd[i_ddd]*f_t[i_fd]])
                    # - determine indices in SS space
                    i_tau_l = np.argmax(rtau>tau_l)
                    i_tau_u = np.argmax(rtau>tau_u)
                    # - index 0 means out of boundaries, except for irrelevant cases
                    if not (i_tau_l==0 or i_tau_u==0):
                        for i_tau in range(i_tau_l,i_tau_u+1):
                            # - compute fractional width of pixel
                            width = (np.min([tau_u,rtau[i_tau]])-np.max([tau_l,ltau[i_tau]]))/dtau
                            # - read in area weighted values
                            if not f_nu[i_tau]==0.:
                                linSS[i_fd,i_ddd] += SSpec[i_fd,i_tau]*width
                                if flag_rNoise:
                                    linSS[i_fd,i_ddd] -= noise*width
                            else:
                                linSS[i_fd,i_ddd] += noise
        bar.finish()
        
        #save the results
        self._add_result(f_t,"doppler")
        self._add_result(ddd,"ddd")
        self._add_result(linSS,"linSS")
        
    def FFT_linSS(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"linSS_real")
        linSS,doppler,ddd = source.load_result(["linSS","doppler","ddd"])
        # - derived quantities
#        N_dop = len(doppler)
        d_dop = doppler[1]-doppler[0]
#        N_ddd = len(ddd)
        d_ddd = ddd[1]-ddd[0]
        
        #load specifications
        disk_radius = self._add_specification("N_disk_radius",0)
        dop_min = self._add_specification("doppler_min",float(doppler[0]))
        dop_max = self._add_specification("doppler_max",float(doppler[-1]))
        ddd_min = self._add_specification("d-d-ratio_min",float(ddd[0]))
        ddd_max = self._add_specification("d-d-ratio_max",float(ddd[-1]))
        flag_from_log10 = self._add_specification("flag_from_log10",True)
        log10_min = self._add_specification("log10_min",0.)
        
        #crop data
        dop_mask = np.where((doppler>dop_min) & (doppler<dop_max))[0]
        ddd_mask = np.where((ddd>ddd_min) & (ddd<ddd_max))[0]
        linSS = linSS[np.ix_(dop_mask,ddd_mask)]
        doppler = doppler[dop_mask]
        ddd = ddd[ddd_mask]
        N_ddd = len(ddd)
        N_dop = len(doppler)
        #apply log10 if requested
        if flag_from_log10: 
            linSS_log10 = np.log10(linSS)
            linSS_log10[linSS==0.] = .5*log10_min
            linSS_log10[linSS_log10<log10_min] = log10_min
            linSS_log10 -= log10_min
            linSS = linSS_log10
            
        
        #create containers
        FFT_linSS = np.empty(linSS.shape,dtype=float)
        f_dop = np.fft.fftshift(np.fft.fftfreq(N_dop,d_dop))
        f_ddd = np.fft.fftshift(np.fft.fftfreq(N_ddd,d_ddd))
        
        if not disk_radius==0:
            # - apply logarithmic scale
            min_nonzero = np.min(linSS[np.nonzero(linSS)])
            linSS[linSS == 0] = min_nonzero
            linSS = np.log10(linSS)
            # - normalize spectrum
            linSS = cv2.normalize(linSS, dst=linSS, alpha=0.0, beta=0.999, norm_type=cv2.NORM_MINMAX)
            selem = morphology.disk(disk_radius)
            linSS = skimage.filters.rank.equalize(linSS,selem=selem)
        
        #perform the computation
        FFT_linSS = np.abs(np.fft.fft2(linSS))
        FFT_linSS = np.fft.fftshift(FFT_linSS,axes=(0,1))
        
        #save the results
        self._add_result(FFT_linSS,"FFT_linSS")
        self._add_result(f_dop,"f_dop")
        self._add_result(f_ddd,"f_ddd")
        
    def FFT_linSS_sum(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"FFT_linSS")
        FFT_linSS,f_dop,f_ddd = source.load_result(["FFT_linSS","f_dop","f_ddd"])
        # - derived quantities
        dddd = f_ddd[1]-f_ddd[0]
        
        #load specifications
#        curv_min = self._add_specification("curv_min",float(f_dop[-1]/f_ddd[-1]))
#        curv_max = self._add_specification("curv_max",float(f_dop[-1]/dddd))
        f_dop_max = self._add_specification("f_dop_max",float(f_dop[-1]))
        f_ddd_max = self._add_specification("f_ddd_max",float(f_ddd[-1]))
        flag_logarithmic = self._add_specification("flag_logarithmic",True)
        fit_curv_min = self._add_specification("fit_curv_min",0.)
        fit_curv_max = self._add_specification("fit_curv_max",1.)
        
        #create containers
        # - cut one quarter of FFT of FFT of real quantity
        i_dop0 = np.argmax(f_dop==0.)
        i_ddd0 = np.argmax(f_ddd==0.)
        #FFT_linSS = FFT_linSS[i_dop0:,i_ddd0+1:]
        FFT_linSS = FFT_linSS[i_dop0:,i_ddd0:]
        f_dop = f_dop[i_dop0:]
        #f_ddd = f_ddd[i_ddd0+1:]
        f_ddd = f_ddd[i_ddd0:]
        N_dop = len(f_dop)
        N_ddd = len(f_ddd)
#        # - cut f_dop such that minimal requested curvature corresponds to upper right corner
#        min_avail_curv = f_dop/f_ddd[-1]
#        i_dop_u = np.argmax(min_avail_curv>curv_min)
#        print(i_dop_u,min_avail_curv[0],min_avail_curv[-1])
#        if not i_dop_u==0:
#            f_dop = f_dop[:i_dop_u+1]
#            FFT_linSS = FFT_linSS[:i_dop_u+1,:]
#        # - determine f_ddd index corresponding to highest requested curvature
#        avail_curv = f_dop[-1]/f_ddd
#        i_ddd_l = np.argmax(avail_curv<curv_max)
#        # - get number of remaining indices
#        N_ddd = len(f_ddd)
#        # - compute curvatures
#        curv = f_dop[-1]/f_ddd[i_ddd_l:]
#        N_curv = len(curv)
        # - cut requested maximum values
        i_dop_u = np.argmax(f_dop>f_dop_max)
        if i_dop_u==0:
            i_dop_u = N_dop
        i_ddd_u = np.argmax(f_ddd>f_ddd_max)
        if i_ddd_u==0:
            i_ddd_u = N_ddd
        f_dop = f_dop[:i_dop_u]
        f_ddd = f_ddd[:i_ddd_u]
        FFT_linSS = FFT_linSS[:i_dop_u,:i_ddd_u]
        # - compute curvatures
        curv = f_dop[-1]/f_ddd[1:]
        N_curv = len(curv)
        N_dop = len(f_dop)
        N_ddd = len(f_ddd)
        # - prepare counter
        curv_sum = np.zeros(curv.shape,dtype=float)
        if flag_logarithmic:
            # - safely remove zeros if there are any
            min_nonzero = np.min(FFT_linSS[np.nonzero(FFT_linSS)])
            FFT_linSS[FFT_linSS == 0] = min_nonzero
            # - apply logarithmic scale
            FFT_linSS = np.log10(FFT_linSS)
        #compute the curvature mask for the fitting
        curv_mask = np.where((curv>fit_curv_min) & (curv<fit_curv_max))[0]
        xdata = curv[curv_mask]
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=N_curv, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_curv,v_curv in enumerate(curv):
            bar.update(i_curv)
            for i_dop,v_dop in enumerate(f_dop):
                i_ddd = int((v_dop/v_curv-f_ddd[0])/(f_ddd[-1]-f_ddd[0])*N_ddd-0.5)
                curv_sum[i_curv] += FFT_linSS[i_dop,i_ddd]
        bar.finish()
        # - fit peak
        fit_result = None
        fit_error = None
        def parabola(data,x,y,curv):
            #print(x,y,curv)
            return curv*(data-x)**2+y
        try:
            ydata = curv_sum[curv_mask]
            popt, pcov = curve_fit(parabola,xdata,ydata,p0=[xdata[0],ydata[0],(np.min(ydata)-np.max(ydata))/4./(xdata[-1]-xdata[0])**2],bounds=([np.min(xdata),np.min(ydata),-np.inf],[np.max(xdata),np.max(ydata),0.]))
            perr = np.sqrt(np.diag(pcov))
            fit_result = popt
            fit_error = perr
            print("Curvature fit peak: {0} +- {1}".format(fit_result[0],fit_error[0]))
        except:
            print("Curvature could not be estimated by fit!")
        
        #save the results
        self._add_result(curv,"curv")
        self._add_result(curv_sum,"curv_sum")
        self._add_result(xdata,"xdata")
        self._add_result(fit_result,"fit_result")
        self._add_result(fit_error,"fit_error")
        
    def square_thth(self):
        #load specifications
        veff = self._add_specification("veff",305000.) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas) #radians
        ext_source_lines = self._add_specification("external_source_lines","/mnt/c/Ubuntu/MPIfR/scint_thth/lines.npy") #find a way to create secs file even when throwing error
    
        #load and check data
        source = scinter_computation(self.dict_paths,"thth_real")
        thth,theta_data = source.load_result(["thth","thetas"])
        lines = np.load(ext_source_lines)
        # - derived quantities
        N_th = len(theta_data)
        N_lines = len(lines)
        
        #create containers
        thth_square = np.zeros(thth.shape,dtype=float)
        thetas = np.empty(N_lines,dtype=float)
        gammas = np.empty(N_lines,dtype=float)
        theta = np.empty(N_th,dtype=float)
        gamma = np.empty(N_th,dtype=float)
        
        #preparations
        # - scale thetas to physical units
        theta = (-self.LightSpeed/self.nu_half/veff*theta_data)/theta_scale
        
        #perform the computation
        # - get sorted arrays of images
        for i_line in range(N_lines):
            thetas[i_line] = lines[i_line,0]
            gammas[i_line] = lines[i_line,1]
            if gammas[i_line] < 0.:
                thetas[i_line] *= -1.
                gammas[i_line] *= -1.
        indices = np.argsort(thetas)
        thetas = thetas[indices]
        gammas = gammas[indices]
        # - interpolate gamma(theta)
        ip = interpolate.interp1d(thetas,gammas,kind='cubic',fill_value=1.,bounds_error=False)
        for i_th in range(N_th):
            gamma[i_th] = ip(theta[i_th])
        # - create thth coordinates
        th1 = 0.5*((theta[:,na]**2-theta[na,:]**2)/(gamma[:,na]*theta[:,na]-gamma[na,:]*theta[na,:])+gamma[:,na]*theta[:,na]-gamma[na,:]*theta[na,:])
        th2 = 0.5*((theta[:,na]**2-theta[na,:]**2)/(gamma[:,na]*theta[:,na]-gamma[na,:]*theta[na,:])-gamma[:,na]*theta[:,na]+gamma[na,:]*theta[na,:])
        # - evaluate pixels in thth along these coordinates
        minth = np.min(theta)
        maxth = np.max(theta)
        spanth = maxth-minth
        min_nonzero = np.min(thth[np.nonzero(thth)])
        for i_th1 in range(N_th):
            for i_th2 in range(N_th):
                try:
                    i1 = N_th-int((th1[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    i2 = N_th-int((th2[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    thth_square[i_th1,i_th2] = thth[i1,i2]
                except:
                    thth_square[i_th1,i_th2] = min_nonzero
        
        #save the results
        self._add_result(thth_square,"thth_square")
        self._add_result(gamma,"gamma")
        self._add_result(th1,"th1")
        self._add_result(th2,"th2")
        
    def import_lines(self):
        #load specifications
        veff = self._add_specification("veff",305000.) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas) #radians
        ext_source_lines = self._add_specification("external_source_lines","/mnt/c/Ubuntu/MPIfR/scint_thth/lines.npy") #find a way to create secs file even when throwing error
        
        #load and check data
        lines = np.load(ext_source_lines)
        source = scinter_computation(self.dict_paths,"thth_real")
        theta, = source.load_result(["thetas"])
        # - derived quantities
        N_th = len(theta)
        N_lines = len(lines)
        thetas = lines[:,0]
        gammas = lines[:,1]
        
        #create containers
        th1 = np.empty((N_lines,N_th),dtype=float)
        th2 = np.empty((N_lines,N_th),dtype=float)
        
        #preparations
        # - rescale thetas to internal units
        thetas /= -self.LightSpeed/self.nu_half/veff/theta_scale
        
        #perform the computation
        th1 = 0.5*((theta[na,:]**2-thetas[:,na]**2)/(theta[na,:]-gammas[:,na]*thetas[:,na])+theta[na,:]-gammas[:,na]*thetas[:,na])
        th2 = 0.5*((theta[na,:]**2-thetas[:,na]**2)/(theta[na,:]-gammas[:,na]*thetas[:,na])-theta[na,:]+gammas[:,na]*thetas[:,na])
        
        #save the results
        self._add_result(thetas,"thetas")
        self._add_result(gammas,"gammas")
        self._add_result(th1,"th1")
        self._add_result(th2,"th2")
        
    def desquare_thth(self):
        #load specifications
        veff = self._add_specification("veff",305000.) #m/s
        theta_scale = self._add_specification("theta_scale",self.mas) #radians
        range_of_mean = self._add_specification("range_of_mean",1)
    
        #load and check data
        source = scinter_computation(self.dict_paths,"thth_real")
        thetas, = source.load_result(["thetas"])
        source = scinter_computation(self.dict_paths,"square_thth")
        gamma,th1,th2 = source.load_result(["gamma","th1","th2"])
        source = scinter_computation(self.dict_paths,"thth_eigenvector")
        mu,weights = source.load_result(["eigenvector","weights"])
        # - derived quantities
        N_th = len(thetas)
        
        #create containers
        thth_model = np.zeros((N_th,N_th),dtype=float)
        
        #preparations
        # - scale thetas to physical units
        thetas = (-self.LightSpeed/self.nu_half/veff*thetas)/theta_scale
        
        #perform the computation
        # minth = np.min(thetas)
        # maxth = np.max(thetas)
        # spanth = maxth-minth
        # index1 = np.full((N_th,N_th),np.nan,dtype=int)
        # index2 = np.full((N_th,N_th),np.nan,dtype=int)
        # for i_th1 in range(N_th):
            # for i_th2 in range(N_th):
                # try:
                    # i1 = N_th-int((th1[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    # i2 = N_th-int((th2[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    # #thth_model[i1,i2] = mu[i_th1]*mu[i_th2]
                    # result = mu[i_th1]*mu[i_th2]
                    # previous = thth_model[i1,i2]
                    # if result>previous:
                        # thth_model[i1,i2] = result
                        # index1[i1,i2] = i_th1
                        # index2[i1,i2] = i_th2
                # except:
                    # pass
        
        #old solution producing gaps
        # - translate the pixels back using the same transformation
        minth = np.min(thetas)
        maxth = np.max(thetas)
        spanth = maxth-minth
        for i_th1 in range(N_th):
            for i_th2 in range(N_th):
                try:
                    i1 = N_th-int((th1[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    i2 = N_th-int((th2[i_th1,i_th2]-minth)/spanth*N_th+0.5)
                    #thth_model[i1,i2] = mu[i_th1]*mu[i_th2]
                    result = mu[i_th1]*mu[i_th2]
                    previous = thth_model[i1,i2]
                    if result>previous:
                        thth_model[i1,i2] = result
                except:
                    pass
        # - fill in empty spaces by mean of surrounding pixels
        thth_model[thth_model==0.] = np.nan
        # replacement = np.nanmin(thth_model)
        # for i_th1 in range(N_th):
            # for i_th2 in range(N_th):
                # if np.isnan(thth_model[i_th1,i_th2]):
                    # #thth_model[i_th1,i_th2] = np.nanmean(thth_model[i_th1-range_of_mean:i_th1+range_of_mean+1,i_th2-range_of_mean:i_th2+range_of_mean+1])
                    # thth_model[i_th1,i_th2] = replacement
                    # print(i_th1,i_th2)
                # else:
                    # replacement = thth_model[i_th1,i_th2]
        
        #save the results
        self._add_result(thth_model,"thth_model")
        
    def render_screen(self):
        #load specifications
        beta = self._add_specification("beta",-28.6) #degrees
    
        #load and check data
        source = scinter_computation(self.dict_paths,"thth_real")
        theta, = source.load_result(["thetas"])
        source = scinter_computation(self.dict_paths,"square_thth")
        gamma, = source.load_result(["gamma"])
        source = scinter_computation(self.dict_paths,"thth_eigenvector")
        mu, = source.load_result(["eigenvector"])
        # - derived quantities
        N_th = len(theta)
        
        #create containers
        screen = np.empty((2*N_th,3),dtype=float)
        
        #preparations
        
        #perform the computation
        for i_im in range(N_th):
            #compute both possible alpha values and check if they are in range
            if -1.<=gamma[i_im]*np.cos(np.deg2rad(beta))<=1.:
                alpha1 = beta - np.rad2deg(np.arccos(gamma[i_im]*np.cos(np.deg2rad(beta))))
                alpha2 = beta + np.rad2deg(np.arccos(gamma[i_im]*np.cos(np.deg2rad(beta))))
                if alpha1 > 180.:
                    alpha1 -= 360.
                elif alpha1 <= 180.:
                    alpha1 += 360.
                if alpha2 > 180.:
                    alpha2 -= 360.
                elif alpha2 <= 180.:
                    alpha2 += 360.
                screen[i_im,0] = theta[i_im]*np.cos(np.deg2rad(alpha1))
                screen[i_im,1] = theta[i_im]*np.sin(np.deg2rad(alpha1))
                screen[N_th+i_im,0] = theta[i_im]*np.cos(np.deg2rad(alpha2))
                screen[N_th+i_im,1] = theta[i_im]*np.sin(np.deg2rad(alpha2))
            else:
                screen[i_im,0] = np.nan
                screen[i_im,1] = np.nan
                screen[N_th+i_im,0] = np.nan
                screen[N_th+i_im,1] = np.nan
            screen[i_im,2] = mu[i_im]
            screen[N_th+i_im,2] = mu[i_im]
        
        #save the results
        self._add_result(screen,"screen")
        
################### fD-th diagram ###################

    def thfD_diagram(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"secondary_spectrum")
        SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        
        #load specifications
        N_th = self._add_specification("N_th",100)
        delay_max = self._add_specification("delay_max",6.0e-6) #s
        doppler_range = self._add_specification("doppler_range",2.0e-3) #Hz
        D_eff = self._add_specification("D_eff",137.9) #pc
        v_eff = self._add_specification("v_eff",61.58) #km/s
        #th_offset = self._add_specification("th_offset",0.0) #mas
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        flag_rNoise = self._add_specification("flag_remove_noise",True)
        
        #preparations
        D_eff *= self.pc
        v_eff *= 1000.
        delay_max *= 2.*math.pi
        doppler_range *= 2.*math.pi
        SSpec = np.abs(SSpec[pos1,pos2,:,:])
        noise = np.median(SSpec)
        dfd = f_t[1]-f_t[0]
        N_fd = int(np.ceil(doppler_range/dfd/2.))
        theta_max = np.sqrt(2.*self.LightSpeed*delay_max/D_eff/(2.*math.pi))
        thetas = np.linspace(-theta_max,theta_max,num=N_th,endpoint=True)
        dth = thetas[1]-thetas[0]
        thetas_u = thetas+dth/2.
        thetas_l = thetas-dth/2.
        taus = -np.sign(thetas)*D_eff/(2.*self.LightSpeed)*thetas**2*2.*math.pi
        taus_u = -np.sign(thetas_l)*D_eff/(2.*self.LightSpeed)*thetas_l**2*2.*math.pi
        taus_l = -np.sign(thetas_u)*D_eff/(2.*self.LightSpeed)*thetas_u**2*2.*math.pi
        fD_c = -self.nu_half/self.LightSpeed*np.abs(v_eff*thetas)*2.*math.pi
        for i,tau in enumerate(taus):
            if tau < 0.:
                taus[i] *= -1.
                taus_u[i] += taus[i]-tau
                taus_l[i] += taus[i]-tau
                fD_c[i] *= -1.
        fD_u = fD_c + N_fd*dfd
        fD_l = fD_c - N_fd*dfd
        # - get pixel boundaries of SS
        dfd = f_t[1]-f_t[0]
        lfd = f_t - dfd/2.
        rfd = f_t + dfd/2.
        dtau = f_nu[1]-f_nu[0]
        ltau = f_nu - dtau/2.
        rtau = f_nu + dtau/2.
        
        #create containers
        thfD = np.zeros((N_th,2*N_fd+1),dtype=float)
        fDs = np.linspace(-N_fd*dfd,N_fd*dfd,num=2*N_fd+1,dtype=float)
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in range(N_th):
            bar.update(i_th)
            # - determine indices of boundary pixels
            i_tau_l = np.argmax(rtau>taus_l[i_th])
            i_tau_u = np.argmax(rtau>taus_u[i_th])
            # - determine indices of f_D range
            i_fD_l = np.argmax(rfd>fD_l[i_th])
            i_fD_u = np.argmax(rfd>fD_u[i_th])
            # - sum pixels
            for i_tau in range(i_tau_l,i_tau_u+1):
                length = (np.min([taus_u[i_th],rtau[i_tau]])-np.max([taus_l[i_th],ltau[i_tau]]))/dtau
                thfD[i_th,:] += SSpec[i_fD_l:i_fD_u+1,i_tau]*length
                if flag_rNoise:
                    thfD[i_th,:] -= noise*length
        bar.finish()
        
        #save the results
        self._add_result(thetas,"thetas")
        self._add_result(fDs,"fDs")
        self._add_result(thfD,"thfD")
    
    def thfD_diagram_direct(self):
        #load specifications
        DS_source = self._add_specification("DS_source",None) #["DynSpec_connected", None, "DynSpec_connected"]
        
        #load and check data
        if DS_source==None:
            DynSpec = self._load_base_file("DynSpec")
        else:
            self._warning("Using {0} instead of unprocessed dynamic spectrum.".format(DS_source))
            source = scinter_computation(self.dict_paths,DS_source[0])
            DynSpec, = source.load_result([DS_source[2]])
        
        #load specifications
        N_th = self._add_specification("N_th",400)
        th_range = self._add_specification("th_range",[-5.,5.]) #mas
        N_fD = self._add_specification("N_fD",11)
        doppler_range = self._add_specification("doppler_range",2.0e-3) #Hz
        D_eff = self._add_specification("D_eff",137.9) #pc
        MJD_start = self._add_specification("MJD_start",58947.88607757718) #MJD
        veff_par_file = self._add_specification("veff_par_file","/mnt/c/Ubuntu/MPIfR/scint_evolution/veff_par.npy")
        veff_MJD_file = self._add_specification("veff_MJD_file","/mnt/c/Ubuntu/MPIfR/scint_evolution/veff_MJD.npy")
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        
        #preparations
        D_eff *= self.pc
        v_eff = np.full(self.N_t,63953.92964502) #placeholder
        table_veff_par = np.load(veff_par_file)
        table_veff_MJD = np.load(veff_MJD_file)
        table_veff_MJD = (table_veff_MJD-MJD_start)*24.*3600.
        ip = interpolate.interp1d(table_veff_MJD,table_veff_par,kind='cubic',fill_value="extrapolate")
        shift = np.zeros(self.N_t,dtype=float)
        for i_t in range(self.N_t):
            v_eff[i_t] = ip(self.t[i_t])
            shift[i_t], err = scipy.integrate.quad(ip, self.t[0], self.t[i_t])
        shift /= D_eff
        #print(v_eff,shift/self.mas)
        doppler_range *= 2.*math.pi
        
        #create containers
        thfD = np.zeros((N_th,N_fD),dtype=float)
        thetas = np.linspace(th_range[0],th_range[1],num=N_th,dtype=float,endpoint=True)*self.mas
        fDs = np.linspace(-doppler_range/2.,doppler_range/2.,num=N_fD,dtype=float,endpoint=True)
        
        # #perform the computation
        # factor_fD = -self.nu_half/self.LightSpeed*v_eff*self.t*2.*math.pi
        # factor_tau = D_eff/2./self.LightSpeed*self.nu*2.*math.pi
        # shift_term = factor_fD*shift
        # DynSpec = DynSpec[pos1,pos2,:,:]*np.exp(1.j*shift_term[:,na])
        # fD_term = np.exp(-1.j*(fDs[:,na]*self.t[na,:]))
        # # - main computation
        # bar = progressbar.ProgressBar(maxval=N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        # for i_th in range(N_th):
            # bar.update(i_th)
            # DS = DynSpec*np.exp(-1.j*(factor_fD[:,na]*thetas[i_th] + factor_tau[na,:]*(thetas[i_th]-shift[:,na])**2))
            # try:
                # thfD[i_th,:] = np.abs(np.sum(DS*fD_term[:,:,na],axis=(1,2)))
            # except:
                # for i_fD in range(N_fD):
                    # thfD[i_th,i_fD] = np.abs(np.sum(DS*fD_term[i_fD,:,na]))
        # bar.finish()
        
        #perform the computation
        interim = np.zeros((N_th,self.N_t),dtype=complex)
        DS = DynSpec[pos1,pos2,:,:]
        pre_tau = -1.j*D_eff/2./self.LightSpeed*self.nu*2.*math.pi
        term_quad = (thetas[:,na]-shift[na,:])**2
        bar = progressbar.ProgressBar(maxval=N_th+N_fD, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in range(N_th):
            bar.update(i_th)
            interim[i_th,:] = np.sum(DS*np.exp(pre_tau[na,:]*term_quad[i_th,:,na]),axis=1)
        pre_fD = -self.nu_half/self.LightSpeed*v_eff*self.t*2.*math.pi
        term_fD = np.exp(-1.j*pre_fD[na,:]*(thetas[:,na]-shift[na,:]))
        interim = interim*term_fD
        term_dfd = np.exp(-1.j*fDs[:,na]*self.t[na,:])
        for i_fD in range(N_fD):
            bar.update(N_th+i_fD)
            thfD[:,i_fD] = np.abs(np.sum(interim[:,:]*term_dfd[na,i_fD,:],axis=1))
        bar.finish()
        
        #save the results
        self._add_result(thetas,"thetas")
        self._add_result(fDs,"fDs")
        self._add_result(thfD,"thfD")
        
    def thfD_line(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"thfD_diagram")
        thetas,fDs,thfD = source.load_result(["thetas","fDs","thfD"])
        
        #perform the computation
        profile = np.median(thfD,axis=0)
        profile -= np.min(profile)
        profile /= np.sum(profile)
        line = np.sum(thfD*profile[na,:],axis=1)
        
        #save the results
        self._add_result(profile,"profile")
        self._add_result(line,"line")
        
        
################### deconvolution ###################

    def Gini_deconvolution(self):
        #load specifications
        eta = self._add_specification("curvature",0.08825745414070099)
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        delay_max = self._add_specification("delay_max",4.0e-6) #s
        delay_min = self._add_specification("delay_min",0.3e-6) #s
        doppler_range = self._add_specification("doppler_range",1.0e-3) #Hz
        method = self._add_specification("method","Powell")
        maxiter = self._add_specification("maxiter",3)
        t_sampling = self._add_specification("t_sampling",1)
        nu_sampling = self._add_specification("nu_sampling",4)
        init = self._add_specification("init","/mnt/c/Ubuntu/MPIfR/scinter_data/data/B1508_20200408/computations/Gini_deconvolution/multiplicator.npy")
        print(method)
        
        #load and check data
        DSpec = self._load_base_file("DynSpec")
        #source = scinter_computation(self.dict_paths,"secondary_spectrum")
        #SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        
        #preparations
        dynspec = np.abs(DSpec[pos1,pos2,:,:])
        # - downsampling
        dynspec = block_reduce(dynspec, block_size=(t_sampling,nu_sampling), func=np.mean)
        coordinates = np.array([self.t,self.t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=self.t[-1])
        t = coordinates[0,:]
        coordinates = np.array([self.nu,self.nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=self.nu[-1])
        nu = coordinates[0,:]
        N_t = len(t)
        N_nu = len(nu)
        dt = t[1]-t[0]
        dnu = nu[1]-nu[0]
        A_DynSpec = np.fft.fft2(dynspec)
        secspec = np.abs(np.fft.fftshift(A_DynSpec,axes=(0,1)))
        f_t = np.fft.fftshift(np.fft.fftfreq(N_t,dt))*2.*math.pi
        f_nu = np.fft.fftshift(np.fft.fftfreq(N_nu,dnu))*2.*math.pi
        #secspec = SSpec[pos1,pos2,:,:]
        delay_min *= 2.*math.pi
        delay_max *= 2.*math.pi
        doppler_range *= 2.*math.pi
        dft = f_t[1]-f_t[0]
        dfnu = f_nu[1]-f_nu[0]
        lft = f_t - dft/2.
        # - compute relevant indices
        indices_tau = []
        for i_fnu,fnu in enumerate(f_nu):
            if (delay_min<fnu<delay_max) or (delay_min<-fnu<delay_max):
                indices_tau.append(i_fnu)
        indices_tau = np.array(indices_tau,dtype=int)
        N_ft_half = int(doppler_range/(2.*dft))
        N_ft = 2*N_ft_half+1
        N_fnu = len(indices_tau)
        indices_ft = np.empty((N_ft,N_fnu),dtype=int)
        for i_tau,i_fnu in enumerate(indices_tau):
            fnu = f_nu[i_fnu]
            v_ft = fnu/eta
            i_ft_mid = np.where(v_ft>lft)[0][-1]
            indices_ft[:,i_tau] = range(i_ft_mid-N_ft_half,i_ft_mid+N_ft_half+1)
        # def gini(x):
            # # (Warning: This is a concise implementation, but it is O(n**2)
            # # in time and memory, where n = len(x).  *Don't* pass in huge
            # # samples!)
            # # Mean absolute difference
            # mad = np.abs(np.subtract.outer(x, x)).mean()
            # # Relative mean absolute difference
            # rmad = mad/np.mean(x)
            # # Gini coefficient
            # g = 0.5 * rmad
            # return g
        # - for reproducible results
        np.random.seed (1234)
        # function to minimize
        def func_for_min(vector, dynspec,indices_tau,indices_ft):
            #start = time.time()
            N_ft,N_fnu = indices_ft.shape
            # - compute the secondary spectrum
            DynSpec = dynspec/vector[:,na]
            #t_dec = time.time()-start
            A_DynSpec = np.fft.fft2(DynSpec)
            SecSpec = np.abs(np.fft.fftshift(A_DynSpec,axes=(0,1)))
            #t_SS = time.time()-t_dec-start
            
            # - compute gini measure
            measure = np.zeros(N_fnu,dtype=float)
            for i_tau in range(N_fnu):
                data = SecSpec[indices_ft[:,i_tau],indices_tau[i_tau]]
                # - compute Gini index
                mad = np.abs(np.subtract.outer(data, data)).mean()
                rmad = mad/np.mean(data)
                gini = 0.5 * rmad
                measure[i_tau] = 1.-gini
            #t_gini = time.time()-t_SS-t_dec-start

            # full result:
            f = np.sum(measure)
            print(f) #,vector[0],vector[-1],t_dec,t_SS,t_gini
            
            return f
        
        #create containers
        SS_deconv = np.zeros(secspec.shape,dtype=float)
        DS_deconv = np.zeros(dynspec.shape,dtype=float)
        multiplicator = np.zeros(N_t,dtype=float)
        
        #perform the computation
        # - perform the minimization
        if not init==None:
            v0 = np.load(init)
        else:
            v0 = np.mean(dynspec,axis=1)
        trust = N_fnu*1.0e-4
        print(trust)
        #bounds= scipy.optimize.Bounds (v0*0.01, v0*2.)
        res = scipy.optimize.minimize(
            func_for_min, v0, args= (dynspec,indices_tau,indices_ft),tol=trust,
            method=method, jac=False, options=dict(disp=True,ftol=trust,return_all=True,maxiter=maxiter) )
        multiplicator = res.x
        multiplicator = multiplicator/np.mean(multiplicator)
        DS_deconv = dynspec/multiplicator[:,na]
        A_DynSpec = np.fft.fft2(DS_deconv)
        SS_deconv = np.abs(np.fft.fftshift(A_DynSpec,axes=(0,1)))
        
        #save the results
        self._add_result(DS_deconv,"DS_deconv")
        self._add_result(t,"t")
        self._add_result(nu,"nu")
        self._add_result(f_t,"f_t")
        self._add_result(f_nu,"f_nu")
        self._add_result(SS_deconv,"SS_deconv")
        self._add_result(multiplicator,"multiplicator")
    