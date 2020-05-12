import numpy as np
from numpy import newaxis as na
from numpy.ctypeslib import ndpointer
import scipy
from scipy.ndimage.filters import uniform_filter1d
from scipy.optimize import curve_fit
import math
import progressbar
import ctypes
import cv2
import skimage
import skimage.morphology as morphology
#from astropy.timeseries import LombScargle #does not work for python 2 !

from .result_source import result_source as scinter_computation

class defined_computations():
    
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
        DSpec = np.delete(DSpec,flags,axis=0)
        
        #perform the computation
        amplitude = np.zeros(f_t.shape,dtype = float)
        angle = np.zeros(f_t.shape,dtype = float)
        posf_t = f_t[int(self.N_t/2):]
        bar = progressbar.ProgressBar(maxval=self.N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_nu in range(self.N_nu):
            bar.update(i_nu)
            LS = LombScargle(t, DSpec[:,i_nu])
            amplitude[int(self.N_t/2):] = np.sqrt(LS.power(posf_t,method='cython'))
            amplitude[1:int(self.N_t/2)] = amplitude[2*int(self.N_t/2)+1:int(self.N_t/2):-1]
            for i_t in range(int(self.N_t/2)+1,self.N_t):
                angle[i_t] = np.angle(LS.model_parameters(f_t[i_t])[2]+1.0j*LS.model_parameters(f_t[i_t])[1])
            angle[1:int(self.N_t/2)] = -angle[2*int(self.N_t/2)+1:int(self.N_t/2):-1]
            LS_SSpec[:,i_nu] = amplitude*np.exp(1.0j*angle)
        bar.finish()
        # - perform FFT along remaining axis
        LS_SSpec = np.fft.fft(LS_SSpec, axis=1)
        LS_SSpec = np.fft.fftshift(LS_SSpec, axes=1)
        
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
        nu_pos = np.linspace(0,lnu-1,num=N_slice,endpoint=True,dtype=int)
        nus = np.zeros(nu_pos.shape,dtype=float)
        midnus = np.zeros(nu_pos.shape,dtype=float)
        for i,index in enumerate(nu_pos):
            nus[i] = self.nu[index]
            midnus[i] = self.nu[index+lnu/2]
        nuref = self.nu_half
        DSpec = np.zeros((N_slice,self.N_t,lnu),dtype=float)
        for i_slice in range(N_slice):
            # - complex values are not supported by the C FT function
            DSpec[i_slice,:,:] = np.abs(DynSpec[pos1,pos2,:,nu_pos[i_slice]:lnu+nu_pos[i_slice]])
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
        
    def linSS_real(self):
        #load and check data
        source = scinter_computation(self.dict_paths,"nut_SecSpec")
        SSpec,f_t,f_nu = source.load_result(["secondary_spectrum","doppler","delay"])
        # - sparate 2pi from the Fourier frequency
        f_t /= 2.*math.pi
        f_nu /= 2.*math.pi
        
        #load specifications
        N_delay = self._add_specification("N_delay",1000)
        doppler_max = self._add_specification("doppler_max",60.0e-3) #Hz
        ddd_max = self._add_specification("delay_doppler_ratio_max",1./20.) #s^2
        pos1 = self._add_specification("telescope1",0)
        pos2 = self._add_specification("telescope2",0)
        
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
        
        #crop data
        dop_mask = np.where((doppler>dop_min) & (doppler<dop_max))[0]
        ddd_mask = np.where((ddd>ddd_min) & (ddd<ddd_max))[0]
        linSS = linSS[np.ix_(dop_mask,ddd_mask)]
        doppler = doppler[dop_mask]
        ddd = ddd[ddd_mask]
        N_ddd = len(ddd)
        N_dop = len(doppler)
        
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
        
        #save the results
        self._add_result(curv,"curv")
        self._add_result(curv_sum,"curv_sum")