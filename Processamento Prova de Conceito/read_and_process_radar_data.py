# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:51:13 2020

@author: Alexandra Lopes
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def read_uwb(filename):
    
    f = open(filename, "r")
    
    fps = float(f.readline().split()[1])
    area_start = float(f.readline().split()[1])
    area_end = float(f.readline().split()[1])
    fs = float(f.readline().split()[1])
    fc = float(f.readline().split()[1])
    prf = float(f.readline().split()[1])
    dac_min = float(f.readline().split()[1])
    dac_max = float(f.readline().split()[1])
    interations = float(f.readline().split()[1])
    duty_ = float(f.readline().split()[1])
    pulses_per_step = float(f.readline().split()[1])
    #space
    f.readline()
    f.readline()
    f.readline()
    lines = f.readlines()
    f.close()
    
    time = lines[0::4]
    amplitude = lines[1::4]
 
    time_list = []
    amplitude_list = []
    
    for row in time:
        time_list.append(np.fromstring(row, dtype=float, sep=" "))
        
    for row in amplitude:
        amplitude_list.append(np.fromstring(row, dtype=float, sep=" "))   
        
   
    amplitude_list = np.asarray(amplitude_list)
    
    time_list = np.asarray(time_list)
    time_list = time_list - time_list[0]
    #print(time_list)
    
    return [{"fps": fps, "area_start": area_start, "area_end": area_end, "fs": fs, "fc": fc, "prf": prf, "dac_min": dac_min,"dac_max":dac_max,"interations":interations,
            "duty_": duty_, "pulses_per_step": pulses_per_step}, time_list, amplitude_list]

def proc(rangeProfile):
    vec = rangeProfile[0::2] + rangeProfile[1::2]*(2**8)
    for i in range(len(vec)):
        if vec[i] > (2**15):
            vec[i] = vec[i] - (2**16)

    vec_cplx = vec[0::2]+1j*vec[1::2]
    return vec_cplx

def process(data):
    data_cplx = []
    for row in data:
        if (len(row)%2) == 1:
            data_cplx.append(proc(row[184:-3]))
        if (len(row)%2) == 0:
            data_cplx.append(proc(row[184:]))
    
    return data_cplx

def frames(data):
    # find index value
    #print(data)
    searchval = [2,1,4,3,6,5,8,7]
    N = len(searchval)
    possibles = np.where(data == searchval[0])[0]
    solns = []
    for p in possibles:
        check = data[p:p+N]
        if np.all(check == searchval):
            solns.append(p)
    # append 
    data_frame = []
    a = 0
    for i in solns:
        data_frame.append(np.array(data[a:i]))
        a = i
        
    return data_frame

def read_texas(filename):
    f = open(filename, "r")

    f.readline()
    lines = f.readlines()
    f.close()

    fps = float(20)
    area_start = float(0)
    area_end = float(3)

    time = lines[0::2]
    amplitude = lines[1::2]
    #print(amplitude)
 
    time_list = []
    amplitude_list = []
    
    for row in time:
        time_list.append(np.fromstring(row, dtype=float, sep=" "))
        
    for row in amplitude:
        amplitude_list.extend(np.fromstring(row, dtype=float, sep=" "))   
   
    amplitude_list = np.array(amplitude_list)
    
    time_list = np.asarray(time_list)
    time_list = time_list - time_list[0]
    #print(amplitude_list)
    
    # process raw by initial word
    data_frame = frames(amplitude_list)
    # process data to complex
    data_cplx = process(data_frame)[3::]
    
    a = len(data_cplx)
    b = len(data_cplx[1])
    print(a)
    print(b)
    teste = np.zeros((a,b), dtype = complex)
    
    for i in range(len(data_cplx)):
        row = data_cplx[i]
        for j in range(len(row)):
            teste[i,j] = complex(row[j])

    return [{"fps": fps, "area_start": area_start, "area_end": area_end}, time_list, teste]

def downconversion(rf_data,fc,fs):
    # multiply frame by a complex sine 
    csine = np.exp(-1j*fc/fs*2*np.pi*np.arange(rf_data.shape[1]))
    cframe = rf_data * csine
    
    # low pass filter with a hamming low pass tps = 26   
    taps = 26
    cut_off = 0.1
    h = signal.firwin(taps, cut_off, window='hamming')
    
    baseband_data = []
    
    for i in range(rf_data.shape[0]):
        baseband_data.append(signal.filtfilt(h, 1.0, cframe[i,:]))
        
    baseband_data = np.asarray(baseband_data)
    
    return baseband_data


def make_fft(ys):
    return np.fft.fft(ys)

def hamming_transformation(ys):
    hamming = np.hamming(len(ys))
    signal = ys * hamming
    return signal

def phase_transformation(data):
    col = np.unwrap(np.angle(data))
    return col

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def FIR_filter(signal_, cut_value, taps, type_filter, axis):
   
    if axis == "slow_time":
        axis = 0
    else:
        axis = 1
        
    pass_zero_ = False
    if type_filter == "lowpass":
        pass_zero_ = True
    
    h = signal.firwin(taps, cut_value, window='hamming', pass_zero = pass_zero_)
        
    filtered_signal = []
    
    for i in range(signal_.shape[0]):
        filtered_signal.append(signal.filtfilt(h, 1.0, signal_[i,:]))
        
    filtered_signal = np.asarray(filtered_signal)
    
    return filtered_signal