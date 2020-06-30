import radar_processing as radar
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as sfft
import read_and_process_radar_data as rd
import novainstrumentation as ni

# @parameters 
# filenamne: name of file
# baseband: = 1 is transformations is required
# plot: = 1 if need to plot data

def start(filename, baseband, plot):
    
    # namefile
    radar_type = filename.split("_")[0]
    distance = float((filename.split("_")[2]).split(".txt")[0])
    
    # Create UWB object 
    uwb = radar.Radar(filename,radar_type)
    
    # Cut data
    #breath.slowtime_cut(5, 25)
    
    # Plot Raw Data
    if plot == 1:
        uwb.plot_matrix_radar(bas ="absolute", title="UWB Raw Data")
        plt.axvline(x = distance, color='red')
        plt.show()
        
    # Remove Clutter and Plot
    uwb.subtractMean()
    
    if plot == 1:
        uwb.plot_matrix_radar(bas ="absolute", title="UWB after Remove Clutter DC Data")
        plt.axvline(x = distance, color='red')
        plt.show()
    
    # Plot 3D Graph 
    # uwb.plot_matrix_radar_3D(bas ="absolute", title="3D plot")
        
    # Tranform data to Baseband Data  
    if baseband == 1:
        uwb.RF_downconversion()

    # Generate time, distamce and istance axis
    distance_axis = uwb.distance_axis()
    time_axis = uwb.time_axis()
    freq_axis = uwb.freq()
    
    # Plot Range Doppler && Plot Range Doppler until 
    if plot == 1:
        uwb.plot_range_doppler_matrix(bas = "absolute")
        plt.axvline(x = distance, color='red')
        uwb.plot_range_doppler_matrix(bas = "absolute",max_frequency_value = 2)
    
    # Plot Spectrum at index y
    if plot == 1:
        d_index = uwb.distance_to_bin(distance)
       # uwb.plot_spectrum_index(bas = "absolute", index = d_index)
        uwb.plot_spectrum_index(bas = "absolute", index = d_index, max_frequency_value = 2)
        
    # Plot Data at index y: first convert distance to index, plot two type of graphs raw#normal,absolute#abs,phase#angle
    index_distance = uwb.distance_to_bin(distance)
    # Plot
    if plot == 1:
        if baseband == 1:
            uwb.plot_signal_distance(index_distance, bas = "phase")
            plt.show()
        else:   
            uwb.plot_signal_distance(index_distance, bas = "raw")
            plt.show()

    return uwb.data_(), uwb.time_axis(), uwb.distance_axis(), uwb.freq()

def return_index_raw(filename, baseband):
    
    # namefile
    radar_type = filename.split("_")[0]
    distance = float((filename.split("_")[2]).split(".txt")[0])
    
    # Create UWB object 
    uwb = radar.Radar(filename,radar_type)
    
    # Cut data
    #breath.slowtime_cut(5, 25) 
        
    # Remove Clutter and Plot
    uwb.subtractMean()
        
    # Tranform data to Baseband Data  
    if baseband == 1:
        uwb.RF_downconversion()

    # Generate time, distance and instance axis
    distance_axis = uwb.distance_axis()
    time_axis = uwb.time_axis()
    freq_axis = uwb.freq()
        
    # Plot Data at index y: first convert distance to index, plot two type of graphs raw#normal,absolute#abs,phase#angle
    index_distance_start = uwb.distance_to_bin(distance - 0.1)
    index_distance_end = uwb.distance_to_bin(distance + 0.1)
    index_distance = uwb.distance_to_bin(distance)
    
    # plot distance - 0.1
    # if baseband == 1:
    #     uwb.plot_signal_distance(index_distance_start, bas = "phase")
    # else:
    #     uwb.plot_signal_distance(index_distance_start, bas = "raw")
    # plt.show()
    
    # # plot distance + 0.1
    # if baseband == 1:
    #     uwb.plot_signal_distance(index_distance, bas = "phase")
    # else:
    #     uwb.plot_signal_distance(index_distance, bas = "raw")
    # plt.show()  
     
    # # plot distance 
    # if baseband == 1:
    #     uwb.plot_signal_distance(index_distance_end, bas = "phase")
    # else:
    #     uwb.plot_signal_distance(index_distance_end, bas = "raw")
    # plt.show()
    
    # test unwrapping phase
    if baseband == 1:
        return process_raw(time_axis, np.angle(uwb.distance(index_distance)))
    else:
        return process_raw2(time_axis, uwb.distance(index_distance))
    
def process_raw2(time, signal):    
   # signal = rd.hamming_transformation(signal)
    plt.plot( time, signal)
    plt.title("signal raw")
    plt.show()    
    
    
    # low pass
    signal_lp = ni.lowpass(signal,f = 2.0, fs=10)
    plt.plot(signal_lp)
    plt.title("ni.lowpass")
    plt.show() 
    
    # plot diff
    signal_diff = (np.diff(signal,1))+5
    
    index = np.where( (signal_diff) >10* np.mean(signal_diff))
    for i in index[0]:
        plt.axvline(x = i, color='red')
    plt.plot(signal_diff)
    plt.title("np.dif")
    plt.show()   
    
    signal = signal_diff 
    #signal = abs(signal_diff)

    # low pass
    signal_lp = ni.lowpass(signal,f = 2.0, fs=10, use_filtfilt=True)
    plt.plot(signal_diff)
    plt.plot(signal_lp)
    plt.title("ni.lowpass5")
    plt.show() 
    
    
        #band pass
    signal_pb = ni.bandpass(signal, 1.0, 1.6, order = 10, fs = 10, use_filtfilt=True)
    plt.plot(signal_diff-5)
    plt.plot(signal_pb)
    plt.title("ni.bandpass")
    plt.show() 
    
    
    dif = signal_diff-signal_lp
    plt.plot(dif)
    print(dif)
    plt.title("dif")
    plt.show() 
    
    ##########

    
    #high pass
    # signal_hp = ni.highpass(signal,f = 1.0, fs = 10)
    # plt.plot(signal_hp)
    # plt.title("ni.highpass")
    # plt.show() 
    
    # signal_hp = ni.highpass(signal,f = 1.0, fs = 10, use_filtfilt = True)
    # plt.plot(signal_hp)
    # plt.title("ni.highpass")
    # plt.show() 
    
    # #band pass
    # signal_pb = ni.bandpass(signal, 1.0, 1.6, order = 10, fs = 10)
    # plt.plot(signal_pb)
    # plt.title("ni.bandpass")
    # plt.show() 
     
    return signal_pb
    
    #median
    # signal_smooth = rd.smooth(signal,window_len = 20)
    # plt.plot(signal)
    # plt.plot(signal_smooth)
    # plt.show()
    
# faz um passa banda e nao inventes mais...
def process_raw(time, signal):    
    signal = np.unwrap(signal)
    plt.plot( time, signal)
    plt.title("signal unwrapped")
    plt.show()    
    
    # plot diff
    signal_diff = np.diff(signal,1)
    bad_index = np.where(abs(signal_diff) > 0.25)
    for i in bad_index[0]:
       # plt.axvline(x = int(i), color='red')
        signal_diff[i] = 0.05
    plt.plot(signal_diff)
    plt.title("np.dif")
    plt.show()   
    
    # process
    # signal_proc = signal
    # signal_diff = np.diff(signal,1)
    # signal_diff_index = np.where(abs(signal_diff) >= 0.25 )
    # plt.plot(signal)
    # plt.plot(signal_proc)
    # plt.show()
    
    #low pass
    signal_hp = ni.highpass(signal,f = 1.5, fs = 10)
    signal_hp = ni.lowpass(signal_hp,f = 3.0, fs = 10)
    signal_pb = ni.bandpass(signal, 1.0, 1.6, order = 10, fs = 10, use_filtfilt = True)
    # signal_diff_hp = ni.highpass(signal_diff,f = 1.0, fs = 20)
    # signal_diff_hp = ni.lowpass(signal_diff_hp,f = 2.0, fs = 20)
   # plt.plot(signal)
    plt.plot(signal_hp)
    # plt.plot(signal_diff_hp)
    plt.show()
    
   # plt.plot(signal)
    plt.plot(signal_pb)
    # plt.plot(signal_diff_hp)
    plt.show()
    
    signal_pb = ni.bandpass(signal_diff, 1.0, 1.6, order = 10, fs = 10, use_filtfilt = True)
    # signal_diff_hp = ni.highpass(signal_diff,f = 1.0, fs = 20)
    # signal_diff_hp = ni.lowpass(signal_diff_hp,f = 2.0, fs = 20)
   # plt.plot(signal)
    plt.plot(signal_pb)
    # plt.plot(signal_diff_hp)
    plt.show()
    
    
    return signal_pb
    
    #median
    # signal_smooth = rd.smooth(signal,window_len = 20)
    # plt.plot(signal)
    # plt.plot(signal_smooth)
    # plt.show()

    

def unwrapping(p, discont=np.pi, axis=-1):

    p = np.asarray(p)
    nd = p.ndim
    dd = np.diff(p, axis=axis)
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)
    ddmod = np.mod(dd + np.pi, 2*np.pi) - np.pi
    np.copyto(ddmod, np.pi, where=(ddmod == -np.pi) & (dd > 0))
    ph_correct = ddmod - dd
    np.copyto(ph_correct, 0, where=(abs(dd) < discont))
    up = np.array(p, copy=True, dtype='d')
    plt.plot(ph_correct)
    plt.show()
    up[slice1] = p[slice1] + ph_correct.cumsum(axis)
    return up

