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

def start(filename, plot):
    
    # namefile
    radar_type = filename.split("_")[0]
    print(filename)
    distance = float((filename.split("_")[2]).split(".txt")[0])
    print(distance)

    # Create FMCW object 
    fmcw = radar.Radar(filename,radar_type)
    
    # Cut data
    #fmcw.slowtime_cut(5, 25)
    
    # Plot Raw Data
    if plot == 1:
        fmcw.plot_matrix_radar(bas ="absolute", title="FMCW Raw Data")
        plt.axvline(x = distance, color='red')
        plt.show()
        
    # Remove Clutter and Plot
    fmcw.subtractMean()
    if plot == 1:
        fmcw.plot_matrix_radar(bas ="absolute", title="FMCW after Remove Clutter DC Data")
        plt.axvline(x = distance, color='red')
        plt.show()
        
    # Phase transformation
    fmcw.Phase_conversion()
    if plot == 1:
        fmcw.plot_matrix_radar(bas ="normal", title="FMCW after phase along axis transformation")
        plt.axvline(x = distance, color='red')
        plt.show()

    # Plot 3D Graph 
    #fmcw.plot_matrix_radar_3D(bas ="absolute", title="3D plot")
    
    # Generate time, distamce and istance axis
    distance_axis = fmcw.distance_axis()
    time_axis = fmcw.time_axis()
    freq_axis = fmcw.freq()
    
    # Plot Range Doppler && Plot Range Doppler until 
    if plot == 1:
        fmcw.plot_range_doppler_matrix(bas = "absolute")
        plt.axvline(x = distance, color='red')
        fmcw.plot_range_doppler_matrix(bas = "absolute",max_frequency_value = 2)
    
    # Plot Spectrum at index y
    if plot == 1:
        d_index = fmcw.distance_to_bin(distance)
        #fmcw.plot_spectrum_index(bas = "absolute", index = d_index)
        fmcw.plot_spectrum_index(bas = "absolute", index = d_index, max_frequency_value = 2)
    
    # Plot Data at index y: first convert ditance to index, plot two type of graphs raw#normal,absolute#abs,phase#angle
    if plot == 1:
        fmcw.plot_signal_distance(d_index, bas = "raw")
        plt.show()

    
    return fmcw.data_(), fmcw.time_axis(), fmcw.distance_axis(), fmcw.freq()


def return_index_raw(filename):
    
    # namefile
    radar_type = filename.split("_")[0]
    distance = float((filename.split("_")[2]).split(".txt")[0])
    
    # Create UWB object 
    fmcw = radar.Radar(filename,radar_type)
    
    # Cut data
    #breath.slowtime_cut(5, 25) 
        
    # Remove Clutter and Plot
    fmcw.subtractMean()
    
    # Phase transformation
    #fmcw.Phase_conversion()

    # Generate time, distance and instance axis
    distance_axis = fmcw.distance_axis()
    time_axis = fmcw.time_axis()
    freq_axis = fmcw.freq()
        
    # Plot at distance d-0.1 , d , d+0.1
    index_distance_start = fmcw.distance_to_bin(distance - 0.1)
    index_distance_end = fmcw.distance_to_bin(distance + 0.1)
    index_distance = fmcw.distance_to_bin(distance)
    
    # Plot at distance d-0.1 , d , d+0.1
    fmcw.plot_signal_distance(index_distance_start, bas = "phase")
    plt.title("-0.1")
    plt.show()    
    fmcw.plot_signal_distance(index_distance, bas = "phase")
    plt.title("0")
    plt.show()
    fmcw.plot_signal_distance(index_distance_end, bas = "phase") 
    plt.title("+0.1")    
    plt.show()
    
    # test unwrapping phase
    return process_raw(time_axis, np.angle(fmcw.distance(index_distance)))

def process_raw(time, signal):    
    
    signal = np.unwrap(signal)
    plt.plot(time, signal)
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
    
    #bandpass
    signal_pb = ni.bandpass(signal, 1.0, 1.6, order = 2, fs = 20)
    plt.plot(signal_pb)
    plt.show()
    #low pass
   #  signal_hp = ni.highpass(signal,f = 1.5, fs = 10)
   #  signal_hp = ni.lowpass(signal_hp,f = 3.0, fs = 10)
   #  signal_pb = ni.bandpass(signal, 1.0, 1.6, order = 10, fs = 10, use_filtfilt = True)
   #  # signal_diff_hp = ni.highpass(signal_diff,f = 1.0, fs = 20)
   #  # signal_diff_hp = ni.lowpass(signal_diff_hp,f = 2.0, fs = 20)
   # # plt.plot(signal)
   #  plt.plot(signal_hp)
   #  # plt.plot(signal_diff_hp)
   #  plt.show()
    
   # # plt.plot(signal)
   #  plt.plot(signal_pb)
   #  # plt.plot(signal_diff_hp)
   #  plt.show()
    
   #  signal_pb = ni.bandpass(signal_diff, 1.0, 1.6, order = 10, fs = 10, use_filtfilt = True)
   #  # signal_diff_hp = ni.highpass(signal_diff,f = 1.0, fs = 20)
   #  # signal_diff_hp = ni.lowpass(signal_diff_hp,f = 2.0, fs = 20)
   # # plt.plot(signal)
   #  plt.plot(signal_pb)
   #  # plt.plot(signal_diff_hp)
   #  plt.show()
    
    
    return signal_pb
    
    #median
    # signal_smooth = rd.smooth(signal,window_len = 20)
    # plt.plot(signal)
    # plt.plot(signal_smooth)
    # plt.show()