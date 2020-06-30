import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

filename = "bitalino_Wilson_0.5m.txt"

def start(filename,plot):
    f = open(filename, "r")
    
    #space
    f.readline()
    time_bip_end = f.readline()
    f.readline()
    
    lines = f.readlines()
    f.close()
    
    fs = 1000
    time_start = lines[0::4]
    time_end = lines[4::4]
    amplitude = lines[1::4]
    
    time_start_list = []
    time_end_list = []
    amplitude_list = []
    
    for row in time_start:
        time_start_list.append(np.fromstring(row, dtype=float, sep=" "))
        
    for row in time_end:
        time_end_list.append(np.fromstring(row, dtype=float, sep=" "))
        
    for row in amplitude:
        amplitude_list.append(np.fromstring(row, dtype=float, sep=" "))   
       
    amplitude_list = np.asarray(amplitude_list)
    
    
    time_start_list = np.asarray(time_start_list)
    #time_start_list = time_start_list - time_start_list[0]
    
    time_end_list = np.asarray(time_end_list)
    #time_end_list = time_end_list - time_end_list[0]
    
    ecg_signal = []
    
    for line in amplitude_list:
        ecg_signal.extend(line[5::6])
    
    if plot == 1:
        plt.plot(ecg_signal)
        plt.title("ECG test")
        plt.show()
    
    t = np.linspace(0, int(len(ecg_signal)/fs), num = len(ecg_signal))
    
    return ecg_signal, t, fs
