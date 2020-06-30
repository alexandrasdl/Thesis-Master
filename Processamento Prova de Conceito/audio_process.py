# XXX_Start is the index of the end of the bip sound 
# XXX_End is the index of the end of the bop sound

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.signal import savgol_filter
import novainstrumentation as ni
from scipy import signal 
import signal_processing_functions as sp


# find_index need the introduction of audio parameters in seconds (need conversion before)
def find_index(data,audio_cut_start,audio_cut_end,style, name_type, plot):
    process_data = np.array(data[audio_cut_start:audio_cut_end])
    if style == "Start":
        index = np.min((np.where(process_data > 50))[0])
    elif style == "End":
        index = np.min((np.where(process_data > 50))[0])
    if (style == "Bitalino_Start"):
        index = np.max((np.where(process_data > 50))[0])
        
    if (plot==1):
        plt.plot(process_data)
        plt.axvline(x = index,color='red')
        plt.title(name_type)
        plt.show()
    # Convert to original data index 
    index = index + audio_cut_start
    return index

# return index where audio start
def start(header, plot):
    
    filename = header["audio_filename"]
    distance = header["distance"]
   
    # read audio
    fs, data = wavfile.read(filename)
    t = np.linspace(0, int(len(data)/fs), num = len(data))
    
    # extract seconds of sound
    
    bitalino_start_time = int(header["bitalino_start_time"] * fs)
    uwb_start_time = int(header["uwb_start_time"] * fs)
    uwb_end_time = int(header["uwb_end_time"] * fs)
    fmcw_start_time = int(header["fmcw_start_time"] * fs) 
    fmcw_end_time = int(header["fmcw_end_time"] * fs)
    bitalino_end_time = int(header["bitalino_end_time"] * fs)
    
    d = 5 * fs

    ### B_Start (between 06:148 and 06:20)
    B_Start = find_index(data,bitalino_start_time,bitalino_start_time + d,"Bitalino_Start","Bitalino_Start" + str(distance) + " m", plot)
    print("ok")
    ### UWB_Start 
    R1_Start =  find_index(data,uwb_start_time,uwb_start_time+d,"Start","UWB_Start" + str(distance) + " m", plot)
    ### UWB_End 
    R1_End =  find_index(data,uwb_end_time,uwb_end_time+d,"End","UWB_End" + str(distance) + " m", plot)
    
    ### FMCW_Start 
    R2_Start =  find_index(data,fmcw_start_time,fmcw_start_time+d,"Start","FMCW_Start" + str(distance) + " m", plot)
    
    ### FMCW_End 
    R2_End =  find_index(data,fmcw_end_time,fmcw_end_time+d,"End","FMCW_End" + str(distance) + " m", plot)
    
    ### B_End (between 06:148 and 06:20)
    B_End =  find_index(data,bitalino_end_time,bitalino_end_time+d,"End","Bitalino_End" + str(distance) + " m", plot)

    return data, t, fs, B_Start, B_End, R1_Start, R1_End, R2_Start, R2_End 

### EXTRA FUNCTIONS
# save .wav file throught data array
# wavfile.write("test.wav", fs, process_data)

def process(data):
    
     data = (abs(data))
     # teste = ni.bandpass(data, 0.2, 0.8)
     # plt.plot(teste)
     # plt.show()     
     data_smooth = ni.smooth(data,100000)
     data_log = 20*np.log10(data_smooth/(2**16))
     # print(len(data_smooth))
     # plt.plot(data_smooth)
     # plt.title("smooth3")
     # plt.show()

     # data_smooth = ni.smooth(data,10000)
     # print(len(data_smooth))
     # plt.plot(data_smooth)
     # plt.title("smooth")
     # plt.show()
     
     # data_smooth = ni.smooth(data_smooth,100000)
     # print(len(data_smooth))
     # plt.plot(data_smooth)
     # plt.title("smooth2")
     # plt.show()

     # plt.plot(20*np.log10(data_smooth/(2**16)))
     # plt.title("log de smooth")
     # plt.show()
 
     return data_log
    
