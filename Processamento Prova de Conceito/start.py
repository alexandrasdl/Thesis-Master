import read_raw_files as read
import audio_process as audio
import bitalino_process as bitalino
import uwb_process as uwb
import fmcw_process as fmcw

import numpy as np
import matplotlib.pyplot as plt
import radar_processing as radar

# >> --- HEADER --- <<
#  filename | distance | start_time_bitalino | uwb_stat_time | uwb_end_time | fmcw_stat_time | fmcw_end_time | end_time_bitalino 

# >>> --- filename type --- <<<
# AUDIO:    audio_name
# BITALINO: bitalino_name
# UWB:      x4_name_distance
# FMCW:     texas_name_distance

# READ FILES INFO
# header = [distance, audio_filename, bitalino_filename, uwb_filename, fmcw_filename , start_time_bitalino , uwb_stat_time , uwb_end_time , fmcw_stat_time , fmcw_end_time , end_time_bitalino]
header_file = read.read()

# AUDIO 
# Read and cut audio file in frames corresponding to bitalino, radar uwb and radar fmcw - Plot 1 to show cut result
[data_audio, t_audio, fs_audio, B_Start, B_End, R1_Start, R1_End, R2_Start, R2_End] = audio.start(header_file[0], 0)

# BITALINO
# Read ecg file -  Plot 1 to show result
[data_ecg, t_ecg, fs_ecg] = bitalino.start(header_file[0]["bitalino_filename"],0)

# RADAR UWB RF
[data_uwb, t_uwb, d_uwb, fs_uwb] = uwb.start(header_file[0]["uwb_filename"],0,0)
signal_bandpass = uwb.return_index_raw(header_file[0]["uwb_filename"],0)

# RADAR UWB BASEABAND
[data_uwb, t_uwb, d_uwb, fs_uwb] = uwb.start(header_file[0]["uwb_filename"],1,0)
#signal_bandpass = uwb.return_index_raw(header_file[0]["uwb_filename"],1)

plt.plot(t_ecg[int(((R1_Start-B_Start)/fs_audio) * fs_ecg) : int(((R1_End-B_Start)/fs_audio) * fs_ecg)], data_ecg[int(((R1_Start-B_Start)/fs_audio) * fs_ecg) : int(((R1_End-B_Start)/fs_audio) * fs_ecg)])
t_uwb = np.linspace(0, len(signal_bandpass)*1/10, len(signal_bandpass), endpoint=True)
plt.plot(t_uwb+(t_ecg[int(((R1_Start-B_Start)/fs_audio) * fs_ecg) : int(((R1_End-B_Start)/fs_audio) * fs_ecg)])[0],signal_bandpass*2000+550)
print("lol")
plt.show()

# RADAR FMCW
[data_fmcw, t_fmcw, d_fmcw, fs_fmcw] = fmcw.start(header_file[0]["fmcw_filename"],0)
#signal_bandpass = fmcw.return_index_raw(header_file[0]["fmcw_filename"])

# plt.plot(t_ecg[int(((R2_Start-B_Start)/fs_audio) * fs_ecg) : int(((R2_End-B_Start)/fs_audio) * fs_ecg)], data_ecg[int(((R2_Start-B_Start)/fs_audio) * fs_ecg) : int(((R2_End-B_Start)/fs_audio) * fs_ecg)])
# t_fmcw = np.linspace(0, len(signal_bandpass)*1/20, len(signal_bandpass), endpoint=True)
# plt.plot(t_fmcw+(t_ecg[int(((R2_Start-B_Start)/fs_audio) * fs_ecg) : int(((R2_End-B_Start)/fs_audio) * fs_ecg)])[0],signal_bandpass*100+550)
# print("lol")
# plt.show()

# >> --- FUNCTIONS --- <<

def check_data():

    print(" ### CHECK DATA INTEGRITY ###")
    print(" >> Check audio and bitalino size. It's supposed be equal!")
    print(" >> DATA_ECG " + str((len(data_ecg)/fs_ecg)/60) + " // DATA_AUDIO " + str((B_End-B_Start)/fs_audio/60))
     
    ### PLOT
    a = data_audio[B_Start:B_End] 
    t_audio = np.linspace(0, len(a)/fs_audio, num = len(a))
    plt.plot(t_audio, a)
    plt.plot(t_ecg, data_ecg)
    plt.tiltle("DATA BITALINO + AUDIO")
    plt.show()


