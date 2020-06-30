# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:28:09 2020

@author: Alexandra Lopes
"""
##notas: 1) verficiar o 20log10(xxx) // 2) 
### Radar ###

# ## read method ##
# __init__ read data

# ## data trasform methods (should be call before any other method!) ##
# RF_downconversion(self) make downconversion
# slowtime_cut(self,time_start,time_end) cut data 
# subtractMean(self) subctract mean and remove static clutter

# ## get methods ##
# data_(self) get all data!
# fps(self):
# fs(self):
# fc(self):
# def fasttime_size(self):
# slowtime_size(self):
# def time_per_bin(self): only for UWB, return number of bin per sample
# distance_to_bin(self, distance): only for UWB, return distance to bin
 
# ## methods for axis ##
# distance_axis(self): return distance axis
# time_axis(self) return time axis
# fasttime_axis(self): return fast time axis (bin of distance axis)   
# slowtime_axis(self): return slow time axis( bin of time axis)
# freq(self): return frequeny axis -- with shift to be used in plot frequency!

# ## method for frame acquisiton ##   
# data_frame(self, time): return data along axis in a fixed time
# def distance(self,distance): return data along axis in a fixed distance (it is the used to measure breathing)
   
# ## plot graphs
# plot_matrix_radar(self, bas, title): plot 2D graph
# plot_matrix_radar_3D(self, bas, title): plot 3D graph
# plot_range_doppler_matrix(self, bas, title = "None") Plot 2D Range Doppler Matrix
# plot_3D_spectrum(self, bas, title = "None"): Plot 3D Range Doppler Matrix
# # plot index funcions below
# plot_signal_time(self,time): plot frame at index x (need the exact index to work!)
# plot_signal_distance(self, distance, bas = "optional"): plot frame distance at index x (need the exact index to work!) -- (it is the used to measure breathing)
# plot_spectrum_index(self, bas, index, title = "None"): plot spectrum at index 

# ## auxiliar methods (used as intermediate in functions above)
# range_doppler_along_slowtime(self): make range doople along slow time after hamming application (REVER)
# range_doppler_along_fasttime(self): make range doople along slow time without hamming application 
        

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import read_and_process_radar_data as rd
from mpl_toolkits.mplot3d import axes3d
import scipy.fftpack as sfft

class Radar:
    
    def __init__(self,filename,radar_type):
        
        if radar_type == "x4":
            data = rd.read_uwb(filename)
            # paramenters of acquistion
            self.data_info = data[0]
            # timestap array of each frame
            self.timestamp = data[1]
            # radar matrix data 
            self.data = data[2]
            # recorded in RF mode...s
            self.baseband = False
        if radar_type == "texas":
            data = rd.read_texas(filename)
            # paramenters of acquistion
            self.data_info = data[0]
            # timestap array of each frame
            self.timestamp = data[1]
            # radar matrix data 
            self.data = data[2]

    
    # methods for radar data manipulation #
    # downconversion to baseband 
            
    def Custom_modification(self):
        print("ola")
        
    def RF_downconversion(self):
        self.baseband = True 
        self.data = rd.downconversion(self.data, self.data_info["fc"], self.data_info["fs"])
    
    def Phase_conversion(self):
        data_phase = np.apply_along_axis(rd.phase_transformation, 0 , self.data)
        self.data = data_phase
        
    def slowtime_cut(self,time_start,time_end):
        self.data = self.data[time_start*self.fps():time_end*self.fps(),:]
        
    def fasttime_cut(self,distance_start,distance_end):
        self.data_info["area_start"] = self.time_per_bin()*distance_start
        self.data_info["area_end"] = self.time_per_bin()*distance_end
        self.data = self.data[:,distance_start:distance_end]
        
    # subtract mean 
    def subtractMean(self):
        self.data = self.data - self.data.mean(axis = 0, keepdims=True)
        
    # methods for data acquisition parameters #
    # return data
             
    def data_(self):
        return np.array(self.data)
    # return frames per second    
    def fps(self):
        return int(self.data_info["fps"])
    # return sampling frequency fast time
    def fs(self):
        return self.data_info["fs"]
    # return central frequency
    def fc(self):
        return self.data_info["fc"]
    # return size of fast time data
    def fasttime_size(self):
        return int(self.data.shape[1])
    # return size of slow time data
    def slowtime_size(self):
        return int(self.data.shape[0])
    
    def time_per_bin(self):
        return 0.00643
    
    def distance_to_bin(self, distance):
        return int((np.where(self.distance_axis() >= distance)[0])[0])
    
    def time_to_bin(self, time):
        return int(time*self.fps())
        
    ## methods for axis ##
    # distance
    def distance_axis(self): 
        return np.linspace(self.data_info["area_start"], self.data_info["area_end"], self.fasttime_size(), endpoint=True)
    # time
    def time_axis(self):
        return np.linspace(0, self.slowtime_size()*1/self.fps(), self.slowtime_size(), endpoint=True)
    # fast time axis
    
    def fasttime_axis(self):
        return np.linspace(0, self.fasttime_size(), self.fasttime_size(), endpoint=False)
    # slow time axis
    
    def slowtime_axis(self): 
        return np.linspace(0, self.slowtime_size(), self.slowtime_size(), endpoint=False) 
    ## method for frame acquisiton ##
    
    def data_frame(self, time):
        return self.data[time,:]
    
    def distance(self,distance):
        return self.data[:,distance]

    ## plot graphs
    def plot_matrix_radar(self, bas, title):
        x = self.distance_axis()
        y = self.time_axis()
        
        if bas == "absolute":
            z = np.absolute(self.data)
        elif bas == "phase":
            z = np.angle(self.data)
        else:
            z = self.data
        
        z_min, z_max = -np.abs(z).max(), np.abs(z).max() 
        fig, ax = plt.subplots() 
        c = ax.pcolormesh(x, y, z, cmap ='viridis', vmin = z_min, vmax = z_max) 
        
        fig.colorbar(c, ax = ax) 
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("time (sec)")
        ax.set_title(title) 
       # plt.axvline(x=x[32],color='red')
       # plt.axvline(x=x[25],color='red')
    #   plt.axvline(x=x[31],color='red')
    #   plt.scatter(x[32],y[575])
        
        
        ## plot graphs
    def plot_matrix_radar_3D(self, bas, title):
        x = self.distance_axis()
        y = self.time_axis()
        
        if bas == "absolute":
            z = np.absolute(self.data)
        elif bas == "phase":
            z = np.angle(self.data)
        else:
            z = self.data
        
        X,Y = np.meshgrid(x,y)
        ax = plt.axes(projection='3d')
        
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("time (sec)")
        ax.set_zlabel('z')
        ax.set_title(title) 

        ax.plot_surface(Y, X, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.contourf(Y, X, z)
        
        plt.show()
        
        
    def plot_signal_distance(self,distance, bas = "optional"):
        index = distance
        x = self.distance_axis()
        #plt(x(index))
        if bas == "raw":
            plt.plot(self.time_axis(), self.distance(distance))
        if bas == "absolute":
            plt.plot(self.time_axis(), np.abs(self.distance(distance)))
        elif bas == "phase":
            plt.plot(self.time_axis(),(np.angle(self.distance(distance))))
            plt.plot(self.time_axis(), np.unwrap(np.angle(self.distance(distance))))
       #plt.plot(self.time_axis(), self.distance(distance))
        plt.title(bas + " signal at distance: " + str(x[index]) )
        plt.ylabel(' ')
        plt.xlabel('time (sec)')
        
        
    def plot_signal_time(self,time):
        
        plt.plot(self.time_distance(), self.data_frame(time))
        plt.title("signal at time: " + str(time))
        plt.ylabel('dB')
        plt.xlabel('distance (m)')
        plt.show()

    def range_doppler_along_slowtime(self):
        data_hamming = np.apply_along_axis(rd.hamming_transformation, 0 , self.data)
        # plt.plot(data_hamming[:,111])
        # plt.show()
        return np.apply_along_axis(rd.make_fft, 0 , data_hamming)

    
    def freq(self):
        n = self.slowtime_size()
        d = 1 / self.fps()
        fs = np.fft.fftfreq(n, d) 
        #fs = np.fft.fftshift(fs)
        return fs
        
    def plot_range_doppler_matrix(self, bas, max_frequency_value = 0):
        
        d = self.distance_axis()
        fs = self.freq()
        data = self.range_doppler_along_slowtime()
        #create fs 
        z = data
        z = sfft.fftshift(z,axes=0)
        
        # cut frequency if different 0
        if max_frequency_value != 0:
            print("plot")
         #cut frequency in index...
            frequency = max_frequency_value
            cut = int((np.where(fs >= frequency)[0])[0])
            [fs, z] = cut_spectrum(fs,z,cut)
        
        if bas == "absolute":
            z = np.abs(z)
        elif bas == "phase":
            z = np.unwrap(np.angle(z)) 
        
       # plt.imshow(20*np.log10(z), extent=[d.min(), d.max(), fs.min(), fs.max()], aspect='auto')
        plt.imshow(z, extent=[d.min(), d.max(), fs.min(), fs.max()], aspect='auto')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Frequency (Hz)')
        
        if bas == "absolute":
            plt.title('Range Doppler Matrix - FFT absolute')
        elif bas == "phase":
            plt.title('range doppler Matrix - FFT phase')
    
        plt.colorbar();
        
        plt.show()
        
    def plot_spectrum_index(self, bas, index, max_frequency_value = 0):
        distance = self.distance_axis()
        ys = self.range_doppler_along_slowtime()
        fs = self.freq()
         # cut frequency if different 0
        if max_frequency_value != 0:
            frequency = max_frequency_value
            cut = int((np.where(fs >= frequency)[0])[0])
            [fs, ys] = cut_spectrum(fs,ys,cut)
        
        if bas == "absolute":
            z = np.abs(ys)[:,index]
        elif bas == "phase":
            z = np.angle(ys)[:,index]
        
        plt.plot(fs,20*np.log10(z))
        plt.ylabel('dB')
        plt.xlabel('Frequency (Hz)')
        plt.title('Range Doppler matrix at index = ' + str(distance[index]))
        plt.show()
        
        
    def plot_3D_spectrum(self, bas, title = "None"):
        #create fs 
        ys = self.range_doppler_along_slowtime()
        fs = self.freq()
        d = self.distance_axis()
        
        z = ys
        
        if bas == "absolute":
            z = np.abs(ys)
        elif bas == "phase":
            z = np.angle(ys) 
           
        X,Y = np.meshgrid(d,fs)
        ax = plt.axes(projection='3d')
        
        ax.set_xlabel('freq (Hz)')
        ax.set_ylabel('distance (m)')
        ax.set_zlabel('z')
        ax.set_title('RF SIGNAL');
        
        ax.plot_surface(Y, X, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.contourf(Y, X, z)

        plt.show()
    

        
### extra functions
# plot_range_doppler_matrix(d, fs, data, bas, max_value = "optional"): plot range doppler matrix from data 

def plot_range_doppler_matrix(d, fs, data, bas, max_value = "optional"):
        #create fs 
        z = data
        z = sfft.fftshift(z,axes=0)
        
        if bas == "absolute":
            z = np.abs(z)
        elif bas == "phase":
            z = np.unwrap(np.angle(z)) 
        
       # plt.imshow(20*np.log10(z), extent=[d.min(), d.max(), fs.min(), fs.max()], aspect='auto')
        plt.imshow(z, extent=[d.min(), d.max(), fs.min(), fs.max()], aspect='auto')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Frequency (Hz)')
        
        if bas == "absolute":
            plt.title('range doppler matrix - FFT absolute')
        elif bas == "phase":
            plt.title('range doppler matrix - FFT phase')
    
        plt.colorbar();
        
        if max_value != "optional":
            index = ((int(max_value)/d.shape[0])*(d.max()-d.min())) + d.min()
            print(index)
            plt.axvline(x=index,color='red')
        
        plt.show()
        
        # X,Y = np.meshgrid(d,fs)

        # ax = plt.axes(projection='3d')
        # if bas == "absolute":
        #     z = np.abs(data)
        # elif bas == "phase":
        #     z = np.angle(data) 
        
        # ax.set_xlabel('freq (Hz)')
        # ax.set_ylabel('distance (m)')
        # ax.set_zlabel('z')
        # ax.set_title('RF SIGNAL');
        
        # ax.plot_surface(X,Y, z)
        # ax.contourf(X,Y, z)

        # plt.show()
        
def cut_spectrum(fs,fft_radar,cut):
        fft_radar = np.concatenate((fft_radar[:cut,:], fft_radar[-cut:,:]))
        fs = np.concatenate((fs[:cut], fs[-cut:]), axis= 0)
        return [fs,fft_radar]
    
def plot_spectrum_index(fs, data, bas, index):
        z = data
        z = np.abs(data)[:,index]
        if bas == "absolute":
            z = np.abs(data)[:,index]
            plt.plot(fs,20*np.log10(z))
            plt.ylabel('dB')
        elif bas == "phase":
            z = np.angle(data)[:,index]
            plt.plot(fs,z)
            plt.ylabel('angle (rad)')

        plt.xlabel('Frequency (Hz)')
        plt.title('spectrum at the distance: ' + str(index))
        plt.show()
        


def spectrum_index(fs, data, bas, index):
        z = data
        z = np.abs(data)[:,index]
        if bas == "absolute":
            z = np.abs(data)[:,index]
            return 20*np.log10(z)
        elif bas == "phase":
            z = np.angle(data)[:,index]
            return z
        
        return 20*np.log10(z)
        
# return max val and index
def max_value(data):
        result = [data.max(), np.where(np.abs(data) == data.max())]
        return result


