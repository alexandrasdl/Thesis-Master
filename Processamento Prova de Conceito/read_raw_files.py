## FUNCTIONS AVAILABLE
# read(): read files from .txt and send the header of files.

# structure data file 
# name | distance | bitalino_start_time | uwb_start_time | uwb_end_time | fmcw_start_time | fmcw_end_time | bitalino_end_time 

def read():
    
    f = open("files.txt", "r")
    
    # read file
    files = f.read().split("\n")
    files =  files[2::2]

    for row in files:
        print(row)

    number_files = len(files)
    
    # header = [distance, audio_filename, bitalino_filename, uwb_filename, fmcw_filename , bitalino_start_time , uwb_start_time , uwb_end_time , fmcw_start_time , fmcw_end_time , bitalino_end_time]
    header = []

    if number_files > 0:
        for i in range(number_files):
            header_file = {}
            # split all elements and remove 
            file = (files[i].split(" "))[1::]
            name = file [0]
            distance = file[1]
            # generate audio filename
            audio_filename = "audio_" + name + ".wav"
            bitalino_filename = "bitalino_" + name + "_" + distance + ".txt"
            uwb_filename = "x4_" + name + "_" + distance + ".txt"
            fmcw_filename = "texas_" + name + "_" + distance + ".txt"
        
            start_time_bitalino = file [2]
            uwb_time_bitalino = file[3]
            uwb_end_time = file[4]
            fmcw_start_time = file[5]
            fmcw_end_time = file[6]
            end_time_bitalino = file[7]
            
            header_file["distance"] = distance
            header_file["audio_filename"] = audio_filename
            header_file["bitalino_filename"] = bitalino_filename
            header_file["uwb_filename"] = uwb_filename 
            header_file["fmcw_filename"] = fmcw_filename
                
            header_file["bitalino_start_time"] = float(start_time_bitalino)
            header_file["uwb_start_time"] = float(uwb_time_bitalino)
            header_file["uwb_end_time"] = float(uwb_end_time)
            header_file["fmcw_start_time"] = float(fmcw_start_time)
            header_file["fmcw_end_time"] = float(fmcw_end_time)
            header_file["bitalino_end_time"] = float(end_time_bitalino)
            
            header.append(header_file)
    else: 
        print("There is no files to process!")
 
    return header


    