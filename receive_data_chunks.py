"""Example program to demonstrate how to read a multi-channel time-series
from LSL in a chunk-by-chunk manner (which is more efficient)."""

from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np 
from time import perf_counter 
import requests


def printMeta(stream_info_obj):
    """
    This function prints some basic meta data of the stream
    """
    print("") 
    print("Meta data")
    print("Name:", stream_info_obj.name())
    print("Type:", stream_info_obj.type())
    print("Number of channels:", stream_info_obj.channel_count())
    print("Nominal sampling rate:", stream_info_obj.nominal_srate())
    print("Channel format:",stream_info_obj.channel_format())
    print("Source_id:",stream_info_obj.source_id())
    print("Version:",stream_info_obj.version())
    print("")

def getRingbufferValues(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer): 
    """
    This function provides the most recent data samples and timestamps in a ringbuffer 
    (first val is oldest, last the newest) 

    Attributes:
        chunk               : current data chunk
        timestamps          : LSL local host timestamp for the data chunk 
        current_local_time  : LSL local client timestamp when the chunk is received
        timestamp offset    : correction factor that needs to be added to the timestamps to map it into the client's local LSL time
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamps buffer of shape (buffer_size, 3). The 3 columns correspond to the host timestamp, the client local time and time correction offset resp.

    Returns:
        data_buffer         : data buffer array of shape (buffer_size, n_channels)
        timestamp_buffer    : timestamp buffer of shape (buffer_size, 3)
    """
    #data 
    current_chunk = np.array(chunk)
    n_samples = current_chunk.shape[0] # shape (samples, channels)

    temp_data = data_buffer[n_samples:, :]
    data_buffer[0:temp_data.shape[0], :] = temp_data
    data_buffer[temp_data.shape[0]:, :] = current_chunk

    # timestamps 
    current_timestamp_buffer = np.array(timestamps)

    temp_time = timestamp_buffer[n_samples:, 0]
    timestamp_buffer[0:temp_time.shape[0], 0] = temp_time
    timestamp_buffer[temp_time.shape[0]:, 0] = current_timestamp_buffer

    # current local time and offset correction 
    temp_local_time = timestamp_buffer[n_samples:, 1]
    timestamp_buffer[0:temp_local_time.shape[0], 1] = temp_local_time
    timestamp_buffer[temp_local_time.shape[0]:, 1] = current_local_time

    temp_offset_time = timestamp_buffer[n_samples:, 2]
    timestamp_buffer[0:temp_offset_time.shape[0], 2] = temp_offset_time
    timestamp_buffer[temp_offset_time.shape[0]:, 2] = timestamp_offset

    return data_buffer, timestamp_buffer


def sendDetectedError(team_name, secret_id, timestamp_buffer_vals, local_clock_time):
    """
    This function gathers all the relevant results and sends it to the host.
    This function should be called everytime an error is detected.

    Attributes:
        team_name (str)         : each team will be assigned a team name which 
        secret_id (str)         : each team will be provided with a secret code
        timestamp_buffer_vals   : subset of the timestamp_buffer array at the instant when you have predicted an error and want to send the current result. Basically the i-th element of the timestamp_buffer array
        local_clock_time        : current LSL local clock time when you have run your classifier and predicted an error. This can be determined with the help of "local_clock()" call.
    """
    # calculate the final values for the timings 
    comm_delay = timestamp_buffer_vals[1] -timestamp_buffer_vals[0] -timestamp_buffer_vals[2]
    computation_time = local_clock_time - timestamp_buffer_vals[1]


    # connection to API for sending the results online 
    url = 'http://10.250.223.221:5000/results'
    myobj = {'team': team_name,
            'secret': secret_id,
            'host_timestamp': timestamp_buffer_vals[0], 
            'comp_time': computation_time, 
            'comm_delay': comm_delay}

    x = requests.post(url, json = myobj)


def main():

    #************************************************************
    # ********************** user params ************************
    #************************************************************

    #params 
    buffer_size = 2500  # size of ringbuffer in samples, currently set to 2500 (5 sec data times 500 Hz sampling rate)
    dt_read_buffer= 0.04 # time in seconds how often the buffer is read  (updated with new incoming chunks)

    # example team info 
    team_name = 'team1'
    secret_id = 'example_team'


    #************************************************************
    #************************************************************
    #************************************************************

    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG') # create data stream 

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0]) 
    stream_info = inlet.info()
    printMeta(stream_info) # print stream info 

    # run continiously 
    running = True


    # uncomment if data should be recorded (not necessary for online prediction, use buffer for that)
    # data_arr = []
    # time_stamp_arr = []


    #inits 
    data_buffer = np.zeros((buffer_size, stream_info.channel_count())) # data buffer has shape (buffer_size, n_channels) last 2 channels are only markers and indices ! 
    timestamp_buffer = np.zeros((buffer_size, 3)) # buffer for different kinds of timestamps or time values 

    # get timestamp offset 
    timestamp_offset = inlet.time_correction()


    while running:
       
        chunk, timestamps = inlet.pull_chunk() # get a new data chunk

        if(chunk): # if list not empty (new data)
            
            # get timing info 
            current_local_time = local_clock()
            timestamp_offset = inlet.time_correction()

            # get the most recent buffer_size amount of values with a rate of dt_read_buffer, logs all important values for some time
            data_buffer, timestamp_buffer = getRingbufferValues(chunk, timestamps, current_local_time, timestamp_offset, data_buffer, timestamp_buffer) 

            #************************************************************
            #*************************** Data to use ********************
            #************************************************************

            # data_buffer can be used for further processing and classification (maybe use parallel processing to not slow down the data access for longer model prediction times)
            

            # if an error was detected, use the following lines to send the timepoint (timestamp) of detection
            local_clock_time = local_clock()
            #error_index_in_buffer = 500 # calculate the error index in the current data buffer. Arrays timestamp_buffer and data_buffer are related to each other (same data points) 
            #sendDetectedError(team_name, secret_id, timestamp_buffer[error_index_in_buffer, :], local_clock_time)
        

            # wait for some time to ensure a "fixed" frequency to read new data from buffer 
            while((perf_counter()-old_time) < dt_read_buffer): 
                pass

            
            # just for checking the loop frequency 
            #print("time: ", (perf_counter()-old_time)*1000)

            # uncomment to record ALL data received (not required for participants)
            #data_arr = data_arr+chunk
            #time_stamp_arr = time_stamp_arr + timestamps


        old_time = perf_counter()
    


if __name__ == '__main__':
    main()
