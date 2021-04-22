import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import math
import random
from scipy import interpolate



def generate_randomWalk(input_srate = 100.,input_total_Time = 500,rho1  = 1.,sigma = 0.02,mu_e  = 0.,smooth_coeff = 0.5):

    global srate
    srate = float(np.copy(input_srate))
    
    global total_Time
    total_Time = float(np.copy(input_total_Time))
    
    global total_points
    total_points = int(total_Time*srate)
    
    y_coordinate     = np.zeros(total_points)
    x_coordinate     = np.zeros(total_points)

    epsy   = np.random.normal(mu_e,sigma,total_points) 
    epsx   = np.random.normal(mu_e,sigma,total_points) 


    for t in range(2,total_points):
        aux = rho1*y_coordinate[t-1] + epsy[t]
        if aux <= 1 and aux >= 0:
            y_coordinate[t] = aux
        else:
            y_coordinate[t] = y_coordinate[t-1]    

    for t in range(2,total_points):

        aux = rho1*x_coordinate[t-1] + epsx[t]
        if aux <= 1 and aux >= 0:
            x_coordinate[t] = aux
        else:
            x_coordinate[t] = x_coordinate[t-1]    

    x_coordinate = imbf.smooth(np.squeeze(x_coordinate),int(smooth_coeff*srate))
    y_coordinate = imbf.smooth(np.squeeze(y_coordinate),int(smooth_coeff*srate))


    timevector = np.linspace(0,total_points/srate,total_points)
    dt = 1/srate
    speed = np.sqrt(np.diff(x_coordinate)**2 + np.diff(y_coordinate)**2) / dt
    speed = np.hstack([speed,0])
    


    return x_coordinate,y_coordinate,speed,timevector

def generate_randomWalk2(input_srate = 100.,input_total_Time = 500,heading_srate = 10., speed_srate = 5., rho1  = 1.,sigma = 0.02,mu_e  = 0.,smooth_coeff = 0.5):
    
    global srate
    srate = float(np.copy(input_srate))
    
    global total_Time
    total_Time = float(np.copy(input_total_Time))
    
    global total_points
    total_points = int(total_Time*srate)
    

    total_points_head = int(total_Time*heading_srate)
    headings = np.zeros(total_points_head)

    heading_sigma = math.pi/4
    heading_mu = 0
    randomphases = np.random.normal(heading_mu,heading_sigma,total_points_head)

    for t in range(1,total_points_head):
        headings[t] = np.angle(np.exp(1j*(headings[t-1] + randomphases[t-1])))


    y_original = headings
    x_original = np.linspace(0,total_Time,headings.shape[0])
    interpol_func = interpolate.interp1d(x_original,y_original,kind = 'cubic')
    x_new = np.linspace(0,total_Time,total_points)
    headings_new = interpol_func(x_new)


    total_points_spd = int(total_Time*speed_srate)
    speeds = np.zeros(total_points_spd)
    randomspeeds = np.random.exponential(100./srate,total_points_spd)
    for t in range(1,total_points_spd):
        speeds[t] = randomspeeds[t-1]

    y_original = speeds
    x_original = np.linspace(0,total_Time,speeds.shape[0])
    interpol_func = interpolate.interp1d(x_original,y_original,kind = 'cubic')
    x_new = np.linspace(0,total_Time,total_points)
    speeds_new = interpol_func(x_new)


    y_coordinate     = np.zeros(total_points)
    x_coordinate     = np.zeros(total_points)

    epsy   = np.random.normal(mu_e,sigma,total_points) 
    epsx   = np.random.normal(mu_e,sigma,total_points) 

    for t in range(1,total_points):

        y_coordinate[t] = y_coordinate[t-1] + speeds_new[t]*np.sin(headings_new[t]) + rho1*epsy[t]
        x_coordinate[t] = x_coordinate[t-1] + speeds_new[t]*np.cos(headings_new[t]) + rho1*epsx[t]

        if y_coordinate[t] > 100 or y_coordinate[t] < 0 or x_coordinate[t] > 100 or x_coordinate[t] < 0:


            headings_new = headings_new+math.pi

            y_coordinate[t] = y_coordinate[t-1] + speeds_new[t]*np.sin(headings_new[t]) + rho1*epsy[t]
            x_coordinate[t] = x_coordinate[t-1] + speeds_new[t]*np.cos(headings_new[t]) + rho1*epsx[t]



    x_coordinate = imbf.smooth(np.squeeze(x_coordinate),int(smooth_coeff*srate))
    y_coordinate = imbf.smooth(np.squeeze(y_coordinate),int(smooth_coeff*srate))
    
#     x_coordinate = (x_coordinate - np.min(x_coordinate))/(np.max(x_coordinate)-np.min(x_coordinate))
#     y_coordinate = (y_coordinate - np.min(y_coordinate))/(np.max(y_coordinate)-np.min(y_coordinate))

#     timevector = np.linspace(0,total_Time,total_points)
    dt = 1/srate
    speed = np.sqrt(np.diff(x_coordinate)**2 + np.diff(y_coordinate)**2) / dt
    speed = np.hstack([speed,0])
    
    return x_coordinate,y_coordinate,speed,timevector

# def generate_realWalk(RatSession,day,access_point,dataset):
    

#     if access_point == 'home':
#         path = '/home/atila/Documents/DataAnalysis/TK_projects/Sync/'
#     elif access_point == 'mpi_server':
#         path = '/beegfs/v1/korotkova_group/matlab_scripts/Rob/Calcium_Imaging/'
#     elif access_point == 'cheops_server':
#         path = '/home/rscheffe/CalciumImaging/'
    
    
#     datafolder = path + '/data/' + dataset + '/' + RatSession + '/clean/'
#     os.chdir(datafolder)    
    
#     filename = RatSession + '.' + dataset + '.Coordinates.Day' + str(day)
#     output = np.load(filename,allow_pickle=True).item()

#     x_coordinates = output['x_coordinates']
#     y_coordinates = output['y_coordinates']
#     track_timevector = output['track_timevector']
#     mean_video_srate = output['mean_video_srate']

#     dt = np.diff(track_timevector)
#     speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2) / dt
#     speed = np.hstack([speed,0])
    
    
#     global srate
#     srate = np.copy(mean_video_srate)
    
    
#     global total_Time
#     total_Time = float(np.copy(len(speed)/srate))
    
    
#     global total_points
#     total_points = int(total_Time*srate)
    
    
#     return x_coordinates,y_coordinates,speed,track_timevector,mean_video_srate


def generate_arrivals(_lambda = 0.5,total_Time=100):

    _num_arrivals = int(_lambda*total_Time)
    _arrival_time = 0
    All_arrival_time = []

    for i in range(_num_arrivals):
        #Get the next probability value from Uniform(0,1)
        p = random.random()

        #Plug it into the inverse of the CDF of Exponential(_lamnbda)
        _inter_arrival_time = -math.log(1.0 - p)/_lambda

        #Add the inter-arrival time to the running sum
        _arrival_time = _arrival_time + _inter_arrival_time
        All_arrival_time.append(_arrival_time)
    All_arrival_time = np.array(All_arrival_time)
    All_arrival_time = All_arrival_time[All_arrival_time < total_Time]
    I_timestamps = (All_arrival_time*srate).astype(int)

    return All_arrival_time,I_timestamps

def get_bins_edges(x_coordinate,y_coordinate,x_nbins,y_nbins):
    
    x_bins = np.linspace(np.nanmin(x_coordinate),np.nanmax(x_coordinate),nbins_x)
    y_bins = np.linspace(np.nanmin(y_coordinate),np.nanmax(y_coordinate),nbins_y)
    
    return x_bins,y_bins

def gaussian_kernel_2d(x_coordinate,y_coordinate,nbins_x=100,nbins_y=100,x_center = 0.5,y_center = 0.5, s = 0.1):
    x_bins,y_bins = get_bins_edges(x_coordinate,y_coordinate,x_nbins,y_nbins)
    
    gaussian_kernel = np.zeros([ybins.shape[0],xbins.shape[0]])
    x_count = 0
    for xx in x_bins:
        y_count = 0
        for yy in y_bins:
            gaussian_kernel[y_count,x_count] = np.exp(-(((xx - x_center)**2 + (yy-y_center)**2)/(2*(s**2))))
            y_count += 1
        x_count += 1
    
    return gaussian_kernel


def digitize_spiketimes(x_coordinate,y_coordinate,I_timestamps,x_nbins=100,y_nbins=100,x_center = 0.5,y_center = 0.5, s = 0.1):
    
    x_bins,y_bins = get_bins_edges(x_coordinate,y_coordinate,x_nbins,y_nbins)

    x_digitized = np.digitize(x_coordinate[I_timestamps],x_bins)-1
    y_digitized = np.digitize(y_coordinate[I_timestamps],y_bins)-1

    gaussian_kernel = gaussian_kernel_2d(x_coordinate,y_coordinate,x_nbins,y_nbins,x_center,y_center, s)
    
    modulated_timestamps = []
    for spk in range(0,x_digitized.shape[0]):
        random_number = random.choices([0,1], [1-gaussian_kernel[y_digitized[spk],x_digitized[spk]],gaussian_kernel[y_digitized[spk],x_digitized[spk]]])[0]
        if random_number == 1:
            modulated_timestamps.append(I_timestamps[spk])
    modulated_timestamps = np.array(modulated_timestamps)
    return modulated_timestamps


def generate_CalciumSignal(modulated_timestamps,noise_level = 0.01, b = 5.):

    dt = 1/srate
    timevector = np.linspace(0,total_points/srate,total_points)

    I_pf_timestamps = (modulated_timestamps).astype(int)

    All_arrival_continuous = np.zeros(timevector.shape[0])
    All_arrival_continuous[I_pf_timestamps] = 1
    a = 1.
    
    x = np.arange(0,5,dt)
    kernel = a*np.exp(-b*x)

    calcium_imag = np.convolve(kernel,All_arrival_continuous,mode='full')
    calcium_imag = np.copy(calcium_imag[0:total_points])
    calcium_imag = calcium_imag + noise_level*np.random.normal(0,1,calcium_imag.shape)
    return calcium_imag,timevector


def just_testing():
    
    print(srate)
    
    return 1