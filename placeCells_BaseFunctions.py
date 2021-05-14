import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
import helper_functions as hf
import detect_peaks as dp
from joblib import Parallel, delayed

# to implement
# def get_speed
# def get_position_binned
# def get_occupancy_map
# def get_visits_map

def run_placeMetrics(RatSession,day,ch,saving_path,dataset,mean_calcium_to_behavior,track_timevector,x_coordinates,y_coordinates,mean_video_srate,mintimespent, minvisits,speed_threshold,nbins_pos_x,nbins_pos_y,nbins_cal,placefield_nbins_pos_x,placefield_nbins_pos_y,num_cores,num_surrogates,saving = False,saving_string = 'CI'):
    
    I_peaks = dp.detect_peaks(mean_calcium_to_behavior,mpd=0.5*mean_video_srate,mph=1.*np.nanstd(mean_calcium_to_behavior))
    
    calcium_mean_occupancy,calcium_mean_occupancy_smoothed,x_grid,y_grid,position_occupancy,visits_occupancy,speed = placeField(track_timevector,x_coordinates,y_coordinates,mean_calcium_to_behavior,mean_video_srate,mintimespent, minvisits,speed_threshold,placefield_nbins_pos_x,placefield_nbins_pos_y)


    calcium_signal_binned_signal,position_binned = get_binned_signals(mean_calcium_to_behavior,x_coordinates,y_coordinates,speed,speed_threshold,nbins_pos_x,nbins_pos_y,nbins_cal)

    
    mutualInfo,entropy1,entropy2,joint_entropy,mutualInfo_distance,perm_mutual_information = mutualInformation(calcium_signal_binned_signal,position_binned,num_surrogates,num_cores,nbins_pos_x,nbins_pos_y,nbins_cal)


    
    inputdict = dict()
    inputdict['signalMap'] = calcium_mean_occupancy
    inputdict['signalMapSmoothed'] = calcium_mean_occupancy_smoothed
    inputdict['ocuppancyMap'] = position_occupancy
    inputdict['x_grid'] = x_grid
    inputdict['y_grid'] = y_grid
    inputdict['numb_events'] = I_peaks.shape[0]
    inputdict['events_index'] = I_peaks
    inputdict['MutualInfo'] = mutualInfo
    inputdict['MutualInfo_Zscored'] = mutualInfo_distance
    inputdict['Shuffled_MutualInfo'] = perm_mutual_information
    

    
    
    if saving == True:
        
        print('Saving file...')
        os.chdir(saving_path)
        filename = RatSession + '.' + saving_string + '.PlaceField.ModulationIndex.' + dataset + '.Day' + str(day) + '.Ch.' + str(ch)
        output = open(filename, 'wb') 
        np.save(output,inputdict)
        output.close()
    else:
        print('File nor saved!')
        
    return inputdict

def get_speed(x_coordinates,y_coordinates,track_timevector):
    
    speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
    speed = hf.smooth(speed/np.diff(track_timevector),window_len=10)
    speed = np.hstack([speed,0])
    return speed



def placeField(track_timevector,x_coordinates,y_coordinates,mean_calcium_to_behavior,mean_video_srate,mintimespent, minvisits,speed_threshold,nbins_pos_x,nbins_pos_y):

    speed = get_speed(x_coordinates,y_coordinates,track_timevector)

    I_speed_thres = speed > speed_threshold

    mean_calcium_to_behavior_speed = mean_calcium_to_behavior[I_speed_thres]
    x_coordinates_speed = x_coordinates[I_speed_thres]
    y_coordinates_speed = y_coordinates[I_speed_thres]
    
    x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
    x_grid_window = x_range/nbins_pos_x
    x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) +x_grid_window/2,x_grid_window)
    
    y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
    y_grid_window = y_range/nbins_pos_y
    y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates)+y_grid_window/2,y_grid_window)

    # calculate position occupancy
    position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))   
    for xx in range(0,x_grid.shape[0]-1):
        for yy in range(0,y_grid.shape[0]-1):

            check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
            check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

            position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/mean_video_srate

            
    # calculate mean calcium per pixel
    calcium_mean_occupancy = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
    for xx in range(0,x_grid.shape[0]-1):
        for yy in range(0,y_grid.shape[0]-1):

            check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
            check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

            calcium_mean_occupancy[yy,xx] = np.nanmean(mean_calcium_to_behavior_speed[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)])

            
    x_center_bins = x_grid + x_grid_window/2
    y_center_bins = y_grid + y_grid_window/2

    I_x_coord = []
    I_y_coord = []

    for xx in range(0,x_coordinates.shape[0]):
        I_x_coord.append(np.argmin(np.abs(x_coordinates[xx] - x_center_bins)))
        I_y_coord.append(np.argmin(np.abs(y_coordinates[xx] - y_center_bins)))

    I_x_coord = np.array(I_x_coord)
    I_y_coord = np.array(I_y_coord)

    dx = np.diff(np.hstack([I_x_coord[0]-1,I_x_coord]))
    dy = np.diff(np.hstack([I_y_coord[0]-1,I_y_coord]))

    newvisitstimes = (-1*(dy == 0))*(dx==0)+1
    newvisitstimes2 = (np.logical_or((dy != 0), (dx!=0))*1)

    I_visit = np.where(newvisitstimes>0)[0]

    # calculate visits


    x_coordinate_visit = x_coordinates[I_visit]
    y_coordinate_visit = y_coordinates[I_visit]

    visits_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))        
    for xx in range(0,x_grid.shape[0]-1):
        for yy in range(0,y_grid.shape[0]-1):

            check_x_ocuppancy = np.logical_and(x_coordinate_visit >= x_grid[xx],x_coordinate_visit < (x_grid[xx+1]))
            check_y_ocuppancy = np.logical_and(y_coordinate_visit >= y_grid[yy],y_coordinate_visit < (y_grid[yy+1]))

            visits_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))


    Valid=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
    Valid[Valid == 0] = np.nan
    calcium_mean_occupancy = calcium_mean_occupancy*Valid

    calcium_mean_occupancy_to_smooth = np.copy(calcium_mean_occupancy)
    calcium_mean_occupancy_to_smooth[np.isnan(calcium_mean_occupancy_to_smooth)] = 0 
    calcium_mean_occupancy_smoothed = hf.gaussian_smooth_2d(calcium_mean_occupancy_to_smooth,2)

    return calcium_mean_occupancy,calcium_mean_occupancy_smoothed,x_grid,y_grid,position_occupancy,visits_occupancy,speed




def get_binned_signals(mean_calcium_to_behavior,x_coordinates,y_coordinates,speed,speed_threshold,nbins_pos_x,nbins_pos_y,nbins_cal):

    I_speed_thres = speed >= speed_threshold

    mean_calcium_to_behavior_speed = np.copy(mean_calcium_to_behavior[I_speed_thres])
    x_coordinates_speed = np.copy(x_coordinates[I_speed_thres])
    y_coordinates_speed = np.copy(y_coordinates[I_speed_thres])

    
#     this part here could be done using this code instead. I will leave both for clarity
#     nbins_cal = 10
#     edges = np.linspace(np.nanmin(mean_calcium_to_behavior),np.nanmax(mean_calcium_to_behavior),nbins_cal)
#     bin_vector = np.digitize(mean_calcium_to_behavior,edges)-1

#     calcium_signal_bins = np.arange(0,1+1/nbins_cal,1/nbins_cal)
    calcium_signal_bins = np.linspace(np.nanmin(mean_calcium_to_behavior_speed),np.nanmax(mean_calcium_to_behavior_speed),nbins_cal+1)
    calcium_signal_binned_signal = np.zeros(mean_calcium_to_behavior_speed.shape[0])
    for jj in range(calcium_signal_bins.shape[0]-1):
        I_amp = (mean_calcium_to_behavior_speed > calcium_signal_bins[jj]) & (mean_calcium_to_behavior_speed <= calcium_signal_bins[jj+1])
        calcium_signal_binned_signal[I_amp] = jj

    
    x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
    x_grid_window = x_range/nbins_pos_x
    x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) +x_grid_window/2,x_grid_window)
    
    y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
    y_grid_window = y_range/nbins_pos_y
    y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates)+y_grid_window/2,y_grid_window)


    # calculate position occupancy
    position_binned = np.zeros(x_coordinates_speed.shape) 
    count = 0
    for xx in range(0,x_grid.shape[0]-1):
        for yy in range(0,y_grid.shape[0]-1):

            check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
            check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

            
            position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
            count += 1

    return calcium_signal_binned_signal,position_binned

def mutualInformation(calcium_signal_binned_signal,position_binned,num_surrogates,num_cores,nbins_pos_x,nbins_pos_y,nbins_cal):
    
    eps = np.finfo(float).eps
    
    nbins_pos = nbins_pos_x*nbins_pos_y
    hdat1 = np.histogram(position_binned,nbins_pos)[0]
    hdat1 = hdat1/np.nansum(hdat1)
    entropy1 = -np.nansum(hdat1*np.log2(hdat1+eps))

    hdat2 = np.histogram(calcium_signal_binned_signal,nbins_cal)[0]
    hdat2 = hdat2/np.nansum(hdat2)
    entropy2 = -np.nansum(hdat2*np.log2(hdat2+eps))

#     this part here could be done using this code instead. I will leave both for clarity
#     nbins_pos = 100
#     edges1 = np.linspace(np.nanmin(position_binned),np.nanmax(position_binned),nbins_pos+1)
#     bin_vector1 = np.digitize(position_binned,edges1)

#     nbins_cal = 10
#     edges2 = np.linspace(np.nanmin(calcium_signal_binned_signal),np.nanmax(calcium_signal_binned_signal),nbins_cal+1)
#     bin_vector2 = np.digitize(calcium_signal_binned_signal,edges2)-1


    
    bin_vector1 = np.copy(position_binned)
    bin_vector2 = np.copy(calcium_signal_binned_signal)

    jointprobs = np.zeros([nbins_pos,nbins_cal])
    for i1 in range(nbins_pos):
        for i2 in range(nbins_cal):
            jointprobs[i1,i2] = np.nansum((bin_vector1==i1) & (bin_vector2==i2))

    jointprobs = jointprobs/np.nansum(jointprobs)
    joint_entropy = -np.nansum(jointprobs*np.log2(jointprobs+eps));

    mutualInfo = entropy1 + entropy2 - joint_entropy
    
    results = Parallel(n_jobs=num_cores)(delayed(processInput_mutualInformation)(bin_vector1,bin_vector2,entropy1,entropy2,permi,nbins_pos,nbins_cal) for permi in range(num_surrogates))
    perm_mutual_information = np.array(results)
    
    mutualInfo_distance = (mutualInfo-np.nanmean(perm_mutual_information))/np.nanstd(perm_mutual_information)

    return mutualInfo,entropy1,entropy2,joint_entropy,mutualInfo_distance,perm_mutual_information



def processInput_mutualInformation(bin_vector1,bin_vector2,entropy1,entropy2,permi,nbins_pos,nbins_cal):
    eps = np.finfo(float).eps
    
    bin_vector_shuffled = []
    I_break = np.random.choice(np.arange(int(bin_vector2.shape[0]*0.1),int(bin_vector2.shape[0]*0.9)),1)[0].astype(int)

    if np.mod(permi,4) == 0:
        bin_vector_shuffled = np.concatenate([bin_vector2[I_break:], bin_vector2[0:I_break]])
    elif np.mod(permi,4) == 1:
        bin_vector_shuffled = np.concatenate([bin_vector2[:I_break:-1], bin_vector2[0:I_break+1]])
    elif np.mod(permi,4) == 2:
        bin_vector_shuffled = np.concatenate([bin_vector2[I_break:], bin_vector2[I_break-1::-1]])
    else:   
        bin_vector_shuffled = np.concatenate([bin_vector2[I_break:], bin_vector2[0:I_break]])
        bin_vector_shuffled = bin_vector_shuffled[::-1]


    jointprobs = np.zeros([nbins_pos,nbins_cal])
    for i1 in range(nbins_pos):
        for i2 in range(nbins_cal):
            jointprobs[i1,i2] = np.nansum((bin_vector_shuffled==i2) & (bin_vector1==i1))

    jointprobs = jointprobs/np.nansum(jointprobs)
    perm_joint_entropy = -np.nansum(jointprobs*np.log2(jointprobs+eps))
    perm_mutual_information = entropy1 + entropy2 - perm_joint_entropy

    return perm_mutual_information
