import numpy as np
from scipy.io import loadmat
import pandas as pd
import os
from scipy import stats as stats
import helper_functions as hf
import detect_peaks as dp
from joblib import Parallel, delayed


        
class PlaceCell:
    def __init__(self,**kwargs):
           
        kwargs.setdefault('RatSession', [])  
        kwargs.setdefault('day', [])  
        kwargs.setdefault('ch', [])  
        kwargs.setdefault('dataset', [])  
        kwargs.setdefault('mean_video_srate', 30.)  
        kwargs.setdefault('mintimespent', 0.1)  
        kwargs.setdefault('minvisits', 1)  
        kwargs.setdefault('speed_threshold', 2.5)  
        kwargs.setdefault('nbins_pos_x', 10)  
        kwargs.setdefault('nbins_pos_y', 10)  
        kwargs.setdefault('nbins_cal', 10)  
        kwargs.setdefault('placefield_nbins_pos_x', 50)  
        kwargs.setdefault('placefield_nbins_pos_y', 50)  
        kwargs.setdefault('num_cores', 1)  
        kwargs.setdefault('num_surrogates', 200)          
        kwargs.setdefault('saving_path', [])  
        kwargs.setdefault('saving', False)  
        kwargs.setdefault('saving_string', [])             

        valid_kwargs = ['RatSession','day','ch','dataset', 'mean_video_srate','mintimespent', 'minvisits', 'speed_threshold', 'nbins_pos_x', 'nbins_pos_y',                                 'nbins_cal', 'placefield_nbins_pos_x','placefield_nbins_pos_y','num_cores','num_surrogates','saving_path','saving','saving_string']
        
        for k, v in kwargs.items():
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            setattr(self, k, v)

        
    def run_placeMetrics(self,mean_calcium_to_behavior,track_timevector,x_coordinates,y_coordinates):

        self.mean_calcium_to_behavior = mean_calcium_to_behavior
        self.track_timevector = track_timevector
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.speed = self.get_speed(self.x_coordinates,self.y_coordinates,self.track_timevector)
        
        I_peaks = dp.detect_peaks(self.mean_calcium_to_behavior,mpd=0.5*self.mean_video_srate,mph=1.*np.nanstd(self.mean_calcium_to_behavior))

        x_coordinates_valid, y_coordinates_valid, mean_calcium_to_behavior_valid, track_timevector_valid = self.get_valid_timepoints(mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,self.speed_threshold)

        calcium_signal_binned_signal = self.get_binned_signal(mean_calcium_to_behavior_valid,self.nbins_cal)

        x_grid,y_grid,x_center_bins,y_center_bins = self.get_position_grid(x_coordinates,y_coordinates,self.nbins_pos_x,self.nbins_pos_y)

        position_binned = self.get_binned_2Dposition(x_coordinates_valid,y_coordinates_valid,self.nbins_pos_x,self.nbins_pos_y)

        self.nbins_pos = self.nbins_pos_x*self.nbins_pos_y
        
        mutualInfo_original = self.mutualInformation(position_binned,calcium_signal_binned_signal,self.nbins_pos,self.nbins_cal)
        
        mutualInfo_permutation = self.get_perm_distribution(position_binned,calcium_signal_binned_signal,self.nbins_pos,self.nbins_cal,self.num_cores,self.num_surrogates)

        mutualInfo_zscored = self.get_mutualInfo_zscore(mutualInfo_original,mutualInfo_permutation)
        
        calcium_mean_occupancy,calcium_mean_occupancy_smoothed,position_occupancy,visits_occupancy,x_grid_pc,y_grid_pc,x_center_bins_pc,y_center_bins_pc = self.placeField(mean_calcium_to_behavior_valid,x_coordinates_valid,y_coordinates_valid,track_timevector_valid,self.mean_video_srate,self.mintimespent,self.minvisits,self.placefield_nbins_pos_x,self.placefield_nbins_pos_y)
        

        inputdict = dict()
        inputdict['signalMap'] = calcium_mean_occupancy
        inputdict['signalMapSmoothed'] = calcium_mean_occupancy_smoothed
        inputdict['ocuppancyMap'] = position_occupancy
        inputdict['x_grid'] = x_grid_pc
        inputdict['y_grid'] = y_grid_pc
        inputdict['x_center_bins'] = x_center_bins_pc
        inputdict['y_center_bins'] = y_center_bins_pc          
        inputdict['numb_events'] = I_peaks.shape[0]
        inputdict['events_index'] = I_peaks
        inputdict['mutualInfo_original'] = mutualInfo_original
        inputdict['mutualInfo_zscored'] = mutualInfo_zscored
        inputdict['mutualInfo_permutation'] = mutualInfo_permutation

        self.caller_saving(inputdict,self.saving)

        return inputdict


    def caller_saving(self,inputdict,saving):
        if saving == True:
            print('Saving file...')
            os.chdir(saving_path)
            filename = RatSession + '.' + saving_string + '.PlaceField.ModulationIndex.' + dataset + '.Day' + str(day) + '.Ch.' + str(ch)
            output = open(filename, 'wb') 
            np.save(output,inputdict)
            output.close()
        else:
            print('File not saved!')


    def get_speed(self,x_coordinates,y_coordinates,track_timevector):

        speed = np.sqrt(np.diff(x_coordinates)**2 + np.diff(y_coordinates)**2)
        speed = hf.smooth(speed/np.diff(track_timevector),window_len=10)
        speed = np.hstack([speed,0])
        return speed



    def get_position_grid(self,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y):
        # here someone should also be able to set the enviroment edges

        x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
        x_grid_window = x_range/nbins_pos_x
        x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) +x_grid_window/2,x_grid_window)

        y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
        y_grid_window = y_range/nbins_pos_y
        y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates)+y_grid_window/2,y_grid_window)

        x_center_bins = x_grid + x_grid_window/2
        y_center_bins = y_grid + y_grid_window/2

        return x_grid,y_grid,x_center_bins,y_center_bins


    def get_occupancy(self,x_coordinates_speed,y_coordinates_speed,x_grid,y_grid,mean_video_srate):
        # calculate position occupancy
        position_occupancy = np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1))
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

                position_occupancy[yy,xx] = np.sum(np.logical_and(check_x_ocuppancy,check_y_ocuppancy))/mean_video_srate

        return position_occupancy


    def get_calcium_occupancy(self,mean_calcium_to_behavior_speed,x_coordinates_speed,y_coordinates_speed,x_grid,y_grid):

        # calculate mean calcium per pixel
        calcium_mean_occupancy = np.nan*np.zeros((y_grid.shape[0]-1,x_grid.shape[0]-1)) 
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates_speed >= x_grid[xx],x_coordinates_speed < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates_speed >= y_grid[yy],y_coordinates_speed < (y_grid[yy+1]))

                calcium_mean_occupancy[yy,xx] = np.nanmean(mean_calcium_to_behavior_speed[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)])

        return calcium_mean_occupancy


    def get_visits(self,x_coordinates,y_coordinates,x_grid,y_grid,x_center_bins,y_center_bins):

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

            return visits_occupancy

        
    def get_valid_timepoints(self,mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,speed_threshold):
        
        speed = self.get_speed(x_coordinates,y_coordinates,track_timevector)

        I_speed_thres = speed > speed_threshold

        mean_calcium_to_behavior_valid = mean_calcium_to_behavior[I_speed_thres].copy()
        x_coordinates_valid = x_coordinates[I_speed_thres].copy()
        y_coordinates_valid = y_coordinates[I_speed_thres].copy()
        track_timevector_valid = track_timevector[I_speed_thres].copy()
        
        
        return x_coordinates_valid, y_coordinates_valid, mean_calcium_to_behavior_valid, track_timevector_valid
  
    def placeField(self,mean_calcium_to_behavior,x_coordinates,y_coordinates,track_timevector,mean_video_srate,mintimespent, minvisits,placefield_nbins_pos_x,placefield_nbins_pos_y):


        x_grid,y_grid,x_center_bins,y_center_bins = self.get_position_grid(x_coordinates,y_coordinates,placefield_nbins_pos_x,placefield_nbins_pos_y)

        position_occupancy = self.get_occupancy(x_coordinates,y_coordinates,x_grid,y_grid,mean_video_srate)

        calcium_mean_occupancy = self.get_calcium_occupancy(mean_calcium_to_behavior,x_coordinates,y_coordinates,x_grid,y_grid)

        visits_occupancy = self.get_visits(x_coordinates,y_coordinates,x_grid,y_grid,x_center_bins,y_center_bins)


        Valid=(position_occupancy>=mintimespent)*(visits_occupancy>=minvisits)*1.
        Valid[Valid == 0] = np.nan
        calcium_mean_occupancy = calcium_mean_occupancy*Valid

        calcium_mean_occupancy_to_smooth = np.copy(calcium_mean_occupancy)
        calcium_mean_occupancy_to_smooth[np.isnan(calcium_mean_occupancy_to_smooth)] = 0 
        calcium_mean_occupancy_smoothed = hf.gaussian_smooth_2d(calcium_mean_occupancy_to_smooth,2)

        return calcium_mean_occupancy,calcium_mean_occupancy_smoothed,position_occupancy,visits_occupancy,x_grid,y_grid,x_center_bins,y_center_bins



    def get_binned_2Dposition(self,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y):

        x_grid,y_grid,x_center_bins,y_center_bins = self.get_position_grid(x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y)

        # calculate position occupancy
        position_binned = np.zeros(x_coordinates.shape) 
        count = 0
        for xx in range(0,x_grid.shape[0]-1):
            for yy in range(0,y_grid.shape[0]-1):

                check_x_ocuppancy = np.logical_and(x_coordinates >= x_grid[xx],x_coordinates < (x_grid[xx+1]))
                check_y_ocuppancy = np.logical_and(y_coordinates >= y_grid[yy],y_coordinates < (y_grid[yy+1]))


                position_binned[np.logical_and(check_x_ocuppancy,check_y_ocuppancy)] = count
                count += 1

                
        return position_binned
    

    def get_binned_signal(self,mean_calcium_to_behavior,nbins_cal):

        calcium_signal_bins = np.linspace(np.nanmin(mean_calcium_to_behavior),np.nanmax(mean_calcium_to_behavior),nbins_cal+1)
        calcium_signal_binned_signal = np.zeros(mean_calcium_to_behavior.shape[0])
        for jj in range(calcium_signal_bins.shape[0]-1):
            I_amp = (mean_calcium_to_behavior > calcium_signal_bins[jj]) & (mean_calcium_to_behavior <= calcium_signal_bins[jj+1])
            calcium_signal_binned_signal[I_amp] = jj

        return calcium_signal_binned_signal


    def mutualInformation(self,bin_vector1,bin_vector2,nbins_1,nbins_2):
        eps = np.finfo(float).eps

        entropy1 = self.get_entropy(bin_vector1,nbins_1)
        entropy2 = self.get_entropy(bin_vector2,nbins_2)

    #     this part here could be done using this code instead. I will leave both for clarity
    #     nbins_pos = 100
    #     edges1 = np.linspace(np.nanmin(position_binned),np.nanmax(position_binned),nbins_pos+1)
    #     bin_vector1 = np.digitize(position_binned,edges1)

    #     nbins_cal = 10
    #     edges2 = np.linspace(np.nanmin(calcium_signal_binned_signal),np.nanmax(calcium_signal_binned_signal),nbins_cal+1)
    #     bin_vector2 = np.digitize(calcium_signal_binned_signal,edges2)-1

        joint_entropy = self.get_joint_entropy(bin_vector1,bin_vector2,nbins_1,nbins_2)
        mutualInfo = entropy1 + entropy2 - joint_entropy

        return mutualInfo

    def get_mutualInfo_zscore(self,mutualInfo_original,mutualInfo_permutation):
        mutualInfo_zscored = (mutualInfo_original-np.nanmean(mutualInfo_permutation))/np.nanstd(mutualInfo_permutation)
        return mutualInfo_zscored

    def get_perm_distribution(self,bin_vector1,bin_vector2,nbins_1,nbins_2,num_cores,num_surrogates):
        
        results = Parallel(n_jobs=num_cores)(delayed(self.get_surrogate)(bin_vector1,bin_vector2,nbins_1,nbins_2,permi) for permi in range(num_surrogates))
        
        return np.array(results)
    
    def get_joint_entropy(self,bin_vector1,bin_vector2,nbins_1,nbins_2):

        eps = np.finfo(float).eps

        bin_vector1 = np.copy(bin_vector1)
        bin_vector2 = np.copy(bin_vector2)

        jointprobs = np.zeros([nbins_1,nbins_2])
        
        for i1 in range(nbins_1):
            for i2 in range(nbins_2):
                jointprobs[i1,i2] = np.nansum((bin_vector1==i1) & (bin_vector2==i2))

        jointprobs = jointprobs/np.nansum(jointprobs)
        joint_entropy = -np.nansum(jointprobs*np.log2(jointprobs+eps));

        return joint_entropy
    
    
    
    def get_entropy(self,binned_input,nbins):

        eps = np.finfo(float).eps

        hdat = np.histogram(binned_input,nbins)[0]
        hdat = hdat/np.nansum(hdat)
        entropy = -np.nansum(hdat*np.log2(hdat+eps))

        return entropy

    
    def get_surrogate(self,bin_vector1,bin_vector2,nbins_1,nbins_2,permi):
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


        mutualInfo = self.mutualInformation(bin_vector1,bin_vector_shuffled,nbins_1,nbins_2)


        return mutualInfo
