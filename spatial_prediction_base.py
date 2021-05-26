import numpy as np
from scipy.io import loadmat
import os
from scipy import stats as stats
from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB

import numpy as np
from scipy.io import loadmat
import os
from scipy import stats as stats
from joblib import Parallel, delayed
from sklearn.naive_bayes import GaussianNB


def processInput_NaiveRuns(Input_Variable_Cell,Target_Variable,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y,num_surrogates,prior_dist):

    eps = np.finfo(float).eps

    xbins = nbins_pos_x
    ybins = nbins_pos_y
    
#     x_grid_window = 1/xbins
#     x_grid = np.arange(0,1+x_grid_window,x_grid_window)

#     y_grid_window = 1/ybins
#     y_grid = np.arange(0,1+y_grid_window,y_grid_window)
    
    
    x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
    x_grid_window = x_range/nbins_pos_x
    x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) + x_grid_window/2,x_grid_window)
    
    y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
    y_grid_window = y_range/nbins_pos_y
    y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates) + y_grid_window/2,y_grid_window)

    
    x_center = x_grid[0:-1] + np.diff(x_grid)/2
    y_center = y_grid[0:-1] + np.diff(y_grid)/2
    
    bin_x_centers = np.repeat(x_center,ybins)
    bin_y_centers = np.tile(y_center,xbins)


    
    I_rand = np.random.choice(range(int(Input_Variable_Cell.shape[0]*0.9)))
    offset = int(0.1*Input_Variable_Cell.shape[0])
    Trials_testing_set =  np.arange(I_rand,I_rand+offset).astype(int)
    Trials_training_set = np.delete(range(Input_Variable_Cell.shape[0]),Trials_testing_set)

    X_train = Input_Variable_Cell[Trials_training_set,:].copy()
    y_train = Target_Variable[Trials_training_set].copy()

    X_test = Input_Variable_Cell[Trials_testing_set,:].copy()
    y_test = Target_Variable[Trials_testing_set].copy()

    
    priors_in = np.ones(np.unique(y_train).shape[0])/np.unique(y_train).shape[0]

    gnb = GaussianNB(priors = priors_in)
    gnb.fit(X_train, y_train)

    accuracy_original = gnb.score(X_test, y_test)


    y_pred = gnb.predict(X_test)
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)


    diffx = (bin_x_centers[y_pred]-x_coordinates[Trials_testing_set])**2
    diffy = (bin_y_centers[y_pred]-y_coordinates[Trials_testing_set])**2

    concatenated_nearest_dist_to_predicted = np.nanmean(np.sqrt(diffx + diffy))
                
    pred_dist_grid_original = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
    count = 0
    for xx in range(xbins):
        for yy in range(ybins):

            I_test = y_test == count
            if np.any(I_test):
                I_pred = y_pred[I_test].astype(int)

                diffx = (bin_x_centers[I_pred]-x_coordinates[Trials_testing_set][I_test])**2
                diffy = (bin_y_centers[I_pred]-y_coordinates[Trials_testing_set][I_test])**2

                nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                pred_dist_grid_original[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

            count += 1 

            
            
    predict_proba = gnb.predict_proba(X_test)
    mean_error_distance = []
    for fr in range(predict_proba.shape[0]):
        predic_distr_singleTime = predict_proba[fr,:]

        prob_grid = np.zeros((x_grid.shape[0]-1)*(y_grid.shape[0]-1))*np.nan
        prob_grid[gnb.classes_.astype(int)] = predict_proba[fr,:]

        mass_x_coord = np.nansum(prob_grid*bin_x_centers)
        mass_y_coord = np.nansum(prob_grid*bin_y_centers)

        error_distance = np.sqrt((mass_x_coord - x_coordinates[Trials_testing_set][fr])**2 +  (mass_y_coord - y_coordinates[Trials_testing_set][fr])**2)
        mean_error_distance.append(error_distance)
    mean_error_distance = np.nanmean(mean_error_distance)            
             

    pred_dist_grid_surrogates = []
    accuracy_surrogates = []
    concatenated_nearest_dist_to_predicted_perm = []
    mean_error_distance_perm = []
    for _ in range(num_surrogates):
        
        Input_Variable_Cell_Shuffled = []

        I_break = np.random.choice(np.arange(int(Input_Variable_Cell.shape[0]*0.1),int(Input_Variable_Cell.shape[0]*0.9)),1)[0].astype(int)

        if np.mod(_,4) == 0:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
        elif np.mod(_,4) == 1:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[:I_break:-1], Input_Variable_Cell[0:I_break+1]])
        elif np.mod(_,4) == 2:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[I_break-1::-1]])
        else:   
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
            Input_Variable_Cell_Shuffled = Input_Variable_Cell_Shuffled[::-1]

    

        X_train = Input_Variable_Cell_Shuffled[Trials_training_set,:].copy()
        y_train = Target_Variable[Trials_training_set].copy()

        X_test = Input_Variable_Cell_Shuffled[Trials_testing_set,:].copy()
        y_test = Target_Variable[Trials_testing_set].copy()

    
        gnb = GaussianNB(priors = priors_in)
        gnb.fit(X_train, y_train)

        accuracy = gnb.score(X_test, y_test)

        y_pred = gnb.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)

        diffx = (bin_x_centers[y_pred]-x_coordinates[Trials_testing_set])**2
        diffy = (bin_y_centers[y_pred]-y_coordinates[Trials_testing_set])**2

        concatenated_nearest_dist_to_predicted_perm.append(np.nanmean(np.sqrt(diffx + diffy)))
    
        pred_dist_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):

                I_test = y_test == count
                if np.any(I_test):
                    I_pred = y_pred[I_test].astype(int)

                    diffx = (bin_x_centers[I_pred]-x_coordinates[Trials_testing_set][I_test])**2
                    diffy = (bin_y_centers[I_pred]-y_coordinates[Trials_testing_set][I_test])**2

                    nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                    pred_dist_grid[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

                count += 1

        accuracy_surrogates.append(accuracy)
        pred_dist_grid_surrogates.append(pred_dist_grid)
        
        predict_proba = gnb.predict_proba(X_test)
        mean_error_distance_surr = []
        for fr in range(predict_proba.shape[0]):
            predic_distr_singleTime = predict_proba[fr,:]

            prob_grid = np.zeros((x_grid.shape[0]-1)*(y_grid.shape[0]-1))*np.nan
            prob_grid[gnb.classes_.astype(int)] = predict_proba[fr,:]

            mass_x_coord = np.nansum(prob_grid*bin_x_centers)
            mass_y_coord = np.nansum(prob_grid*bin_y_centers)

            error_distance = np.sqrt((mass_x_coord - x_coordinates[Trials_testing_set][fr])**2 +  (mass_y_coord - y_coordinates[Trials_testing_set][fr])**2)
            mean_error_distance_surr.append(error_distance)
            
        mean_error_distance_surr = np.nanmean(mean_error_distance_surr)    
        mean_error_distance_perm.append(mean_error_distance_surr)
        
    return pred_dist_grid_original,accuracy_original,pred_dist_grid_surrogates,accuracy_surrogates,x_coordinates[Trials_testing_set],y_coordinates[Trials_testing_set],concatenated_nearest_dist_to_predicted,concatenated_nearest_dist_to_predicted_perm,mean_error_distance,mean_error_distance_perm



def processInput_NaiveRuns_10fold(Input_Variable_Cell,Target_Variable,x_coordinates,y_coordinates,nbins_pos_x,nbins_pos_y,num_surrogates,fold,prior_dist):

    eps = np.finfo(float).eps

    xbins = nbins_pos_x
    ybins = nbins_pos_y
    
    x_range = (np.nanmax(x_coordinates) - np.nanmin(x_coordinates))
    x_grid_window = x_range/nbins_pos_x
    x_grid = np.arange(np.nanmin(x_coordinates),np.nanmax(x_coordinates) + x_grid_window/2,x_grid_window)
    
    y_range = (np.nanmax(y_coordinates) - np.nanmin(y_coordinates))
    y_grid_window = y_range/nbins_pos_y
    y_grid = np.arange(np.nanmin(y_coordinates),np.nanmax(y_coordinates) + y_grid_window/2,y_grid_window)

    
    x_center = x_grid[0:-1] + np.diff(x_grid)/2
    y_center = y_grid[0:-1] + np.diff(y_grid)/2
    
    bin_x_centers = np.repeat(x_center,ybins)
    bin_y_centers = np.tile(y_center,xbins)

    window_size = int(np.floor(Input_Variable_Cell.shape[0]/10.))
    I_start = np.arange(0,Input_Variable_Cell.shape[0]+1,window_size)

    Trials_testing_set =  np.arange(I_start[fold],I_start[fold+1]).astype(int)
    Trials_training_set = np.setdiff1d(range(Input_Variable_Cell.shape[0]),Trials_testing_set)

    X_train = Input_Variable_Cell[Trials_training_set,:].copy()
    y_train = Target_Variable[Trials_training_set].copy()

    X_test = Input_Variable_Cell[Trials_testing_set,:].copy()
    y_test = Target_Variable[Trials_testing_set].copy()

    
    priors_in = np.ones(np.unique(y_train).shape[0])/np.unique(y_train).shape[0]

    gnb = GaussianNB(priors = priors_in)
    gnb.fit(X_train, y_train)

    accuracy_original = gnb.score(X_test, y_test)


    y_pred = gnb.predict(X_test)
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)


    diffx = (bin_x_centers[y_pred]-x_coordinates[Trials_testing_set])**2
    diffy = (bin_y_centers[y_pred]-y_coordinates[Trials_testing_set])**2

    concatenated_nearest_dist_to_predicted = np.nanmean(np.sqrt(diffx + diffy))
                
    pred_dist_grid_original = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
    count = 0
    for xx in range(xbins):
        for yy in range(ybins):

            I_test = y_test == count
            if np.any(I_test):
                I_pred = y_pred[I_test].astype(int)

                diffx = (bin_x_centers[I_pred]-x_coordinates[Trials_testing_set][I_test])**2
                diffy = (bin_y_centers[I_pred]-y_coordinates[Trials_testing_set][I_test])**2

                nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                pred_dist_grid_original[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

            count += 1 

            
            
    predict_proba = gnb.predict_proba(X_test)
    mean_error_distance = []
    for fr in range(predict_proba.shape[0]):
        predic_distr_singleTime = predict_proba[fr,:]

        prob_grid = np.zeros((x_grid.shape[0]-1)*(y_grid.shape[0]-1))*np.nan
        prob_grid[gnb.classes_.astype(int)] = predict_proba[fr,:]

        mass_x_coord = np.nansum(prob_grid*bin_x_centers)
        mass_y_coord = np.nansum(prob_grid*bin_y_centers)

        error_distance = np.sqrt((mass_x_coord - x_coordinates[Trials_testing_set][fr])**2 +  (mass_y_coord - y_coordinates[Trials_testing_set][fr])**2)
        mean_error_distance.append(error_distance)
    mean_error_distance = np.nanmean(mean_error_distance)            
             

    pred_dist_grid_surrogates = []
    accuracy_surrogates = []
    concatenated_nearest_dist_to_predicted_perm = []
    mean_error_distance_perm = []
    for _ in range(num_surrogates):
        
        Input_Variable_Cell_Shuffled = []

        I_break = np.random.choice(np.arange(int(Input_Variable_Cell.shape[0]*0.1),int(Input_Variable_Cell.shape[0]*0.9)),1)[0].astype(int)

        if np.mod(_,4) == 0:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
        elif np.mod(_,4) == 1:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[:I_break:-1], Input_Variable_Cell[0:I_break+1]])
        elif np.mod(_,4) == 2:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[I_break-1::-1]])
        else:   
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
            Input_Variable_Cell_Shuffled = Input_Variable_Cell_Shuffled[::-1]

    

        X_train = Input_Variable_Cell_Shuffled[Trials_training_set,:].copy()
        y_train = Target_Variable[Trials_training_set].copy()

        X_test = Input_Variable_Cell_Shuffled[Trials_testing_set,:].copy()
        y_test = Target_Variable[Trials_testing_set].copy()

    
        gnb = GaussianNB(priors = priors_in)
        gnb.fit(X_train, y_train)

        accuracy = gnb.score(X_test, y_test)

        y_pred = gnb.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)

        diffx = (bin_x_centers[y_pred]-x_coordinates[Trials_testing_set])**2
        diffy = (bin_y_centers[y_pred]-y_coordinates[Trials_testing_set])**2

        concatenated_nearest_dist_to_predicted_perm.append(np.nanmean(np.sqrt(diffx + diffy)))
    
        pred_dist_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):

                I_test = y_test == count
                if np.any(I_test):
                    I_pred = y_pred[I_test].astype(int)

                    diffx = (bin_x_centers[I_pred]-x_coordinates[Trials_testing_set][I_test])**2
                    diffy = (bin_y_centers[I_pred]-y_coordinates[Trials_testing_set][I_test])**2

                    nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                    pred_dist_grid[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

                count += 1

        accuracy_surrogates.append(accuracy)
        pred_dist_grid_surrogates.append(pred_dist_grid)
        
        predict_proba = gnb.predict_proba(X_test)
        mean_error_distance_surr = []
        for fr in range(predict_proba.shape[0]):
            predic_distr_singleTime = predict_proba[fr,:]

            prob_grid = np.zeros((x_grid.shape[0]-1)*(y_grid.shape[0]-1))*np.nan
            prob_grid[gnb.classes_.astype(int)] = predict_proba[fr,:]

            mass_x_coord = np.nansum(prob_grid*bin_x_centers)
            mass_y_coord = np.nansum(prob_grid*bin_y_centers)

            error_distance = np.sqrt((mass_x_coord - x_coordinates[Trials_testing_set][fr])**2 +  (mass_y_coord - y_coordinates[Trials_testing_set][fr])**2)
            mean_error_distance_surr.append(error_distance)
            
        mean_error_distance_surr = np.nanmean(mean_error_distance_surr)    
        mean_error_distance_perm.append(mean_error_distance_surr)
        
    return pred_dist_grid_original,accuracy_original,pred_dist_grid_surrogates,accuracy_surrogates,x_coordinates[Trials_testing_set],y_coordinates[Trials_testing_set],concatenated_nearest_dist_to_predicted,concatenated_nearest_dist_to_predicted_perm,mean_error_distance,mean_error_distance_perm


    
    
    
def processInput_NaiveRuns_old(Input_Variable_Cell,Target_Variable,bins,x_coordinate_smoothed,y_coordinate_smoothed,num_surrogated,prior_dist):

    eps = np.finfo(float).eps

    xbins = bins
    ybins = bins
    
    x_grid_window = 1/xbins
    x_grid = np.arange(0,1+x_grid_window,x_grid_window)

    y_grid_window = 1/ybins
    y_grid = np.arange(0,1+y_grid_window,y_grid_window)
    
    x_center = x_grid[0:-1] + np.diff(x_grid)/2
    y_center = y_grid[0:-1] + np.diff(y_grid)/2
        
    bin_x_centers = np.repeat(x_center,ybins)
    bin_y_centers = np.tile(y_center,xbins)
    
    
    I_rand = np.random.choice(range(int(Input_Variable_Cell.shape[0]*0.9)))
    offset = int(0.1*Input_Variable_Cell.shape[0])
    Trials_testing_set =  np.arange(I_rand,I_rand+offset).astype(int)
    Trials_training_set = np.delete(range(Input_Variable_Cell.shape[0]),Trials_testing_set)

    X_train = Input_Variable_Cell[Trials_training_set,:].copy()
    y_train = Target_Variable[Trials_training_set].copy()

    X_test = Input_Variable_Cell[Trials_testing_set,:].copy()
    y_test = Target_Variable[Trials_testing_set].copy()

    
    priors_in = np.ones(np.unique(y_train).shape[0])/np.unique(y_train).shape[0]

    gnb = GaussianNB(priors = priors_in)
    gnb.fit(X_train, y_train)

    accuracy_original = gnb.score(X_test, y_test)


    y_pred = gnb.predict(X_test)
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)


    eps = np.finfo(float).eps

    x_test_coordinate = x_coordinate_smoothed
    y_test_coordinate = y_coordinate_smoothed
    


    diffx = (bin_x_centers[y_pred]-x_test_coordinate[Trials_testing_set])**2
    diffy = (bin_y_centers[y_pred]-y_test_coordinate[Trials_testing_set])**2

    concatenated_nearest_dist_to_predicted = np.nanmean(np.sqrt(diffx + diffy))
                
    pred_dist_grid_original = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
    count = 0
    for xx in range(xbins):
        for yy in range(ybins):

            I_test = y_test == count
            if np.any(I_test):
                I_pred = y_pred[I_test].astype(int)

                diffx = (bin_x_centers[I_pred]-x_test_coordinate[Trials_testing_set][I_test])**2
                diffy = (bin_y_centers[I_pred]-y_test_coordinate[Trials_testing_set][I_test])**2

                nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                pred_dist_grid_original[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

            count += 1 

    
    predict_proba = gnb.predict_proba(X_test)
    mean_error_distance = []
    for fr in range(0,predict_proba.shape[0]):
        predic_distr_singleTime = predict_proba[fr,:]

        prob_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):
                I_class= np.where(gnb.classes_ == count)[0]
                if I_class.size>0:
                    prob_grid[yy,xx] = predic_distr_singleTime[I_class]
                count += 1 

        x_sum = 0
        for yy in range(prob_grid.shape[0]):
            x_sum = x_sum + np.nansum(prob_grid[yy,:]*x_center)
        mass_x_coord = x_sum/np.nansum(prob_grid)

        y_sum = 0
        for xx in range(prob_grid.shape[1]):
            y_sum = y_sum + np.nansum(prob_grid[:,xx]*y_center)
        mass_y_coord =  y_sum/np.nansum(prob_grid)


        error_distance = np.sqrt((mass_x_coord - x_test_coordinate[Trials_testing_set][fr])**2 +  (mass_y_coord - y_test_coordinate[Trials_testing_set][fr])**2)
        mean_error_distance.append(error_distance)
    mean_error_distance = np.nanmean(mean_error_distance)            
                
    
    pred_dist_grid_surrogates = []
    accuracy_surrogates = []
    concatenated_nearest_dist_to_predicted_perm = []
    mean_error_distance_perm = []
    for _ in range(num_surrogated):
        
        Input_Variable_Cell_Shuffled = []

        I_break = np.random.choice(np.arange(int(Input_Variable_Cell.shape[0]*0.1),int(Input_Variable_Cell.shape[0]*0.9)),1)[0].astype(int)

        if np.mod(_,4) == 0:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
        elif np.mod(_,4) == 1:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[:I_break:-1], Input_Variable_Cell[0:I_break+1]])
        elif np.mod(_,4) == 2:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[I_break-1::-1]])
        else:   
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
            Input_Variable_Cell_Shuffled = Input_Variable_Cell_Shuffled[::-1]

    

        X_train = Input_Variable_Cell_Shuffled[Trials_training_set,:].copy()
        y_train = Target_Variable[Trials_training_set].copy()

        X_test = Input_Variable_Cell_Shuffled[Trials_testing_set,:].copy()
        y_test = Target_Variable[Trials_testing_set].copy()

    
        gnb = GaussianNB(priors = priors_in)
        gnb.fit(X_train, y_train)

        accuracy = gnb.score(X_test, y_test)

        y_pred = gnb.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)


        diffx = (bin_x_centers[y_pred]-x_test_coordinate[Trials_testing_set])**2
        diffy = (bin_y_centers[y_pred]-y_test_coordinate[Trials_testing_set])**2

        concatenated_nearest_dist_to_predicted_perm.append(np.nanmean(np.sqrt(diffx + diffy)))
    
        pred_dist_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):

                I_test = y_test == count
                if np.any(I_test):
                    I_pred = y_pred[I_test].astype(int)

                    diffx = (bin_x_centers[I_pred]-x_test_coordinate[Trials_testing_set][I_test])**2
                    diffy = (bin_y_centers[I_pred]-y_test_coordinate[Trials_testing_set][I_test])**2

                    nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                    pred_dist_grid[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

                count += 1

        accuracy_surrogates.append(accuracy)
        pred_dist_grid_surrogates.append(pred_dist_grid)        
        
        predict_proba = gnb.predict_proba(X_test)
        mean_error_distance_surr = []
        for fr in range(0,predict_proba.shape[0]):
            predic_distr_singleTime = predict_proba[fr,:]

            prob_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
            count = 0
            for xx in range(xbins):
                for yy in range(ybins):
                    I_class= np.where(gnb.classes_ == count)[0]
                    if I_class.size>0:
                        prob_grid[yy,xx] = predic_distr_singleTime[I_class]
                    count += 1 

            x_sum = 0
            for yy in range(prob_grid.shape[0]):
                x_sum = x_sum + np.nansum(prob_grid[yy,:]*x_center)
            mass_x_coord = x_sum/np.nansum(prob_grid)

            y_sum = 0
            for xx in range(prob_grid.shape[1]):
                y_sum = y_sum + np.nansum(prob_grid[:,xx]*y_center)
            mass_y_coord =  y_sum/np.nansum(prob_grid)


            error_distance = np.sqrt((mass_x_coord - x_test_coordinate[Trials_testing_set][fr])**2 +  (mass_y_coord - y_test_coordinate[Trials_testing_set][fr])**2)
            mean_error_distance_surr.append(error_distance)
        mean_error_distance_surr = np.nanmean(mean_error_distance_surr)    
        mean_error_distance_perm.append(mean_error_distance_surr)
        
    return pred_dist_grid_original,accuracy_original,pred_dist_grid_surrogates,accuracy_surrogates,x_test_coordinate[Trials_testing_set],y_test_coordinate[Trials_testing_set],concatenated_nearest_dist_to_predicted,concatenated_nearest_dist_to_predicted_perm,mean_error_distance,mean_error_distance_perm




def processInput_NaiveRuns_old(Input_Variable_Cell,Target_Variable,bins,x_coordinate_smoothed,y_coordinate_smoothed,num_surrogated,prior_dist):

    xbins = bins
    ybins = bins
    
    x_grid_window = 1/xbins
    x_grid = np.arange(0,1+x_grid_window,x_grid_window)

    y_grid_window = 1/ybins
    y_grid = np.arange(0,1+y_grid_window,y_grid_window)
    
    x_center = x_grid[0:-1] + np.diff(x_grid)/2
    y_center = y_grid[0:-1] + np.diff(y_grid)/2
        
    bin_x_centers = np.repeat(x_center,ybins)
    bin_y_centers = np.tile(y_center,xbins)
    
    
    I_rand = np.random.choice(range(int(Input_Variable_Cell.shape[0]*0.9)))
    offset = int(0.1*Input_Variable_Cell.shape[0])
    Trials_testing_set =  np.arange(I_rand,I_rand+offset).astype(int)
    Trials_training_set = np.delete(range(Input_Variable_Cell.shape[0]),Trials_testing_set)

    X_train = Input_Variable_Cell[Trials_training_set,:].copy()
    y_train = Target_Variable[Trials_training_set].copy()

    X_test = Input_Variable_Cell[Trials_testing_set,:].copy()
    y_test = Target_Variable[Trials_testing_set].copy()

    
    priors_in = np.ones(np.unique(y_train).shape[0])/np.unique(y_train).shape[0]

    gnb = GaussianNB(priors = priors_in)
    gnb.fit(X_train, y_train)

    accuracy_original = gnb.score(X_test, y_test)


    y_pred = gnb.predict(X_test)
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)


    eps = np.finfo(float).eps

    x_test_coordinate = x_coordinate_smoothed.copy()
    y_test_coordinate = y_coordinate_smoothed.copy()
    


    diffx = (bin_x_centers[y_pred]-x_test_coordinate[Trials_testing_set])**2
    diffy = (bin_y_centers[y_pred]-y_test_coordinate[Trials_testing_set])**2

    concatenated_nearest_dist_to_predicted = np.nanmean(np.sqrt(diffx + diffy))
                
    pred_dist_grid_original = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
    count = 0
    for xx in range(xbins):
        for yy in range(ybins):

            I_test = y_test == count
            if np.any(I_test):
                I_pred = y_pred[I_test].astype(int)

                diffx = (bin_x_centers[I_pred]-x_test_coordinate[Trials_testing_set][I_test])**2
                diffy = (bin_y_centers[I_pred]-y_test_coordinate[Trials_testing_set][I_test])**2

                nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                pred_dist_grid_original[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

            count += 1 

            
        
    pred_dist_grid_original2 = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
    count = 0
    for xx in range(xbins):
        for yy in range(ybins):

            I_test = y_pred == count
            if np.any(I_test):

                diffx = (bin_x_centers[count]-x_test_coordinate[Trials_testing_set][I_test])**2
                diffy = (bin_y_centers[count]-y_test_coordinate[Trials_testing_set][I_test])**2

                nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                pred_dist_grid_original2[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

            count += 1 
         
                
                
    pred_dist_grid_surrogates = []
    pred_dist_grid_surrogates2 = []
    accuracy_surrogates = []
    concatenated_nearest_dist_to_predicted_perm = []
    for _ in range(num_surrogated):
        
        Input_Variable_Cell_Shuffled = []

    #     old
    #     I_break = np.random.choice(np.arange(0,int(Input_Variable_Cell.shape[0])),1)[0].astype(int)
    #     Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])

        I_break = np.random.choice(np.arange(int(Input_Variable_Cell.shape[0]*0.1),int(Input_Variable_Cell.shape[0]*0.9)),1)[0].astype(int)

        if np.mod(_,4) == 0:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
        elif np.mod(_,4) == 1:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[:I_break:-1], Input_Variable_Cell[0:I_break+1]])
        elif np.mod(_,4) == 2:
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[I_break-1::-1]])
        else:   
            Input_Variable_Cell_Shuffled = np.concatenate([Input_Variable_Cell[I_break:], Input_Variable_Cell[0:I_break]])
            Input_Variable_Cell_Shuffled = Input_Variable_Cell_Shuffled[::-1]

    

        X_train = Input_Variable_Cell_Shuffled[Trials_training_set,:].copy()
        y_train = Target_Variable[Trials_training_set].copy()

        X_test = Input_Variable_Cell_Shuffled[Trials_testing_set,:].copy()
        y_test = Target_Variable[Trials_testing_set].copy()

    
        gnb = GaussianNB(priors = priors_in)
        gnb.fit(X_train, y_train)

        accuracy = gnb.score(X_test, y_test)

        y_pred = gnb.predict(X_test)
        y_test = y_test.astype(int)
        y_pred = y_pred.astype(int)


        eps = np.finfo(float).eps

        x_test_coordinate = x_coordinate_smoothed.copy()
        y_test_coordinate = y_coordinate_smoothed.copy()


        diffx = (bin_x_centers[y_pred]-x_test_coordinate[Trials_testing_set])**2
        diffy = (bin_y_centers[y_pred]-y_test_coordinate[Trials_testing_set])**2

        concatenated_nearest_dist_to_predicted_perm.append(np.nanmean(np.sqrt(diffx + diffy)))
    
        pred_dist_grid = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):

                I_test = y_test == count
                if np.any(I_test):
                    I_pred = y_pred[I_test].astype(int)

                    diffx = (bin_x_centers[I_pred]-x_test_coordinate[Trials_testing_set][I_test])**2
                    diffy = (bin_y_centers[I_pred]-y_test_coordinate[Trials_testing_set][I_test])**2

                    nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                    pred_dist_grid[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

                count += 1
                
                
        pred_dist_grid2 = np.zeros((x_grid.shape[0]-1,y_grid.shape[0]-1))*np.nan
        count = 0
        for xx in range(xbins):
            for yy in range(ybins):

                I_test = y_pred == count
                if np.any(I_test):

                    diffx = (bin_x_centers[count]-x_test_coordinate[Trials_testing_set][I_test])**2
                    diffy = (bin_y_centers[count]-y_test_coordinate[Trials_testing_set][I_test])**2

                    nearest_dist_to_predicted_bin = np.sqrt(diffx + diffy)

                    pred_dist_grid2[yy,xx] = np.nanmean(nearest_dist_to_predicted_bin)

                count += 1 
                
        accuracy_surrogates.append(accuracy)
        pred_dist_grid_surrogates.append(pred_dist_grid)
        pred_dist_grid_surrogates2.append(pred_dist_grid2)
        
        
    
        
    return pred_dist_grid_original,pred_dist_grid_original2,accuracy_original,pred_dist_grid_surrogates,pred_dist_grid_surrogates2,accuracy_surrogates,x_test_coordinate[Trials_testing_set],y_test_coordinate[Trials_testing_set],concatenated_nearest_dist_to_predicted,concatenated_nearest_dist_to_predicted_perm


