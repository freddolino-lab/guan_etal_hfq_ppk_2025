# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:52:48 2024

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob
from lmfit import Model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Define data name or label
name = 'Hfq_dppk_pre'
# set
set_label = '03'
# Define the integration time for imaging (in seconds)
t_int = .02
# Define the time delay between frames in seconds
t_delay = 0
# Define minimum track length (in frames)
min_frames = 4
# maximum gap allowed
max_gap = 10
# Define the pixel size in microns
pixel_size = 0.049
# Define the localization precision
sigma = 0
# Define the time lag to calculate displacements for in frames
lag = 1
# time lag in seconds
tau = (t_int + t_delay) * lag
# state model to fit to
states = 'es3'
# number of bootstraps
bootstraps = 5


# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def calculate_r2_displacement(trajectory, lag_time=lag):
    
 
    r2_displacements = []
    displacements = []
    
    for ii in range(len(trajectory[1]) - lag_time):
        
        r = ((trajectory[0][ii + lag_time] - trajectory[0][ii])**2 +
            (trajectory[1][ii + lag_time] - trajectory[1][ii])**2)**(0.5)
        
        
        r2_displacements.append((r * pixel_size)**2)
        
        displacements.append(r*pixel_size)
        
        
        
    return r2_displacements, displacements
    
    
    
def calculate_cumulative_probability(r2_displacements):
    '''
    sorted_data = np.sort(r2_displacements)
    cumulative_counts = np.arange(1, len(sorted_data) + 1) 
    cpd = cumulative_counts / len(sorted_data)
    '''
    
    x, counts = np.unique(r2_displacements, return_counts=True)
    cusum = np.cumsum(counts)
    
    return x, cusum / cusum[-1]
    #return sorted_data, cpd
    
    

def es1_model(r_2, D):
    
    return (1 - np.exp(-r_2 / ((8/3)*D*tau + 4*sigma**2)))


def es2_model(r_2, a, D1, D2):
   
    return (1 - (a*np.exp(-r_2 / ((8/3)*D1*tau + 4*sigma**2)) 
                + (1-a) * np.exp(-r_2 / ((8/3)*D2*tau + 4*sigma**2))))

def es3_model(r_2, a1, a2, D1, D2, D3):
   
    return (1 - (a1 * np.exp(-r_2 / ((8/3)*D1*tau + 4*sigma**2)) 
                + a2 * np.exp(-r_2 / ((8/3)*D2*tau + 4*sigma**2)) +
                (1-a1-a2) * np.exp(-r_2 / ((8/3)*D3*tau + 4*sigma**2))))

'''
def es1_model(r_2, D):
    
    return (1 - np.exp(-r_2 / ((8/3)*D*tau)))

def es2_model(r_2, a, D1, D2):
   
    return (1 - (a*np.exp(-r_2 / ((8/3)*D1*tau)) 
                + (1-a) * np.exp(-r_2 / ((8/3)*D2*tau))))


def es3_model(r_2, a1, a2, D1, D2, D3):
   
    return (1 - (a1 * np.exp(-r_2 / ((8/3)*D1*tau )) 
                + a2 * np.exp(-r_2 / ((8/3)*D2*tau )) +
                (1-a1-a2) * np.exp(-r_2 / ((8/3)*D3*tau ))))
'''

def fit_diffusion_model(x_data, y_data, fit_model = 'es2'):

    # Create a model for the single exponential
    if fit_model == 'es1':
        model = Model(es1_model)
        # Set up initial parameter values
        params = model.make_params(D = 0.01)
        params['D'].min = 0.0000001  # Set minimum limit for D
        params['D'].max = 10 # Set maximum limit for D
        #params['sigma'].min = 0.001  # Set minimum limit for D
        #params['sigma'].max = 0.1 # Set maximum limit for D
        
        # Perform the fit
        result = model.fit(y_data, params, r_2=x_data, max_nfev=None)
       
        print("D: ", result.params['D'].value, result.params['D'].stderr)
        #print("sigma: ", result.params['sigma'].value)
        
        
        
    elif fit_model == 'es2':
        model = Model(es2_model)
        # Set up initial parameter values
        params = model.make_params(a = 0.5, D1 = 0.001, D2 = 1)
        params['a'].min = 0.  # Set minimum limit for D
        params['a'].max = 1 # Set maximum limit for D
        params['D1'].min = 0.00001 # Set minimum limit for D
        params['D1'].max = .5 # Set maximum limit for D
        params['D2'].min = 0.01 # Set minimum limit for D
        params['D2'].max = 10 # Set maximum limit for D
        
        # Perform the fit
        result = model.fit(y_data, params, r_2=x_data, max_nfev=None)
       
        print("a1, D1, a2, D2:", (result.params['a'].value, result.params['D1'].value, 
                                 1 - result.params['a'].value, result.params['D2'].value))
        
        
    elif fit_model == 'es3':
        model = Model(es3_model)
        # Set up initial parameter values
        params = model.make_params(a1 = 0.5, a2 = 0.5, D1 = 0.01, D2 = .1, D3 = 3)
        params['a1'].min = 0  # Set minimum limit for D
        params['a1'].max = 1 # Set maximum limit for D
        params['a2'].min = 0  # Set minimum limit for D
        params['a2'].max = 1 # Set maximum limit for D
        params['D1'].min = 0.00001  # Set minimum limit for D
        params['D1'].max = 0.1 # Set maximum limit for D
        params['D2'].min = 0.04 # Set minimum limit for D
        params['D2'].max = 1 # Set maximum limit for D
        params['D3'].min = .1# Set minimum limit for D
        params['D3'].max = 10 # Set maximum limit for D
        
        # Perform the fit
        result = model.fit(y_data, params, r_2=x_data, max_nfev=None)
       
        print("a1, D1, a2, D2, a3, D3:", (result.params['a1'].value, result.params['D1'].value, 
                                 result.params['a2'].value, result.params['D2'].value,
                                 1 - result.params['a1'].value - result.params['a2'].value, 
                                result.params['D3'].value))    
        
        



    return result

def has_gap(values):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > (max_gap+1):
            return True
    return False

directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'tracks' in file]

all_r2_displacements = []
all_displacements = []
track_count = 0
for file in track_files[22:]:
    
    # read csv as dataframe
    df_ = pd.read_csv(file, header=0)
    df = df_[df_['blob_track_class'] != 'out']
   
    # loop through each track
    for track_id in df["TRACK_N"].unique()[:]:
        # load track data
        track_data = df[df["TRACK_N"] == track_id]
        n = len(track_data)
        gap = gap = has_gap(track_data["FRAME_N"].tolist())
        # calculate radius of confinement if track is min_frames long
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames and gap == False:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['LOC_C'].to_numpy()
            track_y_coords = track_data['LOC_R'].to_numpy()
            
            r2_displacements, displacements = calculate_r2_displacement(np.asarray([track_x_coords, track_y_coords]), 
                                                          lag_time = lag)
            
            all_r2_displacements.extend(r2_displacements)
            all_displacements.extend(displacements)
            track_count += 1
            
print("tracks used ", track_count)
print("jumps used ", len(all_r2_displacements))

# perform bootstrap analysis
results = []
cpds = []
for m in range(bootstraps):
    
    # sample dataset
    boostrapped_data =  np.random.choice(all_r2_displacements, replace=True, size=int(len(all_r2_displacements)))
    # calculate CPD of displacements      
    r_values, cpd = calculate_cumulative_probability(boostrapped_data)
    

    cpd_df = pd.DataFrame({'Displacement': r_values, 'Cumulative Probability': cpd})
    #cpd_df.to_csv(directory[:-5] + name + '_cumulativeprob.csv',index = False)


    # fit CPD to diffusion model
    fit_result = fit_diffusion_model(r_values, cpd, fit_model = states)
    if states == 'es1':
        fit_model = es1_model(r_values, fit_result.params['D'].value 
                     )
        
        result_params = [fit_result.params['D'].value]

    if states == 'es2':
        fit_model = es2_model(r_values, fit_result.params['a'].value, 
                      fit_result.params['D1'].value, 
                      fit_result.params['D2'].value)
        
        result_params = [fit_result.params['a'].value, fit_result.params['D1'].value, 
                                1- fit_result.params['a'].value, fit_result.params['D2'].value
                                ]
    if states == 'es3':
            fit_model = es3_model(r_values, fit_result.params['a1'].value, 
                          fit_result.params['a2'].value,
                      fit_result.params['D1'].value, 
                      fit_result.params['D2'].value,
                      fit_result.params['D3'].value)   
            result_params = [fit_result.params['a1'].value, fit_result.params['D1'].value, 
                                     fit_result.params['a2'].value, fit_result.params['D2'].value,
                                     1 - fit_result.params['a1'].value - fit_result.params['a2'].value, 
                                    fit_result.params['D3'].value]
    cpds.append(cpd)
    
    
    results.append(np.asarray(result_params))
    
print("bootstrapped means ", np.mean(np.asarray(results), axis=0))
print("bootstrapped stdevs ", np.std(np.asarray(results), axis=0)  )     
'''
cpd_avg = np.mean(np.asarray(cpds), axis=0)
cpd_CIlow = np.percentile(np.asarray(cpds), 5, axis=0)
cpd_CIhigh = np.percentile(np.asarray(cpds), 95, axis=0)
'''
print(fit_result.fit_report())
 
# Plot the results
fig, axes = plt.subplots(ncols=1,nrows=2, figsize= (3,4), 
                                 sharey=False, sharex=True, dpi=200)
ax = axes.ravel()
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12

#ax[0].fill_between(r_values, cpd_CIlow, cpd_CIhigh, 
#                   alpha=0.5, zorder=-1)
ax[0].plot(r_values, cpd, label="Empirical CDF", linestyle='-', linewidth=1)
ax[0].plot(r_values, fit_model, linewidth=0.5)
#ax[0].set_xlim(10**-5,0.25)
#ax[0].set_xticks([10**-5, 10**-4, 10**-3, 10**-2, 10**-1])
#fit_result.plot_fit()
ax[0].set_xlabel("Displacement")
ax[0].set_xscale('log')
ax[0].set_ylabel("Cumulative Probability")
ax[0].set_ylim(0, 1)

residuals = cpd - fit_model
ax[1].plot(r_values, residuals)
ax[1].set_xscale('log')
ax[1].set_xlabel("Displacement")
ax[1].set_ylabel("Residuals")
#ax[1].set_xlim(10**-5,0.04)
#ax[1].set_xticks([10**-5, 10**-4, 10**-3, 10**-2,])
ax[1].set_ylim(-0.1, 0.1)
#plt.legend()
fig.tight_layout()
plt.savefig(directory[:-5] + name + 'CPD_fit and residuals.svg', dpi=300) 
plt.show()
            
            
            