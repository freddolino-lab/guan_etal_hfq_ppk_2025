# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:07:37 2024

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob
from lmfit import Model
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# Define data name or label
name = 'Hfq_wt_pre'
# Replicate information
set_label = 'all'
# Define the integration time for imaging (in seconds)
t_int = .02
# Define the time delay between frames in seconds
t_delay = 0
# Define minimum track length (in frames)
min_frames = 10
# Define maximum frame allowed within a track
max_gap = 4
# Define the pixel size in microns
pixel_size = 0.049
# define max time lag for MSD curve
max_tau = 4
max_tau_for_fit = 4#int(max_tau / 2)
# plot the single track MSD?
plot_single_track_msd = True




# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    filenames = np.sort(filenames)
    print(filenames)
    return filenames


def has_gap(values):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > (max_gap+1):
            return True
    return False



def calculate_msd(x, y, max_tau):
    # Initialize MSD array
    msd = []
    weights = []
    
    # Compute MSD
    for tau in range(1, max_tau+1):
        displacements = []
        for t in range((max_tau+1) - tau):
            dx = (x[t + tau] - x[t])*pixel_size
            dy = (y[t + tau] - y[t])*pixel_size
            displacements.append(dx**2 + dy**2)
        msd.append(np.mean(displacements))
        weights.append(len(displacements))
    
    return msd, np.asarray(weights)


def anomalous_diffusion(tau, D_alpha, alpha):
    return 4 * D_alpha * tau**alpha

def brownian_blur(tau, D, sig): # normal Brownian motion with blurring
    return (8/3)*D*(tau) + 4*(sig)**2



def calculate_confinement_radius(x_coords, y_coords):
    # Calculate centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    # Calculate distances to centroid
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
    
    # Calculate confinement radius
    confinement_radius = np.mean(distances)*pixel_size
    
    return confinement_radius


directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'tracks' in file]
# Initialize a list to store the track lengths from all files
all_msd_out = []
d_out = []
sigma_out = []
track_length_out = []
track_ids = []
set_ids = []


confinement_radius = []
# load through each file
for file in track_files[:]:
    
    # read csv as dataframe
    df = pd.read_csv(file, header=0)
    # loop through each track
    for track_id in df["TRACK_N"].unique():
        # load track data
        track_data = df[df["TRACK_N"] == track_id]
        n = len(track_data)
        gap = gap = has_gap(track_data["FRAME_N"].tolist())
        # calculate radius of confinement if track is min_frames long
        
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames and gap == False:
            
            max_tau = max_tau
            max_tau_for_fit = max_tau_for_fit
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['LOC_C'].to_numpy()[:]
            track_y_coords = track_data['LOC_R'].to_numpy()[:]
            time = track_data['FRAME_N'].to_numpy()[:]
            #print(len(track_x_coords), len(track_y_coords))
            
            # calculate MSD vs tau curve
            time_lags = np.arange(1,max_tau+1) * (t_int+t_delay)
           
            msd_curve, weights = calculate_msd(track_x_coords, track_y_coords, max_tau)
            weights[weights == 0] = 1  # Avoid division by zero
            sigma = np.sqrt(1 / weights)
            
            # fit to MSD function
            popt, pcov = curve_fit(brownian_blur, time_lags[:max_tau_for_fit], 
                                   msd_curve[:max_tau_for_fit], 
                                   maxfev = 10000, bounds=(0, [np.inf, 2]),
                                   sigma=sigma[:max_tau_for_fit], absolute_sigma=True)
           # popt, pcov = curve_fit(anomalous_diffusion, time_lags, msd, sigma=sigma, absolute_sigma=True)
            # Extract fit parameters
            D, sigma = popt
            # calculate fitted values
            msd_fitted = brownian_blur(time_lags, *popt)
            # Residuals
            residuals = msd_curve[:max_tau_for_fit] - msd_fitted[:max_tau_for_fit]
            # Calculate R^2 and RMSE
            r2 = r2_score(msd_curve[:max_tau_for_fit], msd_fitted[:max_tau_for_fit])
           
            if  r2 < 0 and D < 0.001:
                sigma_out.append(sigma)
                d_out.append(D)
                track_length_out.append(n)
                all_msd_out.append([time_lags, msd_curve])
                track_ids.append(track_id)
                set_ids.append(set_label)
                
                
                radius = calculate_confinement_radius(track_x_coords, track_y_coords)
                confinement_radius.append(radius)
                
                
                
                
                
                    
                    
                    
                    
                    
            #print(f"Fitted parameters: D_alpha = {D_alpha}, alpha = {alpha}")
            
            if plot_single_track_msd == True and n >= 4:
                # plot single track MSD curves (optional)
                fig, axes = plt.subplots(figsize=(6, 3), ncols=2, dpi=200) 
                ax = axes.ravel()
                ax[1].set_title(str(track_id) + ' ' + str(r2))
                ax[0].plot((track_x_coords-track_x_coords[0])*pixel_size,
                           (track_y_coords-track_y_coords[0])*pixel_size )
                ax[0].set_xlim(-.5,.5)
                ax[0].set_ylim(-.5,.5)
                ax[1].plot(time_lags,msd_curve, marker='o')
                ax[1].plot(time_lags, brownian_blur(time_lags, *popt), '-',)
                ax[1].set_xlabel('Time lag (tau)')
                ax[1].set_ylabel('Mean Squared Displacement')
                fig.tight_layout()
                plt.show()
            
            
            
    # Append the track lengths to the list
  
'''  
fig, axes = plt.subplots(figsize=(2.7, 2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
bs_id =  np.random.choice(range(len(all_msd_out)), replace=False, 
                         size=20)
for msd in bs_id:
    
    plt.plot(all_msd_out[msd][0],all_msd_out[msd][1],alpha=0.75, 
             linewidth=1)
    
#plt.plot(time_lags, avg_msd, marker='o', color='black',linestyle='None', markersize=2)
#plt.fill_between(time_lags, std_msd2, std_msd1, alpha=0.25, linewidth=0,color='gray')
plt.xlabel('Time Lag (s)')
plt.ylabel('MSD')
#plt.xscale('log')
plt.xlim(0.08, 3)
plt.xticks([0.5, 1, 1.5, 2, 2.5, 3])
#plt.ylim(10**-4, 10**1)
#plt.yscale('log')


fig.tight_layout()
plt.savefig(directory[:-5] + name + '_msd.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_msd.png', dpi=300) 
plt.show()

time_lags = np.arange(1,max_tau+1) * (t_int+t_delay)
# plot histogram
# Plot VACF
print(len(all_msd_out))

avg_msd = np.mean(all_msd_out, axis = 0)
popt, pcov = curve_fit(anomalous_diffusion, time_lags[:max_tau_for_fit], 
                       avg_msd[:max_tau_for_fit], bounds=(0, [np.inf, 2]))
# Extract fit parameters
print(popt)

std_msd1 = np.percentile(all_msd_out, 95, axis = 0)
std_msd2 = np.percentile(all_msd_out,5, axis = 0) #/ np.sqrt(len(all_msd_out))
#avg_msd = np.mean(all_msd_out, axis = 0)
#std_msd = np.percentile(all_msd_out,95, axis = 0) #/ np.sqrt(len(all_
print(avg_msd)
print(std_msd1)
print(std_msd2)
print("Diff coeff", np.mean(d_out), np.std(d_out)/np.sqrt(len(d_out)), np.percentile(d_out, 5),np.percentile(d_out, 95) )
#print("alpha", np.mean(alpha_out), np.std(alpha_out))

'''
    
labels = [name for i in range(len(track_ids))]
datatosave = [track_ids,d_out,sigma_out, labels, set_ids, track_length_out]
dataDF = pd.DataFrame(datatosave).transpose()
dataDF.columns = ['ID','D','sigma', 'Label', 'set', 'Track_len']
dataDF.to_csv(directory[:-5] + name + '_' + set_label + '_mintracklen-{}'.format(min_frames) + '_Dcoeff_SMT.csv', index = False)

#msd_curves_arr = np.asarray(all_msd_out)    
#msd_curves_df = pd.DataFrame(msd_curves_arr.T)
#msd_curves_df.insert(0,'Time', time_lags)
#msd_curves_df.to_csv(directory[:-5] + name +  '_mintracklen-{}'.format(min_frames) + '_MSD_curves.csv', index = False)
#print(msd_curves_df)  



# plot the data    
def plot_data(dcoeffs):
    
    # no. of bins for all datasets
    binwidth = 0.01
    binBoundaries = np.arange(min(dcoeffs),max(dcoeffs), binwidth)
    # initiate figure
    
    fig, axes = plt.subplots(figsize=(3, 3), dpi=300) 
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    fontsize = 9
    
    plt.title("App. Diff. Coef. (n={})".format(len(dcoeffs)), 
                    fontsize=fontsize, )
    plt.xlabel('D_app (um^2/s)', fontsize=fontsize, )
    plt.ylabel('counts', fontsize=fontsize, )
   
    data_hist, bins = np.histogram(dcoeffs, bins=binBoundaries) 
    print("N bins", len(bins))
    #logbins = np.logspace(np.log10(min(dcoeffs)), np.log10(max(dcoeffs)),  len(bins))
    x, bins, p = plt.hist(dcoeffs, bins=binBoundaries, color='forestgreen',
                            edgecolor='k', alpha=0.75)
    #plt.xscale('log')
    plt.yscale('linear')
    #for item in p:
    #    item.set_height(item.get_height()/sum(x))
    #plt.ylim(0,40)
    #plt.xlim(0, 60)
    fig.tight_layout()
    plt.show()
    
print("N tracks", len(d_out))    
plot_data(confinement_radius)
