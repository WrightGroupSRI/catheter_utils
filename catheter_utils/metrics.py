"""This module contains functions related to evaluating/comparing results."""

import numpy as np
import os
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from textwrap import wrap
import glob
import catheter_utils.cathcoords
import catheter_utils.geometry
import catheter_utils.projections
from get_gt import __main__ as get_gt

def get_coords(data_file):
    '''takes a path to a txt file and extracts coordinate data.
	   
	   data_file is the path to a .txt file that contains the 
	   x,y,z coordinates 
    '''

    file = open(data_file, 'r')
    lines = file.readlines()
    coords = []
    
    for line in lines: 
        data = line.split(' ')
        x, y, z = float(data[0]), float(data[1]), float(data[2])
        coords.append([x, y, z])
    
    return np.array(coords) 


def Chebyshev(distal_file, proximal_file, gt_coords, Geometry):
    '''produces the 95% chebyshev radius.
	
	   distal_file: text file of distal coordinates 
	   proximal_file: text file of proximal coordinates
	   gt_coords: ground truth tip coordinates 
	   Geometry: the geometry of the coils 
	'''
    
    distal_coords = get_coords(distal_file)
    proximal_coords = get_coords(proximal_file)
    
    fit_results = Geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    tip_coords = (fit_results.tip).T

    Cov_mat = np.cov(tip_coords)
    Evals, Evects = np.linalg.eig(Cov_mat)
    
    sigma = max(Evals)
    return sigma * 2 * np.sqrt(5/3) 


def Bias(distal_file, proximal_file, gt_coords, Geometry): 
    '''Return the mean bias between the ground truth and measurements.
	
	   distal_file: text file of distal coordinates 
	   proximal_file: text file of proximal coordinates
	   gt_coords: ground truth tip coordinates 
	   Geometry: the geometry of the coils 
	'''
	
    distal_coords = get_coords(distal_file)
    proximal_coords = get_coords(proximal_file)
    
    fit_results = Geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    mean_tip = np.mean(fit_results.tip, axis=0)
    
    bias_vect = mean_tip - gt_coords
    bias = np.linalg.norm(bias_vect)
    
    return bias

def get_bias_array(distal_file, proximal_file, gt_coords, Geometry): 
    '''return the list of biases between the ground truth and measurements.
	
	   distal_file: text file of distal coordinates 
	   proximal_file: text file of proximal coordinates
	   gt_coords: ground truth tip coordinates 
	   Geometry: the geometry of the coils 
	'''
	
    distal_coords = get_coords(distal_file)
    proximal_coords = get_coords(proximal_file)
    
    fit_results = Geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    
    bias_vect = np.array(fit_results.tip) - np.array(gt_coords)
    bias_vect = np.linalg.norm(bias_vect,axis=1)
    
    return bias_vect

def trackerr(seq_path, src_path, groundtruth, algorithm, dest_path, distal_index, proximal_index, dither_index, expname):
    '''
    retrieve .txt files from localization algorithm folders, FOV from projection files, and groundtruth coordinates
    then calculate Bias and Chebyshev tracking error. Data exported as csv with expname (default is experiment name in GroundTruthCoords.csv)
    '''

    cathcoord_files = []  # store .txt localization algorithm files
    fov_list = []  # store field of view from projection files
    data = []
    # sort algorithms and remove whitespace characters
    algorithm = list(map(str.strip, algorithm.rsplit(',')))
    algorithm = list(filter(None, algorithm))

    # get .txt files from localization algorithm folders
    if src_path is not None:
        if not Path(src_path).is_dir():
            raise Exception("localization_folder argument must be a path and not a file")
        try:
            for folder in algorithm:
                path = os.path.join(src_path, folder)
                if os.path.exists(path) == True:
                    cathcoord_files.append(catheter_utils.cathcoords.discover_files(path))
                else:
                    raise Exception("algorithm folder '{}' does not exist in {}".format(folder, src_path))
        except KeyError:
            raise Exception("Error reading files from folders '{}' in path {}".format(algorithm, src_path))

    # get FOV from projections files in tracking sequence folder
    if seq_path is not None:
        try:
            discoveries, unknown = catheter_utils.projections.discover_raw(seq_path)
            projection_files = discoveries[(discoveries["coil"] == proximal_index) & (discoveries["dither"] == dither_index) & (discoveries["axis"] == 0)]["filename"]
            for file in projection_files.tolist():
                meta, raw, _, _ = catheter_utils.projections.read_raw(file, allow_corrupt=False)
                fov_list.append(catheter_utils.projections.fov_info(meta, raw))
        except KeyError:
            raise Exception("unrecognized projections. Check validity of projection files")
    fov = [(fov.split(' (resolution', 1)[0]) for fov in fov_list]

    # get groundtruth coordinates for distal and proximal coil
    distal_gtcoord = get_gt.read_results(groundtruth, Exp_name=expname, Coil_index=distal_index)
    proximal_gtcoord = get_gt.read_results(groundtruth, Exp_name=expname, Coil_index=proximal_index)
    if isinstance(distal_gtcoord, type(None)) or isinstance(proximal_gtcoord, type(None)):
        raise Exception("coil index {} or experiment name couldn't be found in GroundTruthCoords.csv".format([proximal_index,distal_index]))

    #estimate catheter tip location
    geo = catheter_utils.geometry.estimate_geometry(np.array([distal_gtcoord]), np.array([proximal_gtcoord]))
    fit_gt = geo.fit_from_coils_mse(np.array([distal_gtcoord]), np.array([proximal_gtcoord]))
    tip_gt = fit_gt.tip

    # default output directory is localization src folder
    if isinstance(dest_path, type(None)) or not os.path.exists(dest_path):
        dest_path = src_path
    trackseq = os.path.split(seq_path)[-1]

    # Calculate Bias and Chebyshev for each localization algorithm and recording, and store all collected info into dataframe
    for alg_index, file in enumerate(cathcoord_files):
        for record in range(len(file)):
            data.append([trackseq, algorithm[alg_index], fov[record], dither_index,
                         catheter_utils.metrics.Bias(file[record][distal_index], file[record][proximal_index], tip_gt,geo),
                         catheter_utils.metrics.Chebyshev(file[record][distal_index], file[record][proximal_index],tip_gt, geo)])
    df = pd.DataFrame(data, columns=['trackseq', 'algorithm', 'FOV', 'dither', 'bias', 'chebyshev'])

    # default experiment name is name found in GroundTruthCoords.csv. Note: experiment name used for csv filename
    if isinstance(expname, type(None)):
        expname = pd.read_csv(os.path.join(groundtruth, 'GroundTruthCoords.csv'))
        expname = expname['Experiment Name'].values[0]
    outdir = os.path.join(dest_path, expname + "_trackerr.csv")

    # Export data to csv
    if os.path.exists(outdir):
        # remove duplicates from csv file and format data
        cur_df = pd.read_csv(outdir, na_values="NaN")
        df = pd.concat([cur_df.round(14).astype(str), df.round(14).astype(str)], ignore_index=True)
    df = df.loc[~df.duplicated(keep='first')]
    df.to_csv(outdir, index=False)
    print("{} was saved".format(outdir))

def barplot(dest, expname, x_axis):
    '''
    plot bar graph of Bias and Chebyshev tracking error. x-axis: algorithm (Default), y-axis: Bias + Chebyshev. Rest of data columns are filters
    '''
    # filter options
    filter_by = ['trackseq', 'dither', 'FOV', 'algorithm']

    # get trackingerr csv file. if expname not specified search for first csv file in dest path
    try:
        if isinstance(expname, type(None)):
            df = pd.read_csv(next(glob.iglob("/*".join((dest, '_trackerr.csv')))), usecols=filter_by + ['bias', 'chebyshev'], na_values="NaN")
        else:
            df = pd.read_csv(os.path.join(dest, expname + '_trackerr.csv'), usecols=filter_by + ['bias', 'chebyshev'], na_values="NaN")
    except Exception as err:
        raise Exception(err, f"Error reading trackerr.csv file at {dest}")

    # remove selected x-axis from filter then get filter categories
    filter_by.remove(x_axis)
    filt1_list = df[filter_by[0]].unique()
    filt2_list = df[filter_by[1]].unique()
    filt3_list = df[filter_by[2]].unique()

    # plot per filter category
    for filt1 in filt1_list:
        for filt2 in filt2_list:
            for filt3 in filt3_list:
                dfilt = df[(df[filter_by[0]] == filt1) & (df[filter_by[1]] == filt2) & (df[filter_by[2]] == filt3)]
                if not dfilt.empty:
                    ax1 = dfilt.plot(kind='bar', x=x_axis, y=['bias', 'chebyshev'], stacked=True, rot=0)
                    # set labels, titles, and spacing
                    labels = ['\n'.join(wrap(str(lbl), 15)) for lbl in dfilt[x_axis].tolist()]
                    ax1.set_xticklabels(labels)
                    plt.xlabel(x_axis)
                    plt.ylabel('Distance (mm)')

                    # exclude dither from title if tracking sequence not dithered
                    filt_subtitle = np.array([filter_by[0], filt1, filter_by[1], filt2, filter_by[2], filt3]).astype(str)
                    if len(dfilt.index[dfilt['trackseq'].str.contains('(?i)dithered')]) == 0 and 'dither' in filter_by:
                        dither_loc = np.where(np.array(filt_subtitle) == 'dither')[0]
                        filt_subtitle = np.delete(filt_subtitle, [dither_loc[0], dither_loc[0] + 1])
                    title = 'Tracking Error of {} over varying {}'.format(" ".join(filt_subtitle), x_axis)
                    plt.title("\n".join((wrap(title, 40))))

                    # make legend draggable by mouse
                    leg = plt.legend()
                    leg.set_draggable(True)

                    plt.show()