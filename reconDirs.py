#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to reconstruct tracking projections from subdirectories and create movies
Requires: readRthRaw.py, snrCalc, projPlot & theora_png2theora

For each subdir in current directory:
 - for each file matching the given pattern, create a png directory
  - within this directory, reconstruct the projections
  - then create a video of the projections
  - output basic stats from the coordinates & snrs to files

Each one of the outputs above is optional.

Note there may be multiple files in the subdirs that match the given pattern.

Run it like this depending on the desired outputs - the command below will 
generate all three outputs: plots, movies, and stats files:
 reconDirs.py -p -m -f 
"""
from __future__ import print_function
import os
import glob
import argparse
import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 6:
  raise("Python 2.6+ required...")

BASE_DIR=os.path.dirname(__file__)
READRTHRAW_PATH = BASE_DIR + "/readRthRaw.py"
THEORA_EXEC = "theora_png2theora"

def makeMovie(dirname,movie_name):
    """
    Use theora_png2theora to create a movie from png files
    - dirname: directory containing png files of the form "proj0000.png"
    - movie_name: file base name of theora-encoded movie to be created
    """
    dirname = os.path.abspath(dirname)
    callString = THEORA_EXEC + " " + dirname + "/proj%04d.png -f 20 -F 1 -v 10 -o " + movie_name + ".ogv"
    print(callString)
    os.system(callString)

def recon(fname,dirname,ylim,savePlots=True,saveStats=True,statsPrefix=""):
    """ Reconstruct projections from file "fname"
     - projection plots will be saved to "dirname" if savePlots is set
     - basic stats will be saved to a file if saveStats is true
     - statsPrefix: file prefix (can indicate file path) for stats file
    """
    startDir = os.getcwd()
    os.chdir(dirname)
    # Run the recon script
    optStr = " "
    if savePlots:
        optStr += "-p "
    if saveStats:
        optStr += "-f "
    if statsPrefix:
        optStr += "-x " + statsPrefix + " "
    callString = "python " + READRTHRAW_PATH + " " + fname + optStr + "-y " + str(ylim)
    print(callString)
    os.system(callString)
    os.chdir(startDir)

def descend(basedir,filepatt,ylim,movie_prefix,savePlots=True,saveMovies=True,saveStats=True):
    """
    Will search subdirectories (1 level only) of the basedir for filepatt files
    containing RTHawk projection recordings
     - ylim: the Y-axis maximum (so that everything is scaled the same way)
     - savePlots: reconstruct these into projection plots
     - saveMovies: make movies from the plots
     - saveStats: save basic stats from each recording into files
    """
    if movie_prefix == "":
        movie_prefix = os.path.basename( os.path.abspath(basedir) )
    for lname in os.listdir(basedir):
        subd = os.path.join(basedir, lname)        
        if os.path.isdir(subd):
            print("Subdir: " + subd)
            flist = glob.glob(subd + "/" + filepatt)
            for fname in flist:
                filePrefix = os.path.splitext(fname)[0]
                fileBase = os.path.basename(filePrefix)
                if filePrefix.find("-") != -1:
                    filePrefix = filePrefix[:filePrefix.find("-")]
                    fileBase = fileBase[:fileBase.find("-")]
                pngDir = filePrefix + "-pngDir"
                if savePlots and not os.path.isdir(pngDir):
                    os.mkdir(pngDir)
                elif savePlots:
                    print("Warning: will overwrite files in " + pngDir)
                absFname = os.path.abspath(fname)
                subdBase = os.path.basename(subd)
                namePrefix = movie_prefix+ "-" + subdBase + "-" + fileBase
                recon(absFname,pngDir,ylim,savePlots=savePlots,saveStats=saveStats,
                    statsPrefix=namePrefix)
                if saveMovies:
                    makeMovie(pngDir,namePrefix)

def main():
    parser = argparse.ArgumentParser(description="Reconstruct tracking projections")
    parser.add_argument("--pattern",dest="filepattern",default="cathcoil[4-5]-*.projections",help="file pattern")
    parser.add_argument("--ylim",dest="ylim",type=int,default=800,help="y-axis max")
    parser.add_argument("--prefix",dest="prefix",default="",help="prefix for movie files")
    parser.add_argument("-p","--save-plots",action="store_true",dest="saveplots",
        help="save plot files")
    parser.add_argument("-m","--save-movies",action="store_true",dest="savemovies",
        help="make movie files - plot files must also be saved for this to work")
    parser.add_argument("-f","--save-stats",action="store_true",dest="savestats",
        help="save stats files")

    args = parser.parse_args()
    print("Filepattern: " + args.filepattern)
    descend(".",args.filepattern,args.ylim,args.prefix,savePlots=args.saveplots,
        saveMovies=args.savemovies,saveStats=args.savestats)

if __name__ == "__main__":
    main()
