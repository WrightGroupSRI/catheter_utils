#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Read and plot catheter raw data from RthReconImageExporter

    Reads a single raw data file exported from RTHawk. This is assumed to be a catheter
    projection file. The first projection is plotted and subsequent projections can
    be selected for plotting.

    Plots in png format and/or coordinates & physiology data in text format can be output instead using
    the "-p" and "-c" options. Basic stats on the coordinates and peaks can be output using the "-f"
    option.

    Note that respiratory data from RTHawk is scaled by 10^5 to give an integer (the respiratory information
    from RTHawk is a float between 0 and 1).

    Run readRthRaw without any arguments for details on each option.

    Usage examples:
        > ./readRthRaw.py data/file-0000.projections
        - This will display the projections in a window

        > ./readRthRaw.py data/file-0000.projections -p
        - This will save the projections to PNG files
        
        > ./readRthRaw.py data/file-0000.projections -c
        - This will save the peak coordinates to txt files

"""
from __future__ import print_function
import sys, struct
import math
import scipy
import scipy.fftpack
from optparse import OptionParser
import os
import snrCalc
import numpy as np
from projPlot import *

if sys.version_info[0] < 3 and sys.version_info[1] < 6:
  raise("Python 2.6+ required...")

float_bytes = 8 #These are being written on a 64-bit system

class RawReader:
    def __init__(self,fname="",float_bytes=8,legacy_version=0):
        self.rawFile = fname
        self.legacy = legacy_version
        self.float_bytes = float_bytes
        self.setup()

    def setup(self):
        self.projections = [] # array of tuples, where each tuple is a series of complex floats
        self.projComplex = []
        self.triggerTimes = [] #array of trigger times, one triggerTime per each triplet of projections
        self.respPhases = []
        self.timestamps = []
        self.pg = []
        self.ecg1 = []
        self.ecg2 = []
        self.projNum = 0
        self.projSize = 0
        self.fts = []
        self.projRaw = []     # This will be the same as projComplex but with the x, y and z projections each in their own row
        self.xsize = 0
        self.ysize = 0
        self.zsize = 0
        self.fieldOfView = 0
        self.version = 0

    def readProjectionSizes(self,fp):
      hdr = fp.read(12) # 3 32-bit ints
      if not hdr:
        print("reached EOF")
        return(0,0,0,False)
      else:
        xsize = struct.unpack('>i',hdr[0:4])[0]
        ysize = struct.unpack('>i',hdr[4:8])[0]
        zsize = struct.unpack('>i',hdr[8:12])[0]
        return (xsize,ysize,zsize,True)

    def readFloat64(self,fp):
      hdr = fp.read(8)
      if not hdr:
        print("reached EOF")
        return (-1,False)
      else:
        return (struct.unpack('>d',hdr[0:8])[0],True)

    def readFOV(self,fp):
      return self.readFloat64(fp)

    def readInt64(self,fp):
      hdr = fp.read(8)
      if not hdr:
          print("reached EOF")
          return (-1,False)
      else:
        return (struct.unpack('>q',hdr[0:8])[0],True)

    def readTimestamp(self,fp):
      return self.readInt64(fp)

    def readTrig(self,fp):
      return self.readFloat64(fp)
  
    def readResp(self,fp):
      return self.readFloat64(fp)

    def readTrace(self,fp):
      return self.readFloat64(fp)

    def readV1Header(self,fp):
      (xsize,ysize,zsize,ok) = self.readProjectionSizes(fp)
      NULL_TUPLE = (0,0,0,0,0,0,0,0,0,0)
      fov,ok = self.readFOV(fp)
      timestamp,ok = self.readTimestamp(fp)
      trig,ok = self.readTrig(fp)
      resp,ok = self.readResp(fp)
      pg,ok = self.readTrace(fp)
      ecg1,ok = self.readTrace(fp)
      ecg2,ok = self.readTrace(fp)
      if not ok:
        return NULL_TUPLE
      return (xsize,ysize,zsize,fov,trig,resp,timestamp,pg,ecg1,ecg2)

    def readLegacy3Header(self,fp):
      (xsize,ysize,zsize,ok) = self.readProjectionSizes(fp)
      NULL_TUPLE = (0,0,0,0,0,0,0)
      fov,ok = self.readFOV(fp)
      timestamp,ok = self.readTimestamp(fp)
      trig,ok = self.readTrig(fp)
      resp,ok = self.readResp(fp)
      if not ok:
        return NULL_TUPLE
      return (xsize,ysize,zsize,fov,trig,resp,timestamp)

    def readLegacy2Header(self,fp):
      (xsize,ysize,zsize,ok) = self.readProjectionSizes(fp)
      NULL_TUPLE = (0,0,0,0,0,0)
      fov,ok = self.readFOV(fp)
      trig,ok = self.readTrig(fp)
      resp,ok = self.readResp(fp)
      if not ok:
        return NULL_TUPLE
      else:
        return (xsize,ysize,zsize,fov,trig,resp)

    def readLegacyHeader(self,fp):
      (xsize,ysize,zsize,ok) = self.readProjectionSizes(fp)
      NULL_TUPLE = (0,0,0,0)
      fov,ok = self.readFOV(fp)
      if not ok:
        return NULL_TUPLE
      else:
        return (xsize,ysize,zsize,fov)
    
    def checkFileHeader(self,fp):
        hdrBytes = fp.peek(8)
        fmt = b''.join(struct.unpack('4c',hdrBytes[0:4]))
        fmt = fmt.decode('ascii')
        version = struct.unpack('>i',hdrBytes[4:8])[0]
        if (fmt != "CTHX" or version <= 0):
          return False # legacy format
        else:
          self.version = version
          self.legacy = 0
          fp.read(8) # Digest the header bytes
          return True

    def readFile(self,rawFile = ""):
        if not rawFile:
            rawFile = self.rawFile
        self.setup()
        fp = open(rawFile,"rb")
        if self.checkFileHeader(fp):
            print('Cath raw file version: ' + str(self.version))
        elif not self.legacy:
            self.legacy = 3 # Assume most recent legacy format
            print('Detected legacy format, assuming version ' + str(self.legacy))
        done = False
        first = True
        while not done:
          xs = ys = zs = fov = 0
          if self.legacy == 1:
            xs,ys,zs,fov=self.readLegacyHeader(fp)
          elif self.legacy == 2:
            xs,ys,zs,fov,trig,resp=self.readLegacy2Header(fp)
            self.triggerTimes.append(trig)
            self.respPhases.append(resp)
          elif self.legacy == 3:
            xs,ys,zs,fov,trig,resp,timestamp=self.readLegacy3Header(fp)
            self.triggerTimes.append(trig)
            self.respPhases.append(resp)
            self.timestamps.append(timestamp)
          elif self.legacy == 0 and self.version == 1:
            xs,ys,zs,fov,trig,resp,timestamp,pg,ecg1,ecg2=self.readV1Header(fp)
            self.triggerTimes.append(trig)
            self.respPhases.append(resp)
            self.timestamps.append(timestamp)
            self.pg.append(pg)
            self.ecg1.append(ecg1)
            self.ecg2.append(ecg2)
          else:
            print('Unknown format, version=' + str(self.version) + ', legacy=' + str(self.legacy))
            return
          if (xs == 0 or ys == 0 or zs == 0):
            done = True;
            break
          if first:
            self.xsize = xs
            self.ysize = ys
            self.zsize = zs
            self.fieldOfView = fov
          projSize = xs*ys*zs*2
          projByteSize = projSize*float_bytes
          proj = fp.read(projByteSize)
          if proj is None or len(proj) < projByteSize:
            print("Could not read projection " + str(self.projNum) + " stopping here.")
            break
          self.projections.append( struct.unpack('>'+str(projSize)+'d',proj[0:projByteSize]) )
          self.projNum+=1
        print("Read " + str(self.projNum) + " projections...",end='')
        print("x size = " + str(self.xsize) + ", ysize = " + str(self.ysize) 
            + " zsize = " + str(self.zsize) + " fov = " + str(self.fieldOfView))
        # NOTE: each projection in projComplex and projections contains the x, y and z projections
        for proj in range(0,self.projNum):
          self.projComplex.append([])
          for i in range(0,projSize,2):
            self.projComplex[proj].append( complex(self.projections[proj][i],self.projections[proj][i+1]) )

        print("Num projections " + str(len(self.projComplex)))
        for projection in self.projComplex:
          # split into 'ysize' (3) projections
          for y in range(1,self.ysize+1):
            axis = projection[self.xsize*(y-1):self.xsize*y]
            inverseft = scipy.fftpack.ifft(axis) #,npts)
            self.fts.append( scipy.fftpack.fftshift(inverseft) )
            self.projRaw.append(axis)
        print("Num ffts " + str(len(self.fts)))


def getStats(valueList,valueNames):
    """Return stats of given list of sublists, for each index in the sublists

     - Sublists must be of the same length
     - valueNames must be of the same length as sublists
    Example:
    valueList: [ [100,40,20], [110,42,18] ] # [[snr_x, snr_y, snr_z], [snr_x, snr_y, snr_z]]
    valueNames: ["snr_x", "snr_y", "snr_y"]
    """
    means = np.mean(valueList,0)
    mins = np.amin(valueList,0)
    maxs = np.amax(valueList,0)
    stds = np.std(valueList,0)
    statString = ''
    for idx,name in enumerate(valueNames):
        statString += "{0} Mean: {1:.2f} StdDev: {2:.2f} Min: {3:.2f} Max: {4:.2f}\n".format(
        name,means[idx],stds[idx],mins[idx],maxs[idx] )
    return statString

def getPeaks(rdr):
    """Return peak coordinates, SNRs, and amplitudes from all the projections read into RawReader"""
    allSnrs = []
    allCoords = []
    allPkAmps = []
    numProj = len(rdr.projComplex)
    for i in range(numProj):
        snr_proj = []
        coord_proj = []
        pkAmp_proj = []
        for j in range(0,rdr.ysize):
            mag = np.abs(rdr.fts[i*rdr.ysize+j])
            (peakInd,snr,peak) = getPeakInfo(mag)
            pkCoord = getCoord(peakInd,rdr.xsize,rdr.fieldOfView)
            snr_proj.append(snr)
            coord_proj.append(pkCoord)
            pkAmp_proj.append(peak)
        allSnrs.append(snr_proj)
        allCoords.append(coord_proj)
        allPkAmps.append(pkAmp_proj)
    return (allCoords, allSnrs, allPkAmps)

def main():
    parser = OptionParser(usage=__doc__)
    parser.add_option("-p", "--plot-save", action="store_true", dest="saveplots",help="save plots to files, no gui", default=False)
    parser.add_option("-c", "--coord-save", action="store_true", dest="savecoords",help="save coordinates to files, no gui", default=False)
    parser.add_option("-l", "--legacy_version", metavar="LEGACY_VERSION",dest="legacy", type="choice",
        help="read legacy files missing trig, resp, and/or timestamp values [0,1,2,3]",default='0',
        choices=['0','1','2','3'])
    parser.add_option("-s", "--stemless", action="store_true", dest="stemless", help="stemless - do not display vertical red lines for peak values", default=False)
    parser.add_option("-y", "--ylim", dest="ylim",help="y-axis limit", metavar="YLIM", type="int", default=100)
    parser.add_option("-f", "--statsfile", action="store_true", dest="statsfile",help="save stats to files, no gui", default=False)
    parser.add_option("-x", "--statsprefix",dest="statsPrefix",metavar="PREFIX", help="Prefix for stats file", default="")
    (options,args) = parser.parse_args()

    if (len(args) < 1):
        print(parser.print_help())
        sys.exit(0)
    rawFile = args[0]
    
    rdr = RawReader(rawFile,legacy_version=int(options.legacy))
    rdr.readFile()
    if len(rdr.fts) == 0:
        print("Nothing to see here. Exiting.")
        sys.exit(0)

    if (options.statsfile):
        allCoords,allSnrs,allPkAmps = getPeaks(rdr)
        coordStats = getStats(allCoords,["X_coord", "Y_coord", "Z_coord"])
        snrStats = getStats(allSnrs,["SNR_X", "SNR_Y", "SNR_Z"])
        ampStats = getStats(allPkAmps,["X_amp", "Y_amp", "Z_amp"])
        print( coordStats )
        print( snrStats )
        print( ampStats )
        fbase,fext = os.path.splitext(rawFile)
        if options.statsPrefix:
            fprefix = options.statsPrefix
        else:
            fprefix = os.path.basename(fbase)
        dirname = os.path.dirname(os.path.abspath(rawFile))
        statFile = open(dirname + "/" + fprefix + '-stats.txt','w')
        statFile.write(coordStats + '\n')
        statFile.write(snrStats + '\n')
        statFile.write(ampStats)
        print("Done.")
        sys.exit(0)

    plotter = ProjectionPlot(rdr.fts,rdr.xsize,rdr.fieldOfView,trigTimes=rdr.triggerTimes,
        respArr=rdr.respPhases,timestamps=rdr.timestamps,ysize=rdr.ysize,
        drawStems=(not options.stemless),ylim=options.ylim,pgArr=rdr.pg,
        ecg1Arr=rdr.ecg1,ecg2Arr=rdr.ecg2)


    #Save to files:
    if (options.saveplots or options.savecoords):
      coordFile = None
      allCoords = []
      allSnrs = []
      if options.savecoords:
        fbase,fext = os.path.splitext(rawFile)
        coordFile = open(fbase + '-coords.txt','w')
      for i in range(len(rdr.projComplex)):
        (coords,snrs) = plotter.showProj(i,options.saveplots,options.savecoords,coordFile)
        allCoords.append(coords)
        allSnrs.append(snrs)
        sys.stdout.write("\rSaved projection %i" % i)
        sys.stdout.flush()
      if coordFile:
        coordFile.close()
      print('\n')
      coordStats = getStats(allCoords,["X_coord", "Y_coord", "Z_coord"])
      snrStats = getStats(allSnrs,["SNR_X", "SNR_Y", "SNR_Z"])
      print( coordStats )
      print( snrStats )
      print("Done.")
    else: # launch plotter GUI
      plotter.launchGUI()

    sys.exit(0)

if __name__ == "__main__":
    main()
