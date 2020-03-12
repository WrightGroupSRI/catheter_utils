#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" SNR calculations for projections
"""

import scipy

# Projection is the fft-ed projection
# Peak is the intensity of the peak
# window_size must be less than the size of the projection array
def getSNR(projection, peak, window_size=40):
  leftMean = scipy.mean(projection[:window_size])
  rightMean = scipy.mean(projection[len(projection)-window_size:])
  stdev = 0
  if leftMean > rightMean:
    stdev = scipy.std(projection[len(projection)-window_size:])
  else:
    stdev = scipy.std(projection[:window_size])
  if stdev == 0:
      return 0
  return (peak * 1.0) / stdev
