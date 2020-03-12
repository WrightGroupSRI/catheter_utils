# Catheter Utils

This project contains utils related to our catheter tracking projects.

## Tests

Can run with the command `python -m unittest discover tests`.

# Hawk Reader
Scripts to read and reconstruct projection data from RTHawk tracking sequences saved
using RthReconImageExporter.

Each projection file contains a recording from a single coil; in the 
case of the basic active tracking sequence, it contains X, Y, and Z projections
in that order.

## Requirements
- Python 3+
- matplotlib
- scipy

**NOTE** This code has only been tested with Python 3.6 on Ubuntu 14.04 & 16.04.

### Set up the environment
This will likely **NOT** be necessary if you are already using the recommended packages above. If the code does not work, you can set up your packages as described below.

If you are using [conda](https://conda.io/docs/), you can replicate the environment using the included environment.yml file:
```
conda env create -f environment.yml
```
If you're on conda 4.4+:
```
conda activate hawkr
```
Otherwise, using conda < 4.4 on Mac OS X / Linux:
```
source activate hawkr
```
## Scripts
Two scripts can be run from the command line.
### readRthRaw
Reads and plots catheter raw data from a single projection recording. It can either display the plots in a window or save files:
- the plots in PNG format
- the peak locations
- stats on the peaks and SNRs

The "legacy header" options are only for older recordings.
 
### reconDirs
Reconstructs projection files in batch mode - this will find projection files given a pattern within subdirectories of the current directory and optionally:
- save projection plots from each projection recording
- create a movie of each projection recording
- save basic stats from each projection recording

This works on systems with theora_png2theora installed.

### Other files
These are used by the scripts:
- *snrCalc.py*: Calculates the SNR of a projection
- *projPlot.py*: Plots projections

## Help Docs
To view detailed documentation, run help in a python console:
```python
import readRthRaw
help(readRthRaw)
import reconDirs
help(reconDirs)
```
To show help on the command-line options, you can run the scripts without any arguments, for example: ```./readRthRaw.py```

or: ```./reconDirs.py```

## Usage Examples
More usage examples available in the module help.

**To show plots from a projection file:**
```
/path/to/readRthRaw.py /mydir/data/cathcoil4-0000.projections
```
**To show plots from a projection file with a y-axis limit of 600 and no red stems (which indicate peaks):**
```
/path/to/readRthRaw.py /mydir/data/cathcoil4-0000.projections -s -y 600
```
**To save projection plots and stats files from a projection file:**
```
/path/to/readRthRaw.py /mydir/data/cathcoil5-0001.projections -p -f
```

**To reconstruct coil 4 & coil 5 projections within the subdirectories of the current directory and save these plots and stats to files:**
```
/path/to/reconDirs.py -p -f
```