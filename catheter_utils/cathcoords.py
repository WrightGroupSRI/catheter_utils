"""Manipulate cathcoords files."""

import glob
import logging
import numpy
import pandas
import re
from collections import defaultdict, namedtuple
import catheter_utils.geometry

logger = logging.getLogger(__name__)


def discover_files(dirname):
    """Search the given directory for cathcoords files."""

    filenames = glob.glob(dirname + "/*coil*.txt")
    regex = re.compile(r"-coil(\d+)-(\d{4}).txt")

    def extract_coil_and_id(s):
        match = regex.search(s)
        assert match is not None, "no match in regex"
        return int(match.group(1)), int(match.group(2))

    dir_info = defaultdict(dict)
    for filename in filenames:
        coil, id = extract_coil_and_id(filename)
        dir_info[id][coil] = filename

    return dir_info


Cathcoords = namedtuple("Cathcoords", "coords snr times trigs resps")
"""Object representing all the data that goes into a cathcoords file."""

_NAMES = ["X", "Y", "Z", "SNR", "Timestamp", "Trig", "Resp"]
"""Column names for pandas.DataFrames."""


def _read_file_df(filename):
    """Get a pandas dataframe for a cathcoords file."""
    logger.debug("Reading cathcoords file %s", filename)
    with open(filename, "r") as f:
        return pandas.read_csv(f, sep="\\s+", header=None, names=_NAMES)


def _write_file_df(filename, df):
    """Write dataframe into a CSV."""
    logger.debug("Writing cathcoords file %s", filename)
    with open(filename, "w") as f:
        df.to_csv(f, sep=" ", header=False, index=False, float_format="%.4f")


def read_file(filename):
    """Extract data from a cathcoords file."""

    df = _read_file_df(filename)

    # Extract desired data
    coords = df[["X", "Y", "Z"]].values
    snr = df["SNR"].values
    times = df["Timestamp"].values
    trigs = df["Trig"].values
    resps = df["Resp"].values

    return Cathcoords(coords, snr, times, trigs, resps)


def read_pair(dist_filename, prox_filename):
    """Extract data from a pair of cathcoords files and do simple checks for
    consistency.
    
    Different coil files may have a different number of rows for the same
    recording. This will match rows using timestamps so that coords, snr, etc
    are paired up correctly.
    """

    logger.debug("Reading cathcoords pair.")

    # Read in the files
    dist_df = _read_file_df(dist_filename)
    prox_df = _read_file_df(prox_filename)

    # Only use points acquired at the same time, sometimes different coils
    # have a slightly different amount of data.
    join_df = pandas.merge(
        dist_df, prox_df, on="Timestamp", suffixes=("_dist", "_prox")
    )

    # Sanity check columns that should match.
    for key in ["Trig", "Resp"]:
        if not join_df[key + "_prox"].equals(join_df[key + "_dist"]):
            count = numpy.sum(
                join_df[key + "_prox"].values - join_df[key + "_dist"].values != 0
            )
            logger.warn(
                'Missmatch in "%s": %s row(s) are not equal. Prox: "%s" Dist: "%s".',
                key,
                count,
                prox_filename,
                dist_filename,
            )

    # Extract desired data
    dist_coords = join_df[["X_dist", "Y_dist", "Z_dist"]].values
    prox_coords = join_df[["X_prox", "Y_prox", "Z_prox"]].values
    dist_snr = join_df["SNR_dist"].values
    prox_snr = join_df["SNR_prox"].values
    times = join_df["Timestamp"].values
    trigs = join_df["Trig_dist"].values
    resps = join_df["Resp_dist"].values

    dist = Cathcoords(dist_coords, dist_snr, times, trigs, resps)
    prox = Cathcoords(prox_coords, prox_snr, times, trigs, resps)
    return dist, prox


def write_file(filename, obj):
    """Write catheter coil data to the specified file."""
    df = pandas.DataFrame(
        {
            "X": obj.coords[:, 0],
            "Y": obj.coords[:, 1],
            "Z": obj.coords[:, 2],
            "SNR": obj.snr,
            "Timestamp": obj.times,
            "Trig": obj.trigs,
            "Resp": obj.resps,
        },
        columns=_NAMES,
    )
    _write_file_df(filename, df)


def make_filename(index, coil):
    """Format a cathcoords filename."""
    return "cathcoords-coil{}-{:04d}.txt".format(coil, index)

def get_centroid_mean(coords):
    """Return the centroid of the given coordinate array. Uses the mean."""
    return numpy.mean(numpy.array(coords),axis=0)

def get_centroid_median(coords):
    """Return the centroid of the given coordinate array. Uses the median."""
    return numpy.median(numpy.array(coords),axis=0)

def get_distances_from_centroid(coords, use_mean=True):
    """Return a list of distances from the centroid, and centroid, using the mean centroid by default.
    Set use_mean to False to use median centroid."""
    centroid = numpy.zeros(1)
    if use_mean:
        centroid = get_centroid_mean(coords)
    else:
        centroid = get_centroid_median(coords)
    diffs = coords - centroid
    return numpy.linalg.norm(diffs,axis=1),centroid

def get_tip_variance(distal_file, proximal_file, geometry=None, dof=1):
    '''returns the variance of the tip coordinates

    From distal and proximal coordinate files: computes
    the tip locations and the centroid of the tip locations

    Returns the variance of the euclidean distances between
    tip locations and their centroid. This gives an estimate
    for the spread of the tip points.

    distal_file: text file of distal coordinates
	proximal_file: text file of proximal coordinates
	geometry: the geometry of the coils, estimated if not passed in
    dof: degrees of freedom for variance calculation, 1 by default
    '''
    distal_coords = read_file(distal_file).coords
    proximal_coords = read_file(proximal_file).coords
    if (geometry is None):
        geometry = catheter_utils.geometry.estimate_geometry(distal_coords,proximal_coords)
    fit_results = geometry.fit_from_coils_mse(distal_coords, proximal_coords)
    distances,_ = get_distances_from_centroid(fit_results.tip)
    return numpy.var(distances, ddof=1)
