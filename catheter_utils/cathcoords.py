"""Manipulate cathcoords files."""

import glob
import numpy
import pandas
import re
from collections import defaultdict


def discover_cathcoords_files(dirname):
    """WHAT DOES IT DO"""
    filenames = glob.glob(dirname + "/*coil*.txt")
    regex = re.compile(r"-coil(\d+)-(\d{4}).txt")

    def extract_coil_and_id(s):
        match = regex.search(s)
        assert match is not None, "no match in regex"
        return int(match.group(1)), int(match.group(2))

    byid = defaultdict(dict)
    for filename in filenames:
        coil, id = extract_coil_and_id(filename)
        byid[id][coil] = filename

    return byid


def read_cathcoords_data(dist_filename, prox_filename):
    """WHAT DOES IT DO"""

    def _read_file(filename):
        with open(filename, "r") as f:
            names = ["X", "Y", "Z", "dunno", "Timestamp", "Trig", "Resp"]
            return pandas.read_csv(f, sep="\\s+", header=None, names=names)

    # Read in the files
    dist_df = _read_file(dist_filename)
    prox_df = _read_file(prox_filename)

    # Only use points acquired at the same time
    join_df = pandas.merge(
        dist_df, prox_df, on="Timestamp", suffixes=("_dist", "_prox")
    )

    # Sanity check some data
    for key in ["Trig", "Resp"]:
        if not join_df[key + "_prox"].equals(join_df[key + "_dist"]):
            count = numpy.sum(
                join_df[key + "_prox"].values - join_df[key + "_dist"].values != 0
            )
            print(
                'Missmatch in "{}":\n  prox: {}\n  dist: {}\n  {} row(s) are not equal.'.format(
                    key, prox_filename, dist_filename, count
                )
            )

    # Extract desired data
    dist_coords = join_df[["X_dist", "Y_dist", "Z_dist"]].values
    prox_coords = join_df[["X_prox", "Y_prox", "Z_prox"]].values
    trigs = join_df["Trig_dist"].values
    resps = join_df["Resp_dist"].values

    # probably would make more sense to just return timestamps ...
    dts = 0.001 * join_df.diff()["Timestamp"].values
    dts[0] = 0

    return dist_coords, prox_coords, dts, trigs, resps
