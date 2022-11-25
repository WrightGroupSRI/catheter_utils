"""This module contains things related to reading and manipulating projections."""

import collections
import glob
import logging
import os
import re
import struct

import numpy
import pandas
import scipy
import scipy.fftpack

logger = logging.getLogger(__name__)


def reconstruct(raw):
    """Reconstruct the raw signals into 'projections'.
    These are 1D magnitude images."""
    return numpy.abs(reconstruct_complex(raw))


def reconstruct_complex(raw):
    """Reconstruct the raw signals into 'complex projections'.
    These are the (inverse) FT of the raw signals."""
    images = scipy.fftpack.ifft(raw, axis=1)
    images = scipy.fftpack.fftshift(images, axes=1)
    return images


def discover_raw(path, validate=False):
    """Check for possible projections files in the given directory, and
    summarize their contents.

    Returns (discoveries, unknown) where:
    `discoveries` is a pandas.DataFrame where rows describe individual
        projections and columns are given by:
            scheme: str     # how projections are related,
            recording: int  # the recording number,
            coil: int       # the coil that the projection was acquired on,
            axis: int       # the axis of the projection,
            dither: int     # the dither index (0 for sri),
            filename: str   # the file that the projection is in,
            index: int      # the index of the projection in the file
        If `validate` is set to True then there will be additional columns:
            corrupt: bool   # is the file corrupt
            expected: bool  # projections in the file agree with expectation
            version: int    # the inferred file version

    `unknown` is a list of ".projections" filenames that do not follow our
        file naming conventions. What do they contain? Ask somebody else.
    """

    # Data that we have discovered. Will become rows in a table.
    discoveries = []
    unknown = []
    for name in sorted(glob.glob(os.path.join(path, "*.projections"))):
        discovery = guess_contents_raw(name)
        if discovery is not None:
            discoveries.extend(discovery)
        else:
            unknown.append(name)

    if validate:
        expectation_for_scheme = {"sri": 3, "fh": 1}
        cache = {}
        for discovery in discoveries:
            filename = discovery["filename"]
            try:
                c, n, v = cache[filename]
            except KeyError:
                _, ps, v, c = read_raw(filename, allow_corrupt=True)
                n = len(ps[0])
                cache[filename] = c, n, v

            discovery.update(
                corrupt=c,
                version=v,
                expected=(n == expectation_for_scheme[discovery["scheme"]])
            )

    if len(discoveries) == 0:
        discoveries = pandas.DataFrame(
            columns=["scheme", "recording", "coil", "axis", "dither", "filename", "index"]
        )
    else:
        discoveries = pandas.DataFrame(discoveries)

    # TODO
    #   check that all the required rows are there? E.g., FH requires 4
    #   projections per coil and these come from different files.

    return discoveries, unknown


_RAW_SRI_RE = re.compile(r"cathcoil(?P<coil>\d+)-(?P<recording>\d+).projections")
_RAW_FE_RE = re.compile(r"cathHVC(?P<coil>\d+)P(?P<axis>\d+)D(?P<dither>\d+)-(?P<recording>\d+).projections")
_ProjectionInfo = collections.namedtuple(
        "_ProjectionInfo",
        ["scheme", "recording", "coil", "axis", "dither", "filename", "index"])


def guess_contents_raw(filename):
    """Guess the projection content of a raw projections file with the given
    filename. The guess is based on convention.

    Returns a list of dicts providing info about what each projection is. """

    def unpack(m, fields):
        return {x: int(m.group(x)) for x in fields}

    match = _RAW_SRI_RE.search(filename)
    if match:
        matched_fields = unpack(match, ["coil", "recording"])
        return [
            _ProjectionInfo(
                scheme="sri",
                axis=i,
                dither=0,
                filename=filename,
                index=i,
                **matched_fields
            )._asdict() for i in range(3)
        ]

    match = _RAW_FE_RE.search(filename)
    if match:
        matched_fields = unpack(match, ["coil", "axis", "dither", "recording"])
        return [
            _ProjectionInfo(
                scheme="fh",
                filename=filename,
                index=0,
                **matched_fields
            )._asdict()
        ]

    return None


ReadRawResult = collections.namedtuple(
    "ReadRawResult",
    "meta raw legacy_version corrupt")


def read_raw(filename, legacy_version=None, allow_corrupt=False):
    """Read the raw projection data from the requested file.

    If `legacy_version` is None then this will try to infer the correct version.
    Otherwise it will read the file using the requested version number.

    If `allow_corrupt` is True then this won't raise an exception when
    encountering an apparently corrupt file, and will instead log a
    warning and return as much data as could be read.

    Returns ReadRawResult(meta, raw, version, corrupt) where:
        'meta' is a pandas.DataFrame containing information about each sample;
        'raw' is a list of 2D complex valued numpy arrays with readouts in rows;
        'version' is the inferred or requested legacy_version; and
        'corrupt' indicates an apparently corrupt file.
    """
    logger.debug("read_raw('%s', legacy_version=%s)", filename, legacy_version)

    with open(filename, "rb") as fp:
        if legacy_version is not None:
            return _read_legacy_version(fp, legacy_version, allow_corrupt)

        else:
            for version in [0, 3, 2, 1]:
                try:
                    return _read_legacy_version(fp, version, allow_corrupt)
                except (ProjectionFileVersionCheck, ProjectionFileCorrupted):
                    fp.seek(0)

    raise ProjectionFileVersionCheck("unable to infer legacy_version")


def _read_legacy_version(fp, version, allow_corrupt):
    if version == 0:
        _check_file_header(fp)

    meta = []
    raw = []
    corrupt = False

    try:
        read_header = _READER_FOR_LEGACY[version]
        while True:
            head = read_header(fp)
            body = _read_body(head.xsize, head.ysize, fp)

            meta.append(head._asdict())
            raw.append(body)

    except ProjectionFileCorrupted:
        if allow_corrupt:
            logger.warning("file appears to be corrupt")
            corrupt = True
        else:
            raise

    except _FileEnd:
        pass

    meta = pandas.DataFrame(meta)

    # These should only change if the sequence changes which would disrupt
    # recording. That means this is probably a decent way of sanity checking
    # the file's version.
    if (meta["xsize"].nunique() != 1 or
            meta["ysize"].nunique() != 1 or
            meta["zsize"].unique() != [1]):
        raise ProjectionFileVersionCheck("file version appears to be incorrect")

    meta.drop(columns=["xsize", "ysize", "zsize"], inplace=True)
    return ReadRawResult(meta, raw, version, corrupt)


class _FileEnd(Exception):  # appropriate base? do we even care?
    """Raised to indicate file reading is complete."""


class ProjectionFileCorrupted(ValueError):
    """Raised to indicate that the file is 'corrupt'."""


class ProjectionFileVersionCheck(ValueError):
    """Raised to indicate that the file failed its version check."""


def _read(fp, count):
    """Read data from fp. Make sure we have the right amount of data."""
    h = fp.read(count)
    if len(h) == 0:
        raise _FileEnd()
    if len(h) < count:
        raise ProjectionFileCorrupted("expected {} bytes but received {}".format(count, len(h)))
    return h


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Sample headers contain info on the size of the readout, the number of
# readouts, and the physiological signals recorded during the readout.


# Most fields are "d" for double.
_FIELD_KINDS = collections.defaultdict(lambda: "d")
_FIELD_KINDS.update({"xsize": "i", "ysize": "i", "zsize": "i", "timestamp": "q"})


def _make_header_reader(name, fields):
    """Headers all contain similar data and can be read in *almost* the same way."""
    kind = collections.namedtuple(name, fields)
    fmt = ">" + "".join(_FIELD_KINDS[k] for k in fields)
    size = struct.calcsize(fmt)

    def reader(fp):
        return kind(*struct.unpack(fmt, _read(fp, size)))

    return reader


_read_legacy_v1_header = _make_header_reader(
    "_LEGACY_V1_HEADER",
    ["xsize", "ysize", "zsize", "fov"]
)

_read_legacy_v2_header = _make_header_reader(
    "_LEGACY_V2_HEADER",
    ["xsize", "ysize", "zsize", "fov", "trig", "resp"]
)

_read_legacy_v3_header = _make_header_reader(
    "_LEGACY_V3_HEADER",
    ["xsize", "ysize", "zsize", "fov", "timestamp", "trig", "resp"]
)

_read_header = _make_header_reader(
    "_HEADER",
    ["xsize", "ysize", "zsize", "fov", "timestamp", "trig", "resp", "pg", "ecg1", "ecg2"]
)

_READER_FOR_LEGACY = {
    1: _read_legacy_v1_header,
    2: _read_legacy_v2_header,
    3: _read_legacy_v3_header,
    0: _read_header,
}


def _read_body(pixel_count, readout_count, fp):
    value_count = pixel_count*readout_count  # complex values
    byte_count = 8*2*pixel_count*readout_count  # 8 bytes per real, 2 real per complex
    body = numpy.frombuffer(_read(fp, byte_count), dtype=numpy.complex128, count=value_count)
    body = body.newbyteorder(">")
    body = body.reshape((readout_count, pixel_count))
    return body.astype(numpy.complex128)


def _check_file_header(fp):
    # Check the header for legacy == 0 files.
    header_bytes = fp.peek(8)
    fmt = b''.join(struct.unpack('4c', header_bytes[0:4]))
    fmt = fmt.decode('ascii')
    version = struct.unpack('>i', header_bytes[4:8])[0]
    if fmt != "CTHX" or version != 1:
        raise ProjectionFileVersionCheck("unknown version ({}, {})".format(fmt, version))
    fp.read(8)  # Digest the header bytes


# end of read_raw details
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def fov_info(meta, raw):
    '''
    extract FOV info from pandas.Dataframe (meta), and raw numpy data (raw). Output from read_raw function
    '''
    if "fov" in meta.columns:
        pix = len(raw[0][0])
        if meta["fov"].nunique() == 1:
            fov = meta["fov"][0]
            fov = "{} mm (resolution {:.2f} mm)".format(fov, fov/pix)
        else:
            min_fov, max_fov = meta["fov"].min(), meta["fov"].max()
            fov = "{} - {} (resolution {:.2f} - {:.2f} mm)".format(min_fov, max_fov, min_fov/pix, max_fov/pix)
        return(fov)
    else:
        fov = "unknown fov"
        return(fov)

def snr(fs):
    """Estimate the given signal's SNR."""

    n = len(fs)
    peak = numpy.max(fs)

    window_size = int(n*0.078125)
    left_mean = numpy.mean(fs[:window_size])
    right_mean = numpy.mean(fs[n - window_size:])

    if left_mean > right_mean:
        stdev = numpy.std(fs[n - window_size:])
    else:
        stdev = numpy.std(fs[:window_size])

    return peak / stdev


def variance(fs, xs):
    """Estimate the given signal's variance.

    Treat fs as though it is proportional to a probability density, and
    calculate its variance."""
    # Q: should we check that fs is non-negative?

    d0 = numpy.trapz(fs, xs)
    if d0 <= 0.0:
        # signal must be flat, this is maximum variance.
        return 0.25*(xs[-1] - xs[0])**2

    # signal's mean
    x0 = numpy.trapz(fs*xs, xs)/d0

    # signal's variance
    return numpy.trapz(fs*(xs - x0)**2, xs)/d0