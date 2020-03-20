"""This module contains things related to reading and manipulating projections."""

import collections
import logging
import numpy
import pandas
import struct
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


ReadRawResult = collections.namedtuple(
    "ReadRawResult",
    "meta raw legacy_version corrupt")


def read_raw(filename, legacy_version=None, allow_corrupt=False):
    """Read the raw projection data from the requested file.

    If `legacy_version` is None then this will try to infer the correct version.
    Otherwise it will read the file using the requested version number.

    If `allow_corrupt` is true then this won't raise an exception when
    encountering an apparently corrupt file, and will instead log a
    warning and return as much data as could be read.

    Returns ReadRawResult(meta, raw, version, corrupt) where:
        'meta' is a pandas DataFrame containing information about each sample;
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

