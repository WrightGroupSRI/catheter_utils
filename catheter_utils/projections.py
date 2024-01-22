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

def _read_raw_and_reconstruct(fn):
    """Read projection filenames and reconstruct raw signals
    to projections
    """
    meta, projections, _, _ = read_raw(fn, allow_corrupt=False)
    for i, signal in enumerate(projections):
        projections[i] = reconstruct(projections[i])
    return meta, projections

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
class FindData:
    '''
    Extract coil projection data and synchronized rows of distal & proximal pairs
    '''
    def __init__(self, toc, recording, distal, proximal, dither):
        selection = toc[
            (toc.recording == recording) &
            (toc.dither == dither) &
            toc.coil.isin({distal, proximal})
        ]

        filename_to_data = {
            fn: _read_raw_and_reconstruct(fn)
            for fn in sorted(selection.filename.unique())
        }

        # Need to impose some notion of "simultaneity" to form groups of projections.
        # I tried grouping by timestamp, but FH projections do not share timestamps.

        n = max(len(meta) for meta, _ in filename_to_data.values())

        # Make sure all of the component data sets have the same length.
        for fn, (meta, projections) in filename_to_data.items():
            assert len(meta) == len(projections), "???"
            while len(meta) < n:
                meta=pandas.concat([meta, meta.tail(1)], ignore_index=True)
                projections.append(projections[-1])
            filename_to_data[fn] = meta, projections

        self._timestamp = numpy.mean(
            numpy.vstack([meta.timestamp for meta, _ in filename_to_data.values()]).T,
            axis=1,
        ).astype(int)

        self._alltimestamp = numpy.vstack([meta.timestamp for meta, _ in filename_to_data.values()])

        try:
            self._resp = numpy.mean(
                numpy.vstack([meta.resp for meta, _ in filename_to_data.values()]).T,
                axis=1
            )
        except AttributeError:
            self._resp = numpy.zeros(n)

        try:
            self._trig = numpy.mean(
                numpy.vstack([meta.trig for meta, _ in filename_to_data.values()]).T,
                axis=1
            )
        except AttributeError:
            self._trig = numpy.zeros(n)

        data = {}
        axes = set()
        n = None
        for row in selection.itertuples():
            meta, projections = filename_to_data[row.filename]
            data[(row.coil, row.axis)] = meta, projections, row.index
            axes.add(row.axis)
            if n is None:
                n = len(projections)
            else:
                m = min(n, len(projections))
                if m != n:
                    logger.warning("different readout lengths")
                    n = m

        if n is None:
            raise ValueError("expected a length")

        if axes == {0, 1, 2}:
            # this is an SRI style recording
            pass
        elif axes == {0, 1, 2, 3}:
            # this is a FH style recording
            pass
        else:
            raise ValueError("Expected SRI or FH style recordings")

        # SNR for each coil & axis: an array of snrs corresponding to each readout
        snr_per_readout = {}
        for (coil,axis) in data.keys():
            meta, projections, index = data[(coil,axis)]
            snr_array = []
            for i in range(n): # for each readout
                snr_array.append(snr(projections[i][index]))
            snr_per_readout[(coil,axis)] = snr_array

        self._data = data
        self._axes = sorted(axes)
        self._distal = distal
        self._proximal = proximal
        self._proximal_cache = {}
        self._distal_cache = {}
        self._n = n
        self._snrs_per_readout = snr_per_readout # SNR array per coil and axis
        self._snr_combo_method = 'min'
        self._snr = None # combined SNR over axes, per coil
        self.combine_snr()

    def __len__(self):
        return self._n

    def _get_coil_data(self, coil, i, cache):
        assert i < len(self), "index out of range"
        try:
            return cache[i]
        except KeyError:
            data = []
            for j in range(len(self._axes)):
                meta, projections, index = self._data[(coil, j)]
                fov = meta.fov[i]
                fs = projections[i][index]
                xs = numpy.linspace(-fov / 2, fov / 2, len(fs))
                data.append((fs, xs))
            cache[i] = data
            return data

    def get_timestamp(self, axis, readout):
        return self._alltimestamp[axis, readout]

    def get_proximal_data(self, i):
        return self._get_coil_data(self._proximal, i, self._proximal_cache)

    def get_distal_data(self, i):
        return self._get_coil_data(self._distal, i, self._distal_cache)

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def resp(self):
        return self._resp

    @property
    def trig(self):
        return self._trig

    @property
    def snrs_per_readout(self):
        ''' Return SNRs from each readout
        Returns a dict whose keys are the coil & axis (both ints) and each
        value is an array of SNRs for this recording, ordered by time

        Unlike the snr property, snrs_per_readout does not combine the SNR over axes
        '''
        return self._snrs_per_readout

    @property
    def snr(self):
        ''' Return a dict of SNRs by coil
        - Each value is an arrays of snrs, one for each set of axes over the recording,
        ordered by time
        - Each SNR is a combination of the SNRs of each axis projection
        - see combine_snr & snr_combo_method for details
        '''
        return self._snr

    @property
    def snr_combo_method(self):
        ''' The SNR combination method defines how SNRs are combined across axes
          - Should be one of: 'min', 'median', 'mean'
          - The default value on construction is 'min'
        WARNING: on setting, it may take some time to compute the combined SNRs
        
        '''
        return self._snr_combo_method

    @snr_combo_method.setter
    def snr_combo_method(self, value):
        if not value in ['min','median','mean']:
            raise ValueError('Unexpected value for SNR combination method')
        if (self._snr_combo_method != value):
            self._snr = None
            self._snr_combo_method = value
            self.combine_snr()

    def combine_snr(self):
        ''' Compute dict of snrs for this projection: the keys are the coils,
        and each value is an arrays of snrs, one for each set of axes over the recording,
        ordered by time

        This should not need to be called from outside the class: it is called
        internally when the projections are read and when the snr_combo_method is changed

        SNRS are combined over axes: 3 axes (X, Y, and Z) for the basic "SRI" 3-projection sequence
        and 4 axes (4 diagonals of a cube) for the "FH" hadmard-multiplexed sequence
        The snr_combo_method defines how the SNRs are combined.

        These snr combined values are stored in the snr property. They are computed when this
        method is called and saved until the snr_combo_method is changed
        '''
        snr_per_coil = {}
        snr_per_coil[self._distal] = []
        snr_per_coil[self._proximal] = []
        combo_fxn = numpy.min
        if (self.snr_combo_method == 'mean'):
            combo_fxn = numpy.mean
        elif (self.snr_combo_method == 'median'):
            combo_fxn = numpy.median
        elif (self.snr_combo_method != 'min'):
            raise ValueError('Unexpected value for SNR combination method')
        if (self._snr == None):
            # Compute combined snr array
            for i in range(len(self)):
                snrs = {}
                snrs[self._distal] = []
                snrs[self._proximal] = []
                for (data_coil,axis) in self._data.keys():
                    snrs[data_coil].append(self._snrs_per_readout[(data_coil,axis)][i])
                snr_per_coil[self._distal].append(combo_fxn(snrs[self._distal]))
                snr_per_coil[self._proximal].append(combo_fxn(snrs[self._proximal]))
            self._snr = snr_per_coil

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