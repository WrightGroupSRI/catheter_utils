"""Test catheter_utils.cathcoords."""

import numpy
import numpy.testing
from catheter_utils import cathcoords
from textwrap import dedent
from unittest import TestCase
from unittest.mock import patch, mock_open


class DiscoverCathcoordsFilesTest(TestCase):
    @patch("glob.glob")
    def test_discover_cathcoords_files(self, mock_glob):
        names = [
            "/someroot/somenotroot/cathcoords-coil0-0000.txt",
            "/someroot/somenotroot/cathcoords-coil1-0000.txt",
        ]
        mock_glob.return_value = names
        res = cathcoords.discover_files("somewhere")
        self.assertDictEqual(res, {0: {0: names[0], 1: names[1]}})


class ReadCathcoordsDataTest(TestCase):
    DATA = dedent(
        """
        1.0 2.0 3.0 229 1000 450 1000
        1.0 2.0 3.0 228 2000 450 1000
        1.0 2.0 3.0 240 3000 450 1000
        1.0 2.0 3.0 246 4000 450 1000
    """
    )

    @patch("builtins.open", mock_open(read_data=DATA))
    def test_read_cathcoords_data(self):
        dist, prox = cathcoords.read_pair("dist_filename", "prox_filename")

        numpy.testing.assert_array_equal(dist.coords, prox.coords)
        numpy.testing.assert_array_equal(dist.times, numpy.array([1000, 2000, 3000, 4000]))
        numpy.testing.assert_array_equal(dist.trigs, numpy.array([450, 450, 450, 450]))
        numpy.testing.assert_array_equal(dist.resps, numpy.array([1000, 1000, 1000, 1000]))
