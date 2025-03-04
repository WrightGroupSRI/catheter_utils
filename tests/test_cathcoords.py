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

class CathcoordsCentroid(TestCase):
    coords = numpy.array([[1.0,2.0,3.0]])

    near_points = [[1.0,4.0,5.0],
                   [2.0,4.0,5.0],
                   [3.0,4.0,5.0]]

    def test_centroid_single(self):
        ctr = cathcoords.get_centroid_mean(self.coords)
        numpy.testing.assert_array_equal(ctr, self.coords[0])

    def test_centroid_same(self):
        same_coords = numpy.append(self.coords, [self.coords[0]],axis=0)
        ctr = cathcoords.get_centroid_mean(same_coords)
        numpy.testing.assert_array_equal(ctr, self.coords[0])

    def test_centroid_mean(self):
        coord_pts = numpy.append(self.coords, [[3.0,2.0,3.0]],axis=0)
        ctr = cathcoords.get_centroid_mean(coord_pts)
        numpy.testing.assert_array_equal(ctr, [2.0,2.0,3.0])

    def test_centroid_median(self):
        two_pts = numpy.append(self.coords, [[3.0,2.0,3.0]],axis=0)
        three_pts = numpy.append(two_pts, [[6.0,2.0,3.0]],axis=0)

        ctr = cathcoords.get_centroid_median(three_pts)
        numpy.testing.assert_array_equal(ctr, [3.0,2.0,3.0])

    def test_centroid_distances(self):
        distances,_ = cathcoords.get_distances_from_centroid(self.near_points)
        numpy.testing.assert_array_equal(distances, [1,0,1])

class CathcoordsTipVarianceTest(TestCase):
    DATA_1 = dedent(
        """
        1.0 2.0 3.0 229 1000 450 1000
        1.0 2.0 3.0 228 2000 450 1000
        1.0 2.0 3.0 240 3000 450 1000
        1.0 2.0 3.0 246 4000 450 1000
    """
    )

    DATA_2 = dedent(
        """
        0.0 2.0 3.0 229 1000 450 1000
        1.0 2.0 3.0 229 1000 450 1000
        3.0 2.0 3.0 228 2000 450 1000
        4.0 2.0 3.0 240 3000 450 1000
    """
    )
    @patch("builtins.open", mock_open(read_data=DATA_1))
    def test_tip_variance_zero(self):
        var = cathcoords.get_tip_variance("dist_filename", "prox_filename")

        self.assertEqual(var, 0.0, "Tip variance non zero")

    @patch("builtins.open", mock_open(read_data=DATA_2))
    def test_tip_variance(self):
        var = cathcoords.get_tip_variance("dist_filename", "prox_filename")

        self.assertAlmostEqual(var, 1/3.0, places=5, msg="Tip variance should be 1/3")