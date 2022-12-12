# Catheter Utils

This project contains utils related to our catheter tracking projects.

## `cathcoords`

Contains things for manipulating "cathcoords" files and data.
These are localized catheter coordinates combined with metadata like trigger time and snr.

## `geometry`

Contains things for reasoning about the catheter's geometry.
Things like distances between distal, proximal, and tip locations, as well as helpers for extrapolating the tip.

## `localization`

Contains things for producing estimates of a coil location from projection data.
This includes single projection localization all the way up to full catheter location fit using constrains
and snr-like error weighting.

## `metrics`

Contains evaluation and display of tracking sequence error metrics. Metric methods include Bias and Chebyshev. 

## `projections`

Contains code for reading raw projection data, as well as basic calculations like snr.

## Tests

**Make sure that your current terminal/CMD directory is the catheter_utils folder**

Can run all tests with the command `python -m unittest discover tests`.

## `test_localization`

The localization test ensures that the current localization algorithms match the original/target localization 
algorithms output for the same dataset.

**Note: Currently only tests the PNG (peak-normed gaussian) algorithm, to ensure it outputs the same coordinates as 
when originally run based on data acquired and reconstructed during Jay Soni's experiments in April 2020**

Based on testing from multiple systems/numpy versions, the test precision is 2 decimal places. 
Decimal place should not exceed precision of test data. 

Example Code: np.testing.assert_almost_equal(output, target, decimal=2) # check for 2 decimal place precision/accuracy between current output and target 


### Adding New Test Data
1. Copy test data and target algorithm folder to catheter_utils/tests/testdata
1. Add a copy of 'localization_settings_template.yaml' file (from catheter_utils/tests/testdata) to each added test data folder and edit file parameters accordingly
1. Test settings for the test data and localization algorithms to be modified in the localization_settings_template.yaml file


See /testdata/FH_512_dithered_original-2020-07-09T12_16_32.222 for FH example format,

See /testdata/SRI_April-2020-07-09T12_20_56.765 for SRI example format

Example: Directory Structure
- /testdata/SRI_April-2020-07-09T12_20_56.765
  - .projections files
  - localization_settings.yaml
  - /target
      - png
      - other algorithm folders etc..

localization_settings.yaml template SRI example:
```yaml
settings:
  #path to tracking sequence test data folder
  testdata_path: "./tests/testdata/SRI_April-2020-07-09T12_20_56.765"
  #path to algorithm target folder
  target_path: "./tests/testdata/SRI_April-2020-07-09T12_20_56.765/target"
  #catheter coils
  distal_coil_index : 7
  proximal_coil_index : 6
  #dither (0 if tracking sequence test data not dithered)
  dither_index : 0
  #algorithm parameters - width: window size, sigma: std
  width: 3.5
  sigma: 0.75
```
Note: Currently width and sigma are global parameters for all localization algorithms. Some localization algorithms are not affected by/ignore changes to width/sigma (ex. peak, centroid). 

### Test Localization Results
- Summary of latest test saved in /tests/testdata/test_localization_summary.csv
- Last test failure saved in /tests/testdata/test_localization_fail.log

### Developer Notes:

test_localization.py
- To add more localization algorithms to test function "def localize_test", refer to loc_fns structure found in [cathy](http://panoptes.sri.utoronto.ca:8088/wright-group/cathy/blob/master/cathy/cli.py#L497)
- To add new test to the test case "algorithm_localizer(TestCase)", add new def function with name 'test_' 

Example

class algorithm_localizer(TestCase)
...

def setUp(self):
...

def test_localize(self): 
...

def test_XXXXXXXX(self): ...