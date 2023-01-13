"Test catheter_utils.localization"
import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from catheter_utils import localization, projections, cathcoords
from unittest import TestCase
import os
import glob
from ruamel.yaml import YAML
import logging
from schema import Schema, And, Or, Use, SchemaError

#data retrieval/call functions
def coil_testdata(src_path, distal_index, proximal_index, dither_index):
    '''retrieve data from testfolder for specified dither and distal &
    proximal coils
    '''
    data=[]
    #find projection files
    discoveries, _ = projections.discover_raw(src_path)
    #check that specified coils & dither exist in testdata set
    assert set({distal_index, proximal_index}).issubset(set(discoveries["coil"].values)) and dither_index in discoveries['dither'].values, \
        f"specified distal/proximal coils {distal_index, proximal_index}. coils found are {set(discoveries['coil'].unique())}. Dither specified {dither_index}. Dither(s) found {set(discoveries['dither'].unique())}"

    #extract projections
    for recording in sorted(discoveries.recording.unique()):
        data.append(projections.FindData(discoveries, recording, distal_index, proximal_index, dither_index))
    return(data)

def coil_targetdata(src_path, recording_index, distal_index, proximal_index):
    ''' Retrieves target algorithm files and returns target x,y,z positions
    for specified recording, distal & proximal coils
    '''
    #find coil coordinates .txt files
    cathcoord_files = cathcoords.discover_files(src_path)
    #check that specified coils exist in target/groundtruth files
    assert set({distal_index, proximal_index}).issubset(set(cathcoord_files[recording_index].keys())), \
        f" specified distal/proximal coils: {distal_index, proximal_index}, does not match target file recording {recording_index}" \
        f" coils: {tuple(cathcoord_files[recording_index].keys())}"

    distal_file = cathcoord_files[recording_index][distal_index]
    proximal_file = cathcoord_files[recording_index][proximal_index]
    #extract distal & proximal coordinates
    distal, proximal = cathcoords.read_pair(distal_file, proximal_file)
    return(np.array(distal.coords), np.array(proximal.coords))

def _localizer(fn, args, kwargs):
    '''call localization algorithm to calculate coil positions
    '''
    def _fn(d, p):
        return localization.localize_catheter(d, p, fn, args, kwargs)
    return _fn

def validate_settings_yaml(settings, file):
    '''validate user changes to localization_settings.yaml and raises error for improper values
    (checks settings against settings_schema template)
    '''
    settings_schema = Schema({
        "settings": {
            "testdata_path": And(lambda f: os.path.exists(f), error="Check that testdata_path is valid"),
            "target_path": And(lambda f: os.path.exists(f), error="Check that target_path is valid"),
            "distal_coil_index": And(lambda n: n>=0 and isinstance(n, int), error="distal coil index must be non-negative integer"),
            "proximal_coil_index": And(lambda n: n>=0 and isinstance(n, int), error="proximal coil index must be non-negative integer"),
            "dither_index": And(lambda n: n>=0 and isinstance(n, int), error=" dither index must be non-negative integer"),
            "width": And(Or(int,float), lambda n: n>=0, error="width must be a non-negative number"),
            "sigma": Or(int, float, error="sigma must be a number"),
        }
    })
    try:
        settings_schema.validate(settings)
        assert os.path.exists(settings['settings']['testdata_path']), f"{settings['settings']['testdata_path']} directory could not be found"
    except SchemaError as se:
        errors=str(list(filter(lambda x: isinstance(x, str), se.errors)))
        raise Exception("\n".join((f"Error in {file}",errors))) from None

def coil_compare(output, target, coil_name, algorithm, rec, readout, fail_count, path):
    '''check if output of localization algorithm matches target output
    '''
    try:
        np.testing.assert_almost_equal(output, target, decimal=2) #check for 2 decimal place precision/accuracy between current output and target
    except Exception as err:
        logging.error(f"{path} failed at {algorithm} {coil_name} recording {rec} readout: {readout}.")
        logging.error(f"target_xyz:{target}, calculated_xyz:{output}. Difference {target - output}")
        fail_count += 1
    return(fail_count)

#Test case
class algorithm_localizer(TestCase):
    '''
    run through localization algorithms and verify output
    '''
    def setUp(self):

        # Read all test_localization_settings.yaml files from testdata subdirectories and validate inputs
        self.settings=[]
        self.cur_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
        yaml = YAML(typ='safe')
        for file in glob.iglob(self.cur_dir+'/**/*.yaml'):
            with open(file, 'r') as stream:
                data_loaded = yaml.load(stream)
            validate_settings_yaml(data_loaded, file)
            self.settings.append(data_loaded["settings"])

        # create log file if localization algorithm test fails to match target/groundtruth values
        logging.basicConfig(level=logging.ERROR, format='%(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', handlers=[logging.FileHandler(os.path.join(self.cur_dir,'test_localization_fail.log'),
                            mode='w', delay=True)])

    def test_localize(self):
        '''test verification of localization algorithm to target values
        '''
        results=[]
        # extract configurations from each test folder settings
        for param in self.settings:
            data_records=coil_testdata(param['testdata_path'], param['distal_coil_index'], param['proximal_coil_index'], param['dither_index'])
            # localization algorithms
            '''(add other localization functions in self.loc_fns)
            See cathy/cli.py -> loc_fns. http://panoptes.sri.utoronto.ca:8088/wright-group/cathy/blob/master/cathy/cli.py#L497
            Note: some algorithms don't use arg parameters. Set as None.
            '''
            loc_fns = {
                #"peak": _localizer(localization.peak, None, None),
                #"centroid": _localizer(localization.centroid, None, None),
                #"centroid_around_peak": _localizer(localization.centroid_around_peak, None, dict(window_radius=2 * param['width'])),
                "png": _localizer(localization.png, None, dict(width=param['width'], sigma=param['sigma'])),
            }
            #test each localization algorithm. current format looks for both distal and proximal coils
            for algorithm,loc_fn in loc_fns.items():
                for rec, data in enumerate(data_records):
                    test_fail=0
                    #target/groundtruth values
                    assert os.path.exists(os.path.join(param['target_path'], algorithm)), f"{os.path.join(param['target_path'], algorithm)} directory could not be found"
                    t_distal, t_proximal=coil_targetdata(os.path.join(param['target_path'], algorithm), rec, param['distal_coil_index'], param['proximal_coil_index'])
                    for readout in range(len(data)):
                        d = data.get_distal_data(readout)
                        p = data.get_proximal_data(readout)
                        #catheter location from localization algorithm
                        coil_loc_pair=loc_fn(d, p)
                        #compare localization algorithm to targets/groundtruth values.
                        test_fail=coil_compare(coil_loc_pair[0], t_distal[readout], "distal", algorithm, rec, readout, test_fail, param['testdata_path'])
                        test_fail=coil_compare(coil_loc_pair[1], t_proximal[readout], "proximal", algorithm, rec, readout, test_fail, param['testdata_path'])
                    #test_data_folder, parameters, algorithm, pass/failed data points, recording
                    results.append([param['testdata_path'].split('/')[-1]] + list(param.values())[2:] + [algorithm, bool(test_fail==0), f"{test_fail}/{len(data)*2}", rec])

        # summary of results passed/failed
        results = pd.DataFrame(np.array(results),columns=['testdata folder'] + list(self.settings[0].keys())[2:] +
                                                         ["algorithm", "pass", "#_of_fails", "recording"])
        print("\n" + results.to_string(index=False))
        results.to_csv(os.path.join(self.cur_dir, "test_localization_summary.csv"), index=False)

        #output log of failures
        if results["pass"].str.contains('False').any():
            with open(os.path.join(self.cur_dir,'test_localization_fail.log'), 'r+') as file:
                file_data = file.read()
                file.seek(0, 0)
                file.write("".join(('\t',results.to_string(index=False).replace('\n', '\n\t'),'\n\n','Test Failures ',
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'\n',file_data)))
                self.fail(f"Test Failed. Details saved in ...testdata/test_localization_fail.log")

if __name__ == '__main__':
    unittest.main()
