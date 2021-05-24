import sys

sys.path.append('../')

import os
import unittest
from Analysis.baseline_model_testing import *
from Analysis.os_feature_models import OSFSenseModel
from test_functions import *

from pandas._testing import assert_frame_equal
import numpy as np

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """

    def test_initial_model(self):

        s_info = StudyInfo("2019 study")
        all_scenes = s_info.scene_name_list
        preposition_models_dict = dict()
        for p in PREPOSITION_LIST:
            M = GeneratePrepositionModelParameters(s_info, p, all_scenes,
                                                   features_to_remove=Configuration.object_specific_features.copy())
            M.work_out_models()
            preposition_models_dict[p] = M

        baseline_model = PrototypeModel(preposition_models_dict, all_scenes, s_info)
        sense_model = OSFSenseModel("Sense Model (OS)", all_scenes, all_scenes, s_info,
                                    PREPOSITION_LIST, baseline_model=baseline_model, preserve_empty_polysemes=True)
        config_list = s_info.config_list
        c = config_list[0]

        for model in [baseline_model,sense_model]:
            value_array = np.array(c.row)
            typicality = model.get_typicality('in', value_array, scene=c.scene, figure=c.figure,
                                                              ground=c.ground, study=s_info)
            print(type(typicality))
            self.assertIsInstance(typicality, float)
        for p in PREPOSITION_LIST:
            for polyseme in sense_model.polyseme_dict[p]:
                for f in Configuration.object_specific_features:
                    i = s_info.all_feature_keys.index(f)
                    self.assertEqual(polyseme.weights[i], 0)

                for key in polyseme.greater_feature_dict:
                    self.assertIn(key, Configuration.object_specific_features)
                for key in polyseme.less_feature_dict:
                    self.assertIn(key, Configuration.object_specific_features)

                if polyseme.number_of_instances == 0:
                    print(f"No instances of polyseme {polyseme.polyseme_name} for {p}")


if __name__ == "__main__":
    unittest.main()
