import unittest
import sys
import pandas as pd

sys.path.append('../')

from Analysis.compile_instances import SemanticCollection
from Analysis.baseline_model_testing import PREPOSITION_LIST, GeneratePrepositionModelParameters

from Analysis.process_data import *


class Test(unittest.TestCase):
    """Summary
    """

    # @unittest.skip
    def test_typ_p_values(self):
        study_info = StudyInfo("test study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        typ_data = TypicalityData(userdata)
        c1 = SimpleConfiguration("compsvo13v", "book", "board")
        c2 = SimpleConfiguration("compsvi3", "pear", "mug")
        values = typ_data.calculate_pvalue_c1_better_than_c2("on", c1, c2)

        self.assertEqual(values, [5, 1, 4, 0.96875])

    # @unittest.skip
    def test_svmod_p_values(self):
        study_info = StudyInfo("test study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        svmod_data = ModSemanticData(userdata)
        c1 = SimpleConfiguration("compsvula", "pencil", "lamp")
        c2 = SimpleConfiguration("compsvul", "bowl", "lamp")
        values = svmod_data.calculate_pvalue_c1_better_than_c2("on", c1, c2)
        # print(values)
        self.assertEqual(values, [4, 0, 0, 6, 0.004761904761904759])

    def test_preprocess_data(self):
        study_info = StudyInfo("test study")
        f = study_info.feature_processor
        # print(f.means)
        self.assertAlmostEqual(f.means['support'], 0.2717375845)
        self.assertAlmostEqual(f.means['contact_proportion'], 0.06424559126607)

    @staticmethod
    def output_config_ratios():
        study_info = StudyInfo("test study")
        svcollection = SemanticCollection(study_info)

        svcollection.write_config_ratios()

    def test_feature_values(self):
        self.output_config_ratios()
        study_info = StudyInfo("test study")
        p_data = pd.read_csv(study_info.config_ratio_csv("semantic", "above"))

        found = False
        for idx, row in p_data.iterrows():

            if row.Scene == "compsva23" and row.Figure == "balloon" and row.Ground == "book":
                self.assertAlmostEqual(row.support, -0.6282480394372745)
                self.assertAlmostEqual(row.shortest_distance, 0.663680127661742)
                self.assertAlmostEqual(row.contact_proportion, -0.449503120331216)
                self.assertAlmostEqual(row.above_proportion, 1.31175268118854)
                self.assertAlmostEqual(row.below_proportion, -0.74846210511994)
                self.assertAlmostEqual(row.location_control, -0.805302794609531)
                self.assertAlmostEqual(row.f_covers_g, 0.138029738918367)
                found = True
                break

        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
