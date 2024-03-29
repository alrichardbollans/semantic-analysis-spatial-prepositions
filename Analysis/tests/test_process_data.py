import os
import unittest
import sys

sys.path.append('../')

from test_functions import *
from Analysis.process_data import *
import scipy.stats as stats
from pandas._testing import assert_frame_equal

os.chdir('..')

class Test(unittest.TestCase):
    """Summary
    """

    def test_semantic_data_attributes(self):
        study_info = StudyInfo("2019 study")
        # Generate csvs.
        userdata2019 = UserData(study_info)
        semantic_data = SemanticData(userdata2019)

        self.assertNotEqual(len(semantic_data.clean_data_list), 0)

        for a in semantic_data.clean_data_list:
            self.assertIsInstance(a, SemanticAnnotation)
            self.assertIsInstance(a.prepositions, list)

    def test_comp_data_attributes(self):
        study_info = StudyInfo("2019 study")
        # Generate csvs.
        userdata2019 = UserData(study_info)
        comparative_data = ComparativeData(userdata2019)

        self.assertNotEqual(len(comparative_data.clean_data_list), 0)

        for a in comparative_data.clean_data_list:
            self.assertIsInstance(a, ComparativeAnnotation)
            self.assertIsInstance(a.preposition, str)
            self.assertIsInstance(a.possible_figures, list)

    def test_semmod_data_attributes(self):
        study_info = StudyInfo("2020 study")
        # Begin by loading users
        userdata = UserData(study_info)

        svmod_data = ModSemanticData(userdata)

        self.assertNotEqual(len(svmod_data.clean_data_list), 0)

        for a in svmod_data.clean_data_list:
            self.assertIsInstance(a, SemanticAnnotation)
            self.assertIsInstance(a.prepositions, list)
            self.assertEqual(a.task, 'sv_mod')

    def test_typ_data_attributes(self):
        study_info = StudyInfo("2020 study")
        # Begin by loading users
        userdata = UserData(study_info)
        typ_data = TypicalityData(userdata)

        self.assertNotEqual(len(typ_data.clean_data_list), 0)

        for a in typ_data.clean_data_list:
            self.assertIsInstance(a, TypicalityAnnotation)
            self.assertIsInstance(a.c1_config, SimpleConfiguration)
            self.assertIsInstance(a.c2_config, SimpleConfiguration)
            self.assertIsInstance(a.selection_config, SimpleConfiguration)
            self.assertEqual(a.task, 'typ')

    # @unittest.skip
    def test_clean_outputs_process_data(self):
        ## 2019
        study_info = StudyInfo("2019 study")
        # Generate csvs.
        userdata2019 = UserData(study_info)
        user_csv = output_folder + userdata2019.study_info.clean_user_csv
        userdata2019.output_clean_user_list(path=user_csv)
        semantic_data = SemanticData(userdata2019)
        semantic_data_csv = output_folder + semantic_data.study_info.data_folder + "/" + semantic_data.clean_csv_name
        semantic_data.output_clean_annotation_list(path=semantic_data_csv)
        comparative_data = ComparativeData(userdata2019)
        comp_data_csv = output_folder + comparative_data.study_info.data_folder + "/" + comparative_data.clean_csv_name
        comparative_data.output_clean_annotation_list(path=comp_data_csv)

        new_clean_sem_dataset, original_clean_sem_dataset = generate_dataframes_to_compare(semantic_data_csv)
        new_clean_user_dataset, original_clean_user_dataset = generate_dataframes_to_compare(
            user_csv)
        # Doesn't like reading the full csv as different column lengths, so need to specifywhich columns to use.
        new_clean_comp_dataset, original_clean_comp_dataset = generate_dataframes_to_compare(
            comp_data_csv, columns_to_use=[0, 1, 2, 3, 4, 5, 6, 7])

        assert_frame_equal(new_clean_sem_dataset, original_clean_sem_dataset)
        assert_frame_equal(new_clean_comp_dataset, original_clean_comp_dataset)
        assert_frame_equal(new_clean_user_dataset, original_clean_user_dataset)

        ##2020

        study_info = StudyInfo("2020 study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        typ_data = TypicalityData(userdata)

        # # output typicality csv
        typ_data_csv = output_folder + typ_data.study_info.data_folder + "/" + typ_data.clean_csv_name

        typ_data.output_clean_annotation_list(path=typ_data_csv)
        typ_stats_csv = output_folder + typ_data.study_info.stats_folder + "/" + typ_data.stats_csv_name
        typ_data.output_statistics(path=typ_stats_csv)
        typ_agreements_csv = output_folder + typ_data.study_info.stats_folder + "/" + typ_data.agreements_csv_name

        typ_data.write_user_agreements(path=typ_agreements_csv)

        new_df, original_df = generate_dataframes_to_compare(typ_data_csv)
        assert_frame_equal(new_df, original_df)

        new_df, original_df = generate_dataframes_to_compare(typ_stats_csv)
        assert_frame_equal(new_df, original_df)

        new_df, original_df = generate_dataframes_to_compare(typ_agreements_csv)
        assert_frame_equal(new_df, original_df)
        # # Load and process semantic annotations
        svmod_data = ModSemanticData(userdata)

        ## Outputs
        sv_mod_data_csv = output_folder + svmod_data.study_info.data_folder + "/" + svmod_data.clean_csv_name

        svmod_data.output_clean_annotation_list(path=sv_mod_data_csv)
        sv_mod_stats_csv = output_folder + svmod_data.study_info.stats_folder + "/" + svmod_data.stats_csv_name

        svmod_data.output_statistics(path=sv_mod_stats_csv)
        #
        svmod_agreements_csv = output_folder + svmod_data.study_info.stats_folder + "/" + svmod_data.agreements_csv_name

        svmod_data.write_user_agreements(path=svmod_agreements_csv)
        #

        new_df, original_df = generate_dataframes_to_compare(sv_mod_data_csv)
        assert_frame_equal(new_df, original_df)

        new_df, original_df = generate_dataframes_to_compare(svmod_agreements_csv)
        assert_frame_equal(new_df, original_df)

        new_df, original_df = generate_dataframes_to_compare(sv_mod_stats_csv)
        assert_frame_equal(new_df, original_df)

    # @unittest.skip
    def test_sv_agreements(self):
        study_info = StudyInfo("2019 study")
        # Generate csvs.
        userdata2019 = UserData(study_info)
        semantic_data = SemanticData(userdata2019)
        self.assertEqual(len(semantic_data.native_users), 32)
        sv_agreements_csv = output_folder + semantic_data.study_info.stats_folder + "/" + semantic_data.agreements_csv_name

        semantic_data.write_user_agreements(path=sv_agreements_csv)

        original_csv = get_original_csv(sv_agreements_csv)

        original_dataframe = pd.read_csv(original_csv, header=None, sep='\n')
        new_dataframe = pd.read_csv(sv_agreements_csv, header=None, sep='\n')

        cohen_value1 = float(original_dataframe[0][12].split(',')[3])
        cohen_value2 = float(new_dataframe[0][12].split(',')[3])

        print(cohen_value1)
        print(cohen_value2)

        self.assertAlmostEqual(cohen_value2, cohen_value1, places=6)

    # @unittest.skip
    def test_comp_agreements(self):
        study_info = StudyInfo("2019 study")
        # Generate csvs.
        userdata2019 = UserData(study_info)

        comparative_data = ComparativeData(userdata2019)
        self.assertEqual(len(comparative_data.native_users), 29)
        comp_agreements_csv = output_folder + comparative_data.study_info.stats_folder + "/" + comparative_data.agreements_csv_name

        comparative_data.write_user_agreements(path=comp_agreements_csv)

        new_df, original_df = generate_dataframes_to_compare(comp_agreements_csv)
        print(original_df)
        print(new_df)
        # This may fail from precision.
        try:
            assert_frame_equal(new_df, original_df, check_exact=False, check_less_precise=2)
        except:
            compare_df = new_df.eq(original_df)
            print(compare_df)

    # @unittest.skip
    def test_typ_p_values(self):
        study_info = StudyInfo("tests/test study")
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
        study_info = StudyInfo("tests/test study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        svmod_data = ModSemanticData(userdata)
        c1 = SimpleConfiguration("compsvula", "pencil", "lamp")
        c2 = SimpleConfiguration("compsvul", "bowl", "lamp")
        values = svmod_data.calculate_pvalue_c1_better_than_c2("on", c1, c2)
        print(values)
        self.assertEqual(values, [4, 0, 0, 6, 0.004761904761904759])

    # @unittest.skip
    def test_fisher(self):
        oddsratio, p_value_one_tail_less = stats.fisher_exact([
            [1, 11],
            [9, 3]], alternative='less'
        )

        self.assertAlmostEqual(p_value_one_tail_less, 0.001379728)


if __name__ == "__main__":
    unittest.main()
