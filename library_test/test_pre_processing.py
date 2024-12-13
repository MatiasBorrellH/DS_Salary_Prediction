import unittest
import pandas as pd
from   salary_prediction_lib.pre_processing import (
    convert_country_codes,
    replace_abbreviations,
    drop_unnecessary_columns,
    remove_outliers_iqr
)

class TestLibraryFunctions(unittest.TestCase):

    def setUp(self):
        # Sample dataset for testing
        self.data = pd.DataFrame({
            'employee_residence': ['US', 'IN', 'FR'],
            'company_location': ['UK', 'JP', 'DE'],
            'experience_level': ['EN', 'MI', 'EX'],
            'employment_type': ['PT', 'FT', 'CT'],
            'company_size': ['M', 'L', 'S'],
            'remote_ratio': [100, 50, 0],
            'salary_currency': ['USD', 'INR', 'EUR'],
            'salary': [50000, 60000, 70000],
            'numeric_column': [10, 1000, 50],
        })

    def test_convert_country_codes(self):
        # Expected conversion using a mocked `coco`
        result = convert_country_codes(self.data.copy())
        self.assertTrue('United States' in result['employee_residence'].values)
        self.assertTrue('United Kingdom' in result['company_location'].values)

    def test_replace_abbreviations(self):
        # Check if abbreviations are replaced correctly
        result = replace_abbreviations(self.data.copy())
        self.assertEqual(result['experience_level'][0], "Entry Level")
        self.assertEqual(result['employment_type'][1], "Full Time")
        self.assertEqual(result['company_size'][2], "Small")
        self.assertEqual(result['remote_ratio'][0], "Fully Remote")

    def test_drop_unnecessary_columns(self):
        # Drop specific columns and validate
        result = drop_unnecessary_columns(self.data.copy())
        self.assertNotIn('salary_currency', result.columns)
        self.assertNotIn('salary', result.columns)

    def test_remove_outliers_iqr_specific_column(self):
        # Test outlier removal for a specific column
        result = remove_outliers_iqr(self.data.copy(), column='numeric_column', multiplier=1.5)
        self.assertEqual(len(result), 3)  # Expect 2 outlier removed

    def test_remove_outliers_iqr_all_columns(self):
        # Test outlier removal for all numerical columns
        result = remove_outliers_iqr(self.data.copy(), multiplier=1.5)
        self.assertEqual(len(result), 3)  # Expect rows with outliers removed

    def test_remove_outliers_iqr_no_outliers(self):
        # Test when there are no outliers
        data_no_outliers = pd.DataFrame({'numeric_column': [10, 20, 30]})
        result = remove_outliers_iqr(data_no_outliers, column='numeric_column', multiplier=1.5)
        self.assertEqual(len(result), 3)  # No rows should be removed

    def test_remove_outliers_iqr_edge_case(self):
        # Test edge case with an empty DataFrame
        empty_df = pd.DataFrame({'numeric_column': []})
        result = remove_outliers_iqr(empty_df, column='numeric_column', multiplier=1.5)
        self.assertTrue(result.empty)

if __name__ == '__main__':
    unittest.main()