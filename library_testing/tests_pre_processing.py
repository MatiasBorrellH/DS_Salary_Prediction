import unittest
import pandas as pd
from salary_prediction_lib.pre_processing import (convert_country_codes, replace_abbreviations, 
                          drop_unnecessary_columns, remove_outliers_iqr)

class TestPreprocessingFunctions(unittest.TestCase):

    def setUp(self):
        """
        Set up a sample DataFrame for testing.
        """
        self.sample_data = pd.DataFrame({
            "employee_residence": ["US", "GB", "IN"],
            "company_location": ["US", "GB", "CA"],
            "experience_level": ["EN", "MI", "SE"],
            "employment_type": ["FT", "PT", "CT"],
            "company_size": ["M", "L", "S"],
            "remote_ratio": [100, 50, 0],
            "salary_currency": ["USD", "GBP", "INR"],
            "salary": [120000, 45000, 10000],
            "age": [25, 40, 22],
            "performance_score": [75, 90, 85]
        })

    def test_convert_country_codes(self):
        """
        Test the convert_country_codes function.
        """
        updated_data = convert_country_codes(self.sample_data)
        self.assertIn("United States", updated_data["employee_residence"].values)
        self.assertIn("United Kingdom", updated_data["company_location"].values)

    def test_replace_abbreviations(self):
        """
        Test the replace_abbreviations function.
        """
        updated_data = replace_abbreviations(self.sample_data)
        self.assertIn("Entry Level", updated_data["experience_level"].values)
        self.assertIn("Full Time", updated_data["employment_type"].values)
        self.assertIn("Medium", updated_data["company_size"].values)

    def test_drop_unnecessary_columns(self):
        """
        Test the drop_unnecessary_columns function.
        """
        updated_data = drop_unnecessary_columns(self.sample_data, ["salary_currency", "salary"])
        self.assertNotIn("salary_currency", updated_data.columns)
        self.assertNotIn("salary", updated_data.columns)

    def test_remove_outliers_iqr_specific_column(self):
        """
        Test the remove_outliers_iqr function for a specific column.
        """
        filtered_data = remove_outliers_iqr(self.sample_data, column="performance_score", multiplier=1.5)
        self.assertEqual(len(filtered_data), 3)  # No outliers in this sample dataset

    def test_remove_outliers_iqr_all_columns(self):
        """
        Test the remove_outliers_iqr function for all numerical columns.
        """
        filtered_data = remove_outliers_iqr(self.sample_data)
        self.assertEqual(len(filtered_data), 3)  # No outliers in this sample dataset

if __name__ == "__main__":
    unittest.main()