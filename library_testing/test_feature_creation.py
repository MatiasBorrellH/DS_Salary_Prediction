import unittest
import pandas as pd
from salary_prediction_lib.feature_creation import (
    map_experience_level,
    add_job_title_frequency,
    add_local_employee_feature,
    add_inflation_index,
    add_salary_density
)

class TestFeatureCreation(unittest.TestCase):

    def setUp(self):
        """
        Set up sample datasets for testing.
        """
        self.sample_data = pd.DataFrame({
            "job_title": ["Data Scientist", "Data Engineer", "Data Scientist"],
            "employee_residence": ["US", "GB", "US"],
            "company_location": ["US", "GB", "CA"],
            "experience_level": ["Entry Level", "Mid Level", "Expert Level"],
            "work_year": [2021, 2022, 2020],
            "salary_in_usd": [100000, 120000, 150000],
            "years_of_experience": [1, 4, 8]
        })

    def test_map_experience_level(self):
        """
        Test the map_experience_level function.
        """
        self.assertEqual(map_experience_level("Entry Level"), 1)
        self.assertEqual(map_experience_level("Mid Level"), 4)
        self.assertEqual(map_experience_level("Expert Level"), 8)
        self.assertEqual(map_experience_level("Senior Level"), 10)
        self.assertIsNone(map_experience_level("Unknown Level"))

    def test_add_job_title_frequency(self):
        """
        Test the add_job_title_frequency function.
        """
        result = add_job_title_frequency(self.sample_data.copy())
        self.assertIn("job_title_frequency", result.columns)
        self.assertEqual(result.loc[0, "job_title_frequency"], 2)  # "Data Scientist" appears twice
        self.assertEqual(result.loc[1, "job_title_frequency"], 1)  # "Data Engineer" appears once

    def test_add_local_employee_feature(self):
        """
        Test the add_local_employee_feature function.
        """
        result = add_local_employee_feature(self.sample_data.copy())
        self.assertIn("is_local_employee", result.columns)
        self.assertEqual(result.loc[0, "is_local_employee"], 1)  # US == US
        self.assertEqual(result.loc[1, "is_local_employee"], 1)  # GB == GB
        self.assertEqual(result.loc[2, "is_local_employee"], 0)  # US != CA

    def test_add_inflation_index(self):
        """
        Test the add_inflation_index function.
        """
        result = add_inflation_index(self.sample_data.copy())
        self.assertIn("inflation_index", result.columns)
        self.assertGreater(result.loc[0, "inflation_index"], 1.0)  # Inflation-adjusted value
        self.assertGreater(result.loc[1, "inflation_index"], 1.0)
        self.assertGreater(result.loc[2, "inflation_index"], 1.0)

    def test_add_salary_density(self):
        """
        Test the add_salary_density function.
        """
        result = add_salary_density(self.sample_data.copy())
        self.assertIn("salary_density_by_experience", result.columns)
        # Check that density values are calculated
        self.assertAlmostEqual(result.loc[0, "salary_density_by_experience"], 0)
        self.assertAlmostEqual(result.loc[1, "salary_density_by_experience"], 0)
        self.assertAlmostEqual(result.loc[2, "salary_density_by_experience"], 0)

if __name__ == "__main__":
    unittest.main()