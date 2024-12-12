import unittest
import pandas as pd
from salary_prediction_lib.feature_creation import (
    map_experience_level,
    add_job_title_frequency,
    add_local_employee_feature,
    add_inflation_index,
    add_salary_density
)

class TestFeatureFunctions(unittest.TestCase):

    def setUp(self):
        # Sample dataset for testing
        self.data = pd.DataFrame({
            'job_title': ['Data Scientist', 'Analyst', 'Engineer', 'Analyst'],
            'employee_residence': ['US', 'IN', 'FR', 'US'],
            'company_location': ['US', 'UK', 'FR', 'IN'],
            'work_year': [2021, 2022, 2023, 2023],
            'salary_in_usd': [120000, 80000, 95000, 70000],
            'years_of_experience': [4, 4, 2, 1]
        })

    def test_map_experience_level(self):
        # Test mapping of experience levels
        self.assertEqual(map_experience_level("Entry Level"), 1)
        self.assertEqual(map_experience_level("Mid Level"), 4)
        self.assertEqual(map_experience_level("Expert Level"), 8)
        self.assertEqual(map_experience_level("Senior Level"), 10)
        self.assertIsNone(map_experience_level("Unknown Level"))

    def test_add_job_title_frequency(self):
        # Test addition of job title frequency
        result = add_job_title_frequency(self.data.copy())
        self.assertIn('job_title_frequency', result.columns)
        self.assertEqual(result['job_title_frequency'][0], 1)  # Data Scientist
        self.assertEqual(result['job_title_frequency'][1], 2)  # Analyst

    def test_add_local_employee_feature(self):
        # Test local employee feature
        result = add_local_employee_feature(self.data.copy())
        self.assertIn('is_local_employee', result.columns)
        self.assertEqual(result['is_local_employee'][0], 1)  # US employee in US
        self.assertEqual(result['is_local_employee'][1], 0)  # IN employee in UK

    def test_add_inflation_index(self):
        # Test inflation index calculation
        result = add_inflation_index(self.data.copy())

        # Check if the column 'inflation_index' exists
        self.assertIn('inflation_index', result.columns)

        # Validate inflation index for the first row (United States, 2021)
        us_inflation_2021 = 1 + 0.0470  # Inflation rate for 2021 (US)
        self.assertAlmostEqual(result['inflation_index'][0], us_inflation_2021, places=4)

        # Validate inflation index for the second row (India, 2022)
        global_inflation_2022 = 1 + 0.088  # Inflation rate for 2022 (Global)
        self.assertAlmostEqual(result['inflation_index'][1], global_inflation_2022, places=4)

        # Validate inflation index for the third row (France, 2023)
        global_inflation_2023 = 1 + 0.070  # Inflation rate for 2023 (Global)
        self.assertAlmostEqual(result['inflation_index'][2], global_inflation_2023, places=4)

        # Validate inflation index for the fourth row (United States, 2023)
        us_inflation_2023 = 1 + 0.034  # Inflation rate for 2023 (US)
        self.assertAlmostEqual(result['inflation_index'][3], us_inflation_2023, places=4)
if __name__ == '__main__':
    unittest.main()