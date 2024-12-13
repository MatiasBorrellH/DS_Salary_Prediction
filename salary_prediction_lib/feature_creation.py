# Map experience levels to approximate numeric values
def map_experience_level(level):
    """
    Maps experience levels to approximate numeric values.

    Parameters:
    - level (str): Experience level (e.g., "Entry Level", "Mid Level").

    Returns:
    - int: Numeric value representing years of experience.
    """
    if level == "Entry Level":
        return 1  # 0-2 years
    elif level == "Mid Level":
        return 4  # 3-5 years
    elif level == "Expert Level":
        return 8  # 6-10 years
    elif level == "Senior Level":
        return 10  # 10+ years
    else:
        return None  # Handle unexpected values


def add_job_title_frequency(dataset, job_column='job_title'):
    """
    Adds a column with the frequency of each job title.

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data.
    - job_column (str): Name of the column containing job titles.

    Returns:
    - pd.DataFrame: The DataFrame with the new column 'job_title_frequency'.
    """
    dataset['job_title_frequency'] = dataset[job_column].map(dataset[job_column].value_counts())
    return dataset


def add_local_employee_feature(dataset, employee_location_col='employee_residence', company_location_col='company_location'):
    """
    Adds a column indicating if the employee is working locally in the company's country (1 for local, 0 for foreign).” Cambialo por “Adds a column indicating if the employee is employed by a company in his/her home country (compares employee_residence with comany_location, 1 for local, 0 for foreign).

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data.
    - employee_location_col (str): Name of the column with employee residence.
    - company_location_col (str): Name of the column with company location.

    Returns:
    - pd.DataFrame: The DataFrame with the new column 'is_local_employee'.
    """
    # Check if the specified columns exist in the dataset
    if employee_location_col not in dataset.columns or company_location_col not in dataset.columns:
        raise ValueError(f"The columns '{employee_location_col}' and/or '{company_location_col}' do not exist in the dataset.")
    
    # Create the 'is_local_employee' column
    dataset['is_local_employee'] = (dataset[employee_location_col] == dataset[company_location_col]).astype(int)
    
    return dataset


def add_inflation_index(dataset, year_column='work_year', residence_column='employee_residence'):
    """
    Adds an inflation adjustment index based on the work year and country.
    """
    us_inflation_rates = {2019: 0.0181, 2020: 0.0123, 2021: 0.0470, 2022: 0.065, 2023: 0.034}
    global_inflation_rates = {2019: 0.0219, 2020: 0.0192, 2021: 0.0350, 2022: 0.088, 2023: 0.070}

    def calculate_index(year, residence):
        """Calculate the inflation rate for a specific year and residence."""
        if residence == "United States":
            inflation_rate = us_inflation_rates.get(year, 0)
        else:
            inflation_rate = global_inflation_rates.get(year, 0)
        return 1 + inflation_rate

    dataset['inflation_index'] = dataset.apply(
        lambda row: calculate_index(row[year_column], row[residence_column]), axis=1
    )
    return dataset


def add_salary_density(dataset, salary_column='salary_in_usd', experience_column='years_of_experience'):
    """
    Adds a column measuring salary density (z-score) grouped by years of experience.

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data.
    - salary_column (str): Name of the column containing salaries.
    - experience_column (str): Name of the column representing years of experience.

    Returns:
    - pd.DataFrame: The DataFrame with the new column 'salary_density_by_experience'.
    """
    # Calculate mean and standard deviation by years_of_experience
    salary_stats = dataset.groupby(experience_column)[salary_column].agg(['mean', 'std']).reset_index()
    salary_stats.columns = [experience_column, 'mean_salary', 'std_salary']

    # Merge statistics with the original dataset
    dataset = dataset.merge(salary_stats, on=experience_column, how='left')

    # Calculate salary density
    dataset['salary_density_by_experience'] = (
        (dataset[salary_column] - dataset['mean_salary']) / dataset['std_salary']
    ).fillna(0)  # Handle NaN when std_salary is 0

    # Drop auxiliary columns
    dataset.drop(['mean_salary', 'std_salary'], axis=1, inplace=True)

    return dataset


def create_features(dataset):
    """
    Applies all feature creation steps to the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to process.

    Returns:
    - pd.DataFrame: The dataset with all new features added.
    """
    # Map experience levels to numeric values
    dataset['years_of_experience'] = dataset['experience_level'].map(map_experience_level)

    # Add job title frequency
    #dataset = add_job_title_frequency(dataset)

    # Add local employee feature
    dataset = add_local_employee_feature(dataset, employee_location_col='employee_residence', company_location_col='company_location')

    # Add inflation index
    dataset = add_inflation_index(dataset, year_column='work_year', residence_column='employee_residence')

    return dataset


