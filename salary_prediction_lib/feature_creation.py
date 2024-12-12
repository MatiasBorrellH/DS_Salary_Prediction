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


def add_avg_salary_by_job(dataset, salary_column='salary_in_usd', job_column='job_title'):
    """
    Adds a column with the average salary for each job title.

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data.
    - salary_column (str): Name of the column containing salaries.
    - job_column (str): Name of the column containing job titles.

    Returns:
    - pd.DataFrame: The DataFrame with the new column 'avg_salary_by_job', rounded to the nearest integer.
    """
    # Check if the specified columns exist in the dataset
    if salary_column not in dataset.columns or job_column not in dataset.columns:
        raise ValueError(f"The columns '{salary_column}' and/or '{job_column}' do not exist in the dataset.")
    
    # Calculate and add the average salary column
    dataset['avg_salary_by_job'] = (
        dataset.groupby(job_column)[salary_column]
        .transform('mean')
        .round(0)
        .astype(int)
    )
    return dataset


def add_local_employee_feature(dataset, employee_location_col='employee_residence', company_location_col='company_location'):
    """
    Adds a column indicating if the employee is working locally in the company's country (1 for local, 0 for foreign).

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


def adjust_salary(row):
    """
    Adjusts salary based on annual inflation rates.
    Uses different inflation indices for the United States and global rates.

    Parameters:
    - row (pd.Series): A row from the DataFrame containing 'work_year', 'salary_in_usd', and 'employee_residence'.

    Returns:
    - float: Salary adjusted for inflation.
    """
    us_inflation_rates = {2019: 0.0181, 2020: 0.0123, 2021: 0.0470, 2022: 0.065, 2023: 0.034}
    global_inflation_rates = {2019: 0.0219, 2020: 0.0192, 2021: 0.0350, 2022: 0.088, 2023: 0.070}
    
    year = row['work_year']
    original_salary = row['salary_in_usd']
    residence = row['employee_residence']  # Employee's country of residence

    # If it is the most recent year, no adjustment is needed
    if year == 2024:
        return original_salary

    adjusted_salary = original_salary
    for y in range(year, 2024):
        if residence == "United States":
            inflation_rate = us_inflation_rates.get(y, 0)  # US inflation index
        else:
            inflation_rate = global_inflation_rates.get(y, 0)  # Global inflation index
        
        adjusted_salary *= (1 + inflation_rate)
    
    return adjusted_salary


def add_salary_density(dataset, salary_column='salary_in_usd', job_column='job_title'):
    """
    Adds a column measuring salary density per role (z-score within the same job_title).

    Parameters:
    - dataset (pd.DataFrame): The DataFrame containing the data.
    - salary_column (str): Name of the column containing salaries.
    - job_column (str): Name of the column containing job titles.

    Returns:
    - pd.DataFrame: The DataFrame with the new column 'salary_density'.
    """
    # Calculate mean and standard deviation by 'job_title'
    salary_stats = dataset.groupby(job_column)[salary_column].agg(['mean', 'std']).reset_index()
    salary_stats.columns = [job_column, 'mean_salary', 'std_salary']

    # Merge statistics with the original dataset
    dataset = dataset.merge(salary_stats, on=job_column, how='left')

    # Calculate salary density
    dataset['salary_density'] = (
        (dataset[salary_column] - dataset['mean_salary']) / dataset['std_salary']
    ).fillna(0)  # Handle NaN when std_salary is 0

    # Drop auxiliary columns
    dataset.drop(['mean_salary', 'std_salary'], axis=1, inplace=True)

    return dataset


