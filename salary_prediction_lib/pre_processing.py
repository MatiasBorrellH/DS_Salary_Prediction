import country_converter as coco

def convert_country_codes(dataset, employee_column='employee_residence', company_column='company_location'):
    """
    Converts country codes to full country names for specified columns.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the columns to convert.
    - employee_column (str): Column with employee residence country codes.
    - company_column (str): Column with company location country codes.

    Returns:
    - pd.DataFrame: The dataset with updated country names.
    """
    dataset[employee_column] = coco.convert(names=dataset[employee_column], to="name")
    dataset[company_column] = coco.convert(names=dataset[company_column], to="name")
    return dataset


def replace_abbreviations(dataset):
    """
    Replaces abbreviations in columns with their full descriptions.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the columns to process.

    Returns:
    - pd.DataFrame: The dataset with replaced values.
    """
    dataset["experience_level"].replace(
        {"EN": "Entry Level", "MI": "Mid Level", "EX": "Expert Level", "SE": "Senior Level"}, inplace=True
    )
    dataset["employment_type"].replace(
        {"PT": "Part Time", "FT": "Full Time", "CT": "Contractual", "FL": "Freelance"}, inplace=True
    )
    dataset["company_size"].replace(
        {"M": "Medium", "L": "Large", "S": "Small"}, inplace=True
    )
    dataset["remote_ratio"].replace(
        {100: "Fully Remote", 0: "Non Remote Work", 50: "Partially Remote"}, inplace=True
    )
    return dataset


def drop_unnecessary_columns(dataset, columns_to_drop=None):
    """
    Drops specified columns from the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset to process.
    - columns_to_drop (list): List of column names to drop.

    Returns:
    - pd.DataFrame: The dataset with dropped columns.
    """
    if columns_to_drop is None:
        columns_to_drop = ["salary_currency", "salary"]
    dataset.drop(columns=columns_to_drop, axis=1, inplace=True)
    return dataset

import pandas as pd

def remove_outliers_iqr(dataframe, column=None, multiplier=1.5):
    """
    Removes outliers from a dataset using the IQR technique.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to process.
    - column (str or None): The column to filter outliers from. If None, applies to all numerical columns.
    - multiplier (float): The IQR multiplier to define outliers. Default is 1.5.

    Returns:
    - pd.DataFrame: The filtered DataFrame with outliers removed.
    """
    if column:
        # Compute IQR for the specific column
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter rows within the bounds
        return dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
    else:
        # Compute IQR for all numerical columns
        filtered_data = dataframe.copy()
        for col in dataframe.select_dtypes(include='number').columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Filter rows for the current column
            filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]
        
        return filtered_data


    
import pandas as pd

def preprocess_data(dataset):
    """
    Preprocess the dataset by applying all necessary transformations.

    Parameters:
    - dataset (pd.DataFrame): The input dataset.

    Returns:
    - pd.DataFrame: The preprocessed dataset.
    """
    # Convert country codes to full country names
    dataset = convert_country_codes(dataset, employee_column='employee_residence', company_column='company_location')

    # Replace abbreviations
    dataset = replace_abbreviations(dataset)

    # Drop unnecessary columns
    #dataset = drop_unnecessary_columns(dataset)

    # Remove outliers (if applicable)
    #dataset = remove_outliers_iqr(dataset, column=None, multiplier=1.5)  # Adjust column and multiplier as needed

    return dataset