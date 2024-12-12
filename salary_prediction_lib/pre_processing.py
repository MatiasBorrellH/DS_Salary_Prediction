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