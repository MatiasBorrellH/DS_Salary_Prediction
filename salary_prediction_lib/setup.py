from setuptools import setup, find_packages

setup(
    name='ds_salary_prediction_lib', 
    version='0.1.0',
    description='Library for ds_prediction pipeline',
    author='Matias Borrell, Nastia Cher, Soledad Monge',
    author_email='matias.borrell@bse.eu, nastia.cher@bse.eu, soledad.mong@bse.eu',
    url='https://github.com/MatiasBorrellH/DS_Salary_Prediction', 
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
    ], 
)