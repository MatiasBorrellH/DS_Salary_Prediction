from setuptools import setup, find_packages

setup(
    name="salary_prediction_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=['country_converter', 'lightgbm', 'sklearn', 'numpy', 'random', 'pandas'],  
    author="Tu Nombre",
    author_email="anastasiia.chernavskaia@bse.eu, matias.borrell@bse.eu, soledad.monge@bse.eu",
    description="Library for data related roles salary prediction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MatiasBorrellH/DS_Salary_Prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
