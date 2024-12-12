from setuptools import setup, find_packages

setup(
    name="salary_prediction_lib",
    version="0.1",
    packages=find_packages(),
    install_requires=[],  # Agrega aquí dependencias si las tienes.
    author="Tu Nombre",
    author_email="tu.email@example.com",
    description="Librería para predecir salarios de Data Scientists",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-repo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)