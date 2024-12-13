# DS_Salary_Prediction
A Scalable Library for a Data Science project, focused on predict salaries in Data related jobs. 

Includes:  
  - Data provided by aijobs.net. \
  - Jupyter notebook "ds_salary_main.ipynb" with EDA, data visualization, analysis and recommendations.
  - Prediction Pipeline. \
  - Readme file with instruction on how to scale the project. \
  - API for quick predictions. \
  - JSON test file.

Instructions:

- For a single value predicction:
  1. Open uvicorn using the terminal: uvicorn app:app --reload
  2. Open in your browser http://127.0.0.1:8000/docs# and go to the predict endpoint.
  3. Fill the data as is shown in the file test_json

- For multiple values prediction:
  1. Open uvicorn using the terminal: uvicorn app:app --reload
  2. Go to the file post_request.py 
  3. Change the route of 'test_json' for the location of your JSON file with the data.

- Add a new predictive model to the library.
  1. Go into salary_prediction_lib/model.py add your model with a set of base parameters as an initial test.
  2. Add the name of your new model function to the __init__ file of the library.
  3. Go to the Model Training Section of ds_salary_main.ipynb and use the function created before to do some predictions, use the train and test data available.
  4. Use the salary_prediction_lib/metrics.py section of the library to address the performance of the model, if you need to use different metrics i.e Classificacion metrix you can add them as a function to this section and use them on the main notebook in the Assess the model section.
  5. If you want to try with some hyperparameter tunning, yo can create a function for your model inside salary_prediction_lib/tunned_model.py

- Add new preprocesors:
  1. Go to the preprocesor section of the library: salary_prediction_lib/pre_processing.py 
  2. Create a function for your new preprocessor 
  3. Use this call this new function inside the pre-processdata function.
  4. Add the name of your new function to the __init__ file of the library.

- Add new features:
  1. Go to the feature_creation section of the library: salary_prediction_lib/feature_creation.py 
  2. Create a function for your new feature.
  3. Use this call this new function inside the create_features function.
  4. Add the name of your new function to the __init__ file of the library.


