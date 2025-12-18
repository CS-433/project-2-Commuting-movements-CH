[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UcP9Py08)

 
# Predicting commuting flows using urban indicators and machine learning methods

## Authors

Stefanie Helfenstein, Leila Paolini, Marius Wrobel


## Description
The goal of this project is to study commuter flows between Swiss municipalities and to evaluate machine learning models for geographically predicting both the existence and magnitude of commuting flows using spatial, demographic, and socio-economic indicators. We compare two traditional mobility models, gravity and radiation, with three machine learning approaches: XGBoost, CatBoost, and a fully connected neural network. The task is decomposed into a binary classification problem for flow existence and a regression problem for non-zero flow magnitudes. Models are trained and evaluated using a spatial train–validation–test split based on Swiss cantons to assess generalization to unseen regions. Results show that machine learning methods substantially outperform traditional models, with CatBoost achieving the best classification performance and the neural network yielding the highest regression accuracy.


## Project structure

- Data preprocessing: 
    - MAIN FILE: Features.ipynb : file where all the raw data from heretogeneous datasets is preprocessed and merged to obtain the final dataframe we will work on 
    - verify_rw_matching.ipynb : file where analyse the initial data we have about commuting flows, in particular we check if residence based entries and workplace based entries correspond
    - Data_splitting.ipynb : notebook used to explore and test different splitting options 
    - Correlation.ipynb : notebook where we construct the correlation matrix and correlations to target
- Models: 
    - Traditional models:
        - gravitation_model.ipynb : notebook where gravitation model if created and implemented 
        - radiation_model.ipynb : notbook where gravitaiton model is implemented
    - XGBoost: 
        - XGBoost.ipynb : notebook where both classification and regression models are trained and tested, the two models are in two separate sections, for each models a grid search is performed
    - CatBoost:
        - CatBoosting.ipynb : notebook where both classification and regression models are trained and tested, the two models are in two separate sections, for each models a grid search is performed
    - FCNN:
        - neural_classifier.py : core model implementation for the classifier
        - neural_classifier.ipynb : notebook to use the FCNN classifier. Can run crossvalidation, best model training, and evaluation
        - neural_regressor.py : core model implementation for the regressor
        - neural_regressor.ipynb : notebook to use the FCNN regressor. Can run crossvalidation, best model training, and evaluation
        - neural_utils.py : utility methods for the FCNN models / training
- Results: 
    Results obtained after running grid searches to find best parameters and training the best models are in the results folder, however running all the three different models should reload these results in the main folder


## How to Run
1. Clone this repository
2. Ensure all dependencies are installed
3. Open the data zip file
4. Get the data path to this file 
5. Open the file you want to run
6. Make sure the base path 
7. Run the file

### Data preparation
In the file Features.ipynb edit the base_path to your data path, if you run the file the data_y.npy is created from the data files in the folder. 
The final data_y.npy is already included in the zip file so that you can use it without running the notebook.

### Neural Networks
Edit the data path in neural_utils.prepare_data to your data path.
Run the code at the beginning of the notebook (neural_classifier.ipynb / neural_regressor.py), then whatever section that should be executed.

### XGBoost
Edit the data path base_path to you data path. 
Run the code at the beginning of the notebook XGBoost. 
If you want to run the best models in the results file they can be loaded using the xgb load function 

### CatBoosting 
Edit the data path base_path to you data path. 
Run the code at the beginning of the notebook CatBoosting. 
If you want to run the best models in the results file they can be loaded using the xgb load function 


### Necessary python libraries:
    -tqdm
    -numpy
    -torch
    -matplotlib
    -sklearn
    -pandas
    -catboost
    -geopandas
    -shapely
    -os
    -json
    -scipy
    -seaborn
    -intertools
    -requests
    -xgboost
    


## Results
- Classification F1 score: 
    XGBoost: 0.68
    Catboost: 0.70
    FCNN: 0.67

- Regression R2 score:
    XGBoost: 0.78
    Catboost: 0.34
    FCNN: 0.84
    Gravitation: 0.165
    Radiation: 0.128
