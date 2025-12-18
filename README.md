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
    Results obtained after running grid searches to find best parameters and training the best models are in the results folder, however running all the 3 codes 

## How to Run


### Possible issues when running the code and how to fix them


## Results


