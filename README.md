# Dynamic, Individualized, Artificial Intelligence-Powered Prediction of Recurrent Autoimmune Hepatitis after Liver Transplantation: A Multi-Centre Cohort Study 

## Project Overview

This project implements an end-to-end machine learning workflow that processes data, trains predictive models, and provides detailed model interpretability through SHAP analysis. Our AI-powered clinical decision model provides personalized prediction of recurrent Autoimmune Hepatitis(rAIH) post liver transplant. It provides insight into modifiable and non-modifiable predictors of rAIH, supporting more personalized immunosuppressive strategies to improve long-term outcomes. 

## Repository Structure

```
.
├── requirements.txt    # All packages required for the Python environment
├── data_processing.py    # Data loading, cleaning, and preprocessing
├── models.py             # Machine learning model definitions and training
├── shap_analysis.py      # SHAP-based model interpretability analysis
└── README.md            # Project documentation
```

## File Descriptions

### `data_processing.py`
Handles all data-related operations including:
- Data loading from various sources
- Data cleaning and validation
- Feature engineering
- Train-test splitting
- Data transformation and scaling

### `models.py`
Contains machine learning model implementations:
- Model architecture definitions
- Training procedures
- Hyperparameter configuration
- Model evaluation metrics
- Model saving and loading utilities

### `shap_analysis.py`
Provides model interpretability tools:
- SHAP value computation
- Feature importance visualization
- Individual prediction explanations
- Summary plots and dependence plots
- Model behavior analysis
```
## SHAP Analysis Outputs

The SHAP analysis module generates several types of visualizations:

- **Summary Plots**: Overview of feature importance across all predictions
- **Dependence Plots**: Relationship between feature values and SHAP values
- **Force Plots**: Individual prediction explanations

## Example Results

After running the complete pipeline, you can expect:

- Trained model with performance metrics
- SHAP values for all predictions
- Visualization plots saved to the output directory
- Feature importance rankings

## Configuration

Adjust parameters in each module:

- **Data Processing**: Modify feature engineering steps, scaling methods
- **Models**: Change model types, hyperparameters, cross-validation settings
- **SHAP Analysis**: Adjust plot styles, sample sizes, visualization options
