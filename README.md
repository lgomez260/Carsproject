# Carsproject
Machine learning project using Ford car data to predict vehicle prices. Explored Linear Regression, Decision Tree, and Random Forest—achieving 93% accuracy. Includes data cleaning, feature engineering, and model evaluation using Python and Scikit-learn.
# Business Understanding

The automotive sector is renowned for its wide range of luxury brands and intense competition. The presence of diverse car models within the same brand further adds to the complexity of standing out and maintaining a strong market position. Price is one of the key factors that can fluctuate and significantly influence customer decisions. With that in mind, multiple analyses have been conducted to better understand this dynamic.

Ford, one of the most well-known and affordable car manufacturers based in the United States, has maintained a strong market presence. In fact, it currently offers the freshest lineup in the industry, contributing to a 6% increase in sales. The introduction of new vehicle types—such as electric, hybrid, and luxury models—has helped the company continue performing well and meet its strategic goals. While variety and innovation are likely major contributors to Ford’s success, pricing has also played a crucial role in setting the brand apart from its competitors.

This report aims to examine the accuracy of car pricing and identify the most significant predictive features. The analysis will be based on a large historical dataset and will explore various data models that can enhance the accuracy of price predictions.

## Business Objective

By this stage, Ford recognized that additional factors might be influencing price fluctuations. Gaining a deeper understanding of how these variables interact with pricing became the focus of a new project. Key features associated with the vehicles include: model, year, transmission type, mileage, fuel type, tax, MPG, and engine size. While the company initially believed that the model was the most significant predictor of price accuracy, this analysis aims to test that hypothesis and uncover new insights.

## Situation Assessment

In this phase, it is essential to evaluate what resources are available, what is required, and what potential challenges might arise. Python was chosen as the primary programming language for its powerful libraries and models. The key libraries used included:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

## Determine Data Mining Goals

Several evaluation graphs were created to understand how each feature performed. The dataset presents a characteristic where variables like "year" and "model" are related, representing the car's launch date. Most car models were launched in 2017, with a decline in subsequent years. The analysis revealed that the Fiesta model and model year 2018 had the highest occurrence.

The analysis aims to:
- Achieve at least 90% accuracy.
- Use RMSE, MAE, and R² to evaluate performance.
- Explore alternative models beyond Linear Regression to improve accuracy.

## Produce Project Plan

The analysis was conducted in Python, using libraries for visualization and modeling. Main stages included data understanding, data preparation, modeling, and evaluation.

# Data Understanding

## Collect Initial Data

The dataset was sourced from Kaggle. It contained 9 columns and 17,967 rows and was processed in Python.

## Describe Data

- 9 variables: 6 numerical, 2 categorical.
- 17,967 records, with repeated models across years (kept for analysis).

# Data Preparation

## Select Data

All variables were retained to let the model identify the most predictive features.

## Clean Data

The dataset was clean, and repeated models were kept for insight generation.

## Construct Data

Dummy variables were created for categorical data.

## Integrate Data

A single dataset was used; future work may integrate external sources.

## Data Formatting

Columns were converted to numeric types where necessary, and categorical variables were encoded.

# Modeling

## Select Modeling Techniques

- **Linear Regression**: Baseline, but not ideal due to data non-linearity.
- **Random Forest Regressor**: Selected for accuracy and robustness.
- **Decision Tree Regressor**: Added for interpretability.

## Generate Test Design

80% train / 20% test split; metrics: MAE, RMSE, R².

## Build Models

Models were trained using Scikit-learn, with metrics computed using sklearn.metrics.

## Assess Model

| Metric | Random Forest Regressor | Decision Tree Regressor | Linear Regression |
|---------|------------------------|------------------------|------------------|
| MAE | 866.64 | 1060.85 | - |
| RMSE | 1262.35 | 1590.07 | 1869.18 |
| R² | 0.93 | 0.89 | 0.84 |

Random Forest provided the best predictive performance.

# Evaluation

- **Evaluate Results**: Random Forest met success criteria (R² = 0.93, lowest errors).
- **Review Process**: All stages were well executed; future work could include external data.
- **Next Steps**: Develop a dashboard, explore new features, and generate hypotheses.

# Deployment

The model will be monitored using MAE and RMSE. Retraining will occur if performance degrades. Future improvements include faster testing and more focus on model selection in early stages.
