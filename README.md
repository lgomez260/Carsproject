# Business Understanding

The automotive sector is renowned for its wide range of luxury brands and intense competition. The presence of diverse car models within the same brand further adds to the complexity of standing out and maintaining a strong market position. Price is one of the key factors that can fluctuate and significantly influence customer decisions. With that in mind, multiple analyses have been conducted to better understand this dynamic.

Ford, one of the most well-known and affordable car manufacturers based in the United States, has maintained a strong market presence. In fact, it currently offers the freshest lineup in the industry, contributing to a 6% increase in sales. The introduction of new vehicle types—such as electric, hybrid, and luxury models—has helped the company continue performing well and meet its strategic goals. While variety and innovation are likely major contributors to Ford’s success, pricing has also played a crucial role in setting the brand apart from its competitors.

This report aims to examine the accuracy of car pricing and identify the most significant predictive features. The analysis will be based on a large historical dataset and will explore various data models that can enhance the accuracy of price predictions.

## Business Objective

By this stage, Ford recognized that additional factors might be influencing price fluctuations. Gaining a deeper understanding of how these variables interact with pricing became the focus of a new project. Key features associated with the vehicles include: model, year, transmission type, mileage, fuel type, tax, MPG, and engine size. While the company initially believed that the model was the most significant predictor of price accuracy, this analysis aims to test that hypothesis and uncover new insights.

## Situation assessment

In this phase, it is essential to evaluate what resources are available, what is required, and what potential challenges might arise. To begin, we identified the necessary tools, data sources, and skills needed to complete the analysis. Given the nature of the dataset and the predictive goal, Python was chosen as the primary programming language, as it offers powerful libraries and models suitable for understanding complex relationships between variables. The key libraries used included:

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

With these tools, we assessed the initial performance of the dataset and began evaluating whether the current predictive model was appropriate for pricing. Since the analysis is based on historical data, training a model was necessary to evaluate both its predictive accuracy and explanatory power. As the project advances, additional adjustments and decisions will be made based on the outcomes of this analysis.

The core objective of this project is to identify which features most strongly influence vehicle price, and whether an alternative model could yield more reliable predictions. However, some limitations must be acknowledged. Certain features that could impact price are not present in the historical dataset, which may restrict the analysis. For example, a valuable extension of this project could involve estimating how adjusting prices based on the most predictive features might increase sales — but this remains hypothetical, as sales performance is not included in the dataset.

Additionally, the wide range of models available for testing presents a potential challenge. Exhaustively testing every model may not be practical or time-efficient, and careful selection will be necessary to ensure meaningful and actionable results.

## Determine data mining goals

Now that the goals have been established, it is essential to analyze the most important features identified through the first model, Linear Regression. To begin with, several evaluation graphs were created to understand how each feature performed. The dataset presents a particular characteristic where the variables "year" and "model" are related, representing the launch date of the car. Therefore, it was crucial to examine whether there was any correlation or relevant information that could strengthen the report.

One of the key findings is that most car models were launched in 2017, followed by a noticeable decrease in the number of models in subsequent years. Consequently, identifying the years with the highest number of car models was significant, as it allowed for evaluating both categorical and numerical variables simultaneously. The analysis revealed that the Fiesta model and model year 2018 had the highest occurrence.

At this stage, we recognize the need for further analysis, as several preliminary conclusions have emerged. One potential interpretation is that the company identified the Ford model as a successful car in terms of sales and performance compared to other models.

It is also essential to define what a successful analysis would look like. The most effective way to approach this is by outlining the following key aspects:

- **Accuracy Goal:** Achieving an accuracy of at least 90% is a primary objective, as the features identified in previous graphs appear to provide significant insights into price prediction.
- **Evaluation Metrics:** To assess the model's performance, metrics such as Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and R-Squared will be utilized. These metrics will help determine the model's accuracy and its ability to predict car prices effectively.
- **Alternative Models:** To enhance accuracy, it is important to explore the use of additional models that may outperform the initial linear regression approach. Integrating different techniques could help improve the robustness of the analysis.

## Produce project plan

To effectively analyze the database, various tools and technologies will be utilized throughout the project. The entire analysis will be conducted using the Python programming language, leveraging key libraries to enhance data interpretation and visualization.

For data visualization, Pyplot will be the primary resource, enabling the creation of graphical representations to identify patterns and concentrate data insights. Additionally, various models will be employed to achieve the highest possible accuracy, including linear regression and exponential models. Visualization techniques such as bar charts and box plots will also be used to present data distributions and highlight key findings.

It will also be essential to clearly define each project phase and the resources associated with them to ensure a structured and efficient workflow. Briefly the main stages will be: data understanding, data preparation, modeling and evaluation. Along with the report each of the mentioned phases will be explained and explained.

II. Data understanding:

As the previous information was important to begin understanding the business and why this analysis was conducted. The following analysis will be focused on understanding the data and how from this stage the whole analysis and model will have more sense.

Collecting initial data:

The dataset used for this analysis was sourced from Kaggle, a platform known for hosting a wide variety of datasets for analytical and project-based purposes. For this project, the Ford dataset was selected due to the significant growth and relevance of the automotive industry in recent years. Exploring how various features influence vehicle pricing through predictive models offered valuable insights.

The dataset was originally provided in CSV format and processed using Python, as mentioned previously. It consists of 9 columns and 17,967 rows, each representing different vehicle characteristics. While the dataset includes features identified as important by the company, this analysis aims to evaluate their actual impact on pricing. As a result, some features may be excluded based on their relevance and predictive value.

Describe Data

To describe the dataset used in this project, several key aspects must be considered:

Number of Variables:
The original dataset contains nine variables, of which six are numerical and two are categorical. These variables provide a strong foundation for understanding the different factors influencing car prices. A total of 17,967 records are included in the dataset.

Type of Variables:
The data is primarily divided into numerical and categorical types. Numerical variables include fields such as price, mileage, and year, which are essential for quantitative analysis. Categorical variables, such as model and fuel type, offer valuable context that can reveal trends or preferences related to pricing. For example, analyzing how the model variable interacts with price can help identify whether certain models consistently have higher or lower values.

Record Count and Duplicates:
The dataset includes 17,967 records. During the initial exploration, it was noted that there are repeated models across different years. These repetitions were not removed, as they are relevant for further analysis. In fact, the presence of duplicate models over multiple years may help uncover patterns in pricing or popularity trends and are considered valuable for future exploration.

III. Data preparation

Select data:

For this analysis, all nine available variables were taken into consideration. Since the goal was to identify which features are most closely related to predicting the variable price, it was important to retain the full set of variables. By including all features in the modeling phase, the analysis allows the algorithm to determine which variables contribute most effectively to price prediction and which have less impact.

Clean Data:

The dataset was relatively clean from the start. Due to the nature of the project, it was important to retain most of the data—even entries with repeated model names—as these could provide valuable insights. For instance, the presence of duplicate model names across different years or with varying specifications may reveal patterns relevant to pricing. Moreover, keeping the dataset intact ensures that the machine learning model has access to the full variety of data needed to evaluate predictability accurately.

Construct Data:

Constructing additional features was a fundamental part of this analysis. Given the presence of categorical variables, dummy variables were created to transform these categories into binary format. This step allowed the categorical data to be included in the predictive model, making it possible for the algorithm to interpret and assess their importance in relation to the target variable, price.

Integrate data:

This project was conducted using a single dataset, so data integration was not required. However, this step remains valuable for future extensions of the analysis. For instance, incorporating external data sources—such as customer reviews, regional market trends, or dealership performance—could provide deeper insights. While the current dataset contained sufficient information to identify patterns and make price predictions, access to additional data could support more complex questions, such as identifying best-selling models or forecasting brand performance within the company.

Data formatting:

Data formatting was essential to prepare the dataset for analysis. Several columns originally stored as strings (such as year, mileage, and price) were converted to numeric types to allow for mathematical operations and model training. Column names were standardized using lowercase and underscores for consistency. Categorical variables such as model, and fuel type were encoded where necessary to ensure compatibility with machine learning algorithms.

III. Data preparation

Select data:
For this analysis, all nine available variables were taken into consideration. Since the goal was to identify which features are most closely related to predicting the variable price, it was important to retain the full set of variables. By including all features in the modeling phase, the analysis allows the algorithm to determine which variables contribute most effectively to price prediction and which have less impact.

Clean Data:

The dataset was relatively clean from the start. Due to the nature of the project, it was important to retain most of the data—even entries with repeated model names—as these could provide valuable insights. For instance, the presence of duplicate model names across different years or with varying specifications may reveal patterns relevant to pricing. Moreover, keeping the dataset intact ensures that the machine learning model has access to the full variety of data needed to evaluate predictability accurately.

Construct Data:

Constructing additional features was a fundamental part of this analysis. Given the presence of categorical variables, dummy variables were created to transform these categories into binary format. This step allowed the categorical data to be included in the predictive model, making it possible for the algorithm to interpret and assess their importance in relation to the target variable, price.

Integrate data:

This project was conducted using a single dataset, so data integration was not required. However, this step remains valuable for future extensions of the analysis. For instance, incorporating external data sources—such as customer reviews, regional market trends, or dealership performance—could provide deeper insights. While the current dataset contained sufficient information to identify patterns and make price predictions, access to additional data could support more complex questions, such as identifying best-selling models or forecasting brand performance within the company.

Data formatting:

Data formatting was essential to prepare the dataset for analysis. Several columns originally stored as strings (such as year, mileage, and price) were converted to numeric types to allow for mathematical operations and model training. Column names were standardized using lowercase and underscores for consistency. Categorical variables such as model, and fuel type were encoded where necessary to ensure compatibility with machine learning algorithms.

IV. Modeling
Select modeling techniques: 
In this analysis, three different predictive models were applied to ensure a more accurate evaluation and support data-driven decision-making.
The first model used was Linear Regression, aimed at identifying whether the car features aligned linearly with price. However, the results indicated that this model was not the most suitable, as the data did not follow a clear linear pattern, leading to lower predictive performance.
To address this limitation, the Random Forest Regressor was implemented. This model was selected for its robustness and its ability to reduce overfitting through the use of multiple decision trees. Its strength lies in capturing non-linear relationships and providing greater predictive accuracy, as will be demonstrated in the following sections.
Finally, the Decision Tree Regressor was included as a supervised learning model for numeric prediction. While less accurate than the Random Forest, it offers a clear visual representation of how different variables influence the target variable, making it especially valuable for interpretability and communicating insights.

Generate Test Design:
For this stage of the analysis, the dataset was split into 80% for training and 20% for testing, in order to reduce the likelihood of overfitting. Additionally, key evaluation metrics were used, including MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R² (R-squared Score) to assess model performance.

Build Models:
When building the models, several aspects were considered. Most importantly, libraries such as Scikit-learn were used to train the models and minimize the risk of overfitting. The sklearn.metrics module played a central role in calculating the evaluation metrics. Finally, all relevant libraries were imported to support model creation, training, and evaluation throughout the process.

Assess Model:
At this stage of the process, it is essential to compare and contrast the performance of the different models used. The most effective way to present this comparison is through a table that summarizes how each model performed based on the selected evaluation metrics:

Table # 1 

| Metric           | Random Forest Regressor | Decision Tree Regressor | Linear regression |
|------------------|--------------------------|---------------------------|--------------------|
| MAE              | 866.64                   | 1060.85                   | -                  |
| RMSE             | 1262.35                  | 1590.07                   | 1869.18            |
| R square Score   | 0.93                     | 0.89                      | 0.84               |

The most important takeaways from this table are as follows:
1. The model with an R² score closest to 1.0 is considered the most suitable for accurately predicting car prices using the selected features. Based on this, the Random Forest Regressor stands out as the strongest candidate to be implemented as the final model for this project.
2. On the other hand, while Linear Regression achieved a decent R² score, it also presented a higher error margin (as indicated by the RMSE), which could impact the reliability of the predictions. Therefore, it may not be the most appropriate choice for this dataset.
3. Lastly, the Decision Tree Regressor is a valuable model, particularly for visual analysis. Although it does not perform as well as Random Forest in terms of accuracy, its interpretability makes it a strong complementary tool. This aspect will be further considered in the discussion and recommendations section.

V. Evaluation
Evaluate results:
After comparing and explaining how the metrics should be understood, metrics such as: MAE, RMSE and R Square, it was concluded that the Random Forest Regressor best meets the business success criteria. With an R Square score of 0.93 and the lowest error values, this model provides the most accurate and reliable price predictions. Therefore, it is the recommended model to be approved for further use in supporting pricing strategies or forecasting. 

Review process: 
Throughout this process each stage was very important to prepare and clean the data in order to have better data to work with. The dataset was properly cleaned and formatted, and multiple models were tested to ensure robustness. One area that could be improved in future interactions is the integration of external data sources, such as regional pricing trends or market demand, which could enhance model performance. Overall, the process followed was solid and aligned with CRIPS-DM standards. 

Determine next steps: 
The next step is to implement the model results in an interactive dashboard to better understand how the most important variables—such as model and price—are correlated. This will also provide an opportunity to incorporate additional features and explore how these variables can be combined to generate new insights that go beyond the original analysis. Such an approach will support the development of new hypotheses and enable more focused, exploratory analyses that could inform future business strategies.

VI. Deployment 
For this final stage, it is important to define how the project will be monitored and maintained once it has been delivered.
The model will be updated periodically as new data becomes available in the database. If outliers or anomalies are detected, the data will be reviewed and the model will be retrained to ensure that it continues to operate with clean and reliable inputs.
As part of this monitoring process, key metrics such as MAE and RMSE will be tracked over time. If these metrics show any significant changes or performance degradation, a comprehensive model evaluation and retraining process will be initiated to maintain accuracy and reliability.

As part of this project review, it is important to address the following questions: What went well? What could have been better? What are the future improvements?
Regarding what went well, the project benefited from having high-quality data, which made the data cleaning process much smoother and allowed the project to progress efficiently.
As for what could have been better, more time could have been allocated to quickly identifying the most appropriate model. Early on, additional focus on model selection might have streamlined the process and led to faster decision-making regarding the best-performing model—in this case, the Random Forest Regressor rather than Linear Regression.
For future improvements, efforts should focus on accelerating the testing and model evaluation phase. This would save time and allow greater attention to be given to other stages, such as deployment planning or the integration of additional features.

