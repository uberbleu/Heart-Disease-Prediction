# Heart disease prediction

This project involves predicting the likelihood of heart disease in patients using a dataset containing various health indicators such as age, cholesterol levels, and chest pain type. Logistic regression is employed to model the binary target variable, with the goal of identifying key risk factors for heart disease.
* Full notebook: https://colab.research.google.com/drive/1Zen8L8InYlHnalk37OWUJVmQnPzSw4xY?usp=sharing (includes graphs and outputs)

## 1. Problem definition
* In a statement, given clinical parameters about a patient, can we predict whether or not they have heart disease?

## 2. Data
* The original data came from Cleaveland data from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/heart+Disease 
* There is also a version of it available on kaggle. https://www.kaggle.com/ronitf/heart-disease-uci

## 3. Evaluation
* Can we reach 80% accuracy at predicting whether or not a patient has heart disease?

## 4. Features
* age
* sex
* chest pain type (4 values)
* 0: Typical angina: chest pain related decrease blood supply to the heart
* 1: Atypical angina: chest pain not related to heart
* 2: Non-anginal pain: typically esophageal spasms (non heart related)
* 3: Asymptomatic: chest pain not showing signs of disease
* resting blood pressure
* serum cholestoral in mg/dl
* fasting blood sugar > 120 mg/dl
* resting electrocardiographic results (values 0,1,2)
* maximum heart rate achieved
* exercise induced angina
* oldpeak = ST depression induced by exercise relative to rest
* the slope of the peak exercise ST segment
* number of major vessels (0-3) colored by flourosopy
* thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

## Potential insights:

* Heart Disease Distribution: Approximately 54% of the patients in the dataset have heart disease, and 46% do not.
* Age Distribution: The age of patients is widely distributed, with the highest concentration in the 50-60 age group. Heart disease cases tend to increase with age.
* Sex Distribution: Males are more likely to have heart disease compared to females. This is supported by the crosstab analysis showing a higher prevalence of heart disease in males.
* Cholesterol Levels: The cholesterol distribution shows that higher cholesterol levels are slightly associated with heart disease, though the overlap is significant.
* Maximum Heart Rate: Patients with heart disease tend to have a lower maximum heart rate compared to those without heart disease.
* Chest Pain Types: The type of chest pain is an important indicator of heart disease. Those with atypical angina or no pain (asymptomatic) have a higher prevalence of heart disease, while non-anginal pain shows fewer cases.
* Correlation Analysis: The correlation matrix shows that cholesterol, age, and maximum heart rate (thalach) have meaningful correlations with the likelihood of heart disease.

## Key Variables:
### Binary Variables:
* Sex: A change from female (0) to male (1) results in a significant increase in the odds of having heart disease by a notable percentage.
* Chest Pain Type (cp): Changes in chest pain types, particularly moving from typical to asymptomatic, increase the likelihood of heart disease.
* Exercise-induced Angina (exang): Moving from no exercise-induced angina (0) to yes (1) increases the odds of heart disease significantly.
### Continuous Variables:
* Age: A unit increase in age increases the odds of heart disease, with the effect being statistically significant.
* Resting Blood Pressure (trestbps): A slight but not statistically significant increase in the odds of heart disease.
* Cholesterol (chol): Higher cholesterol levels are associated with an increase in heart disease risk.
* Maximum Heart Rate (thalach): A higher heart rate during exercise is associated with a decrease in the odds of heart disease, making it a protective factor.
## Statistical Significance:
* Significant predictors of heart disease include age, sex, chest pain type, cholesterol levels, and maximum heart rate.
* Features such as fasting blood sugar (fbs) and resting electrocardiographic results (restecg) were not statistically significant at the 0.05 level.

## Conclusions
* Chest pain (cp), resting blood pressure (trestbps), exercise-induced angina (exang), ST depression (oldpeak), and thalassemia (thal) are the most important statistically significant predictors of heart disease.
* Sex, particularly being male, significantly reduces the odds of heart disease in this dataset.
* Some variables, like age, cholesterol, and slope, while influential, are not statistically significant in predicting heart disease.
* After finding the best parameters using the logistic regression model, we have not improved.
* We could try XGBoost model in order to see if there is any improvement.
## Retrospection
### 1. Have we met our goal?
* We successfully met our goal of achieving over 80% accuracy in predicting heart disease, exceeding it by 4% with an overall accuracy of 84%. 
* This was accomplished by optimizing the logistic regression model through hyperparameter tuning, ensuring reliable classification of individuals with or without heart disease.
### 2.What did we learn from our experience?
* Through this analysis, we learned that a well-tuned logistic regression model can effectively predict heart disease, demonstrating the importance of hyperparameter optimization and addressing class imbalance. 
* Additionally, key factors such as chest pain type, cholesterol levels, and maximum heart rate were found to significantly influence the likelihood of heart disease, offering valuable insights for future medical assessments.
### What are some future improvements?
* Future improvements could involve exploring more advanced models like Random Forest or XGBoost to enhance prediction accuracy. 
* Additionally, incorporating more diverse datasets and feature engineering may help refine insights and better capture the nuances of heart disease prediction.
