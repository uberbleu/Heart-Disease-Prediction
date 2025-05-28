# Libraries and data

# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
 
 # Load the data
df = pd.read_csv('heart-disease.csv')
df.head(50)

len(df)
 
df.describe().T
 
# Exploratory data analysis
 
# Let's find out how many of are classified with heart disease or not
df['target'].value_counts()

# Are there any missing values?
df.isna().sum()
 
# Age distribution
plt.hist(df['age'], bins=10, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()
 
# Heart disease distribution
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
 
# Let's look into heart disease frequency according to sex
df.sex.value_counts()
 
# Compare target column with sex column
pd.crosstab(df.target, df.sex)
 
# Create a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind='bar',
                                   figsize=(10, 6),
                                   color=['salmon', 'lightblue']);

plt.title("Heart Disease Frequency for Sex")
plt.xlabel('0 = No Disease, 1 = Disease')
plt.ylabel('Amount')
plt.legend(['Female', 'Male']);
plt.xticks(rotation=0);
 
# Heart disease prevalence over age
pd.crosstab(df.age, df.target).plot(kind='bar',
                                   figsize=(10, 5
                                           ),
                                   color=['lightblue', 'salmon'])

plt.title("Heart Disease Prevalence over Age")
plt.xlabel("Age")
plt.ylabel('Number of people')
plt.legend(['Negative', 'Positive']);
plt.xticks(rotation=0);
 
# Correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt='.2f',
                cmap='YlGnBu');
 
# Potential insights are:
# Cholesterol levels: High cholesterol is expected to be positively correlated with heart disease.
# Maximum heart rate (thalach): Higher heart rates achieved during physical activity might be negatively correlated with heart disease.
# Chest pain types: Patients experiencing certain types of chest pain (e.g., angina) may be more likely to have heart disease.
 
# Cholesterol distribution for patients with and without heart disease
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='chol', hue='target', kde=True)
plt.title('Cholesterol Levels Distribution by Heart Disease')
plt.xlabel('Cholesterol Level (mg/dL)')
plt.ylabel('Frequency')
plt.show()
 
# Maximum heart rate distribution for patients with and without heart disease
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='thalach', hue='target', kde=True)
plt.title('Maximum Heart Rate Distribution by Heart Disease')
plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Frequency')
plt.show()
 
# Bar plot for chest pain types vs heart disease
plt.figure(figsize=(10, 6))
sns.countplot(x='cp', hue='target', data=df)
plt.title('Chest Pain Type vs Heart Disease')
plt.xlabel('Chest Pain Type (cp)')
plt.ylabel('Count')
plt.legend(title='Heart Disease', loc='upper right', labels=['No Heart Disease', 'Heart Disease'])
plt.show()
 
# Let's take a deeper look into age vs max heart rate for heart disease
# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c='salmon');

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c='lightblue');

# Add description
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);
 
# Check the distribution of the age column with a histogram
df.age.plot.hist();
 
# Logistic Regression

# Logistic regression is a simple, interpretable model used for binary classification problems.
# It works well when the relationship between the independent variables (like chol, thalach, etc.) and the target variable (target) is linear.
 
df.head()
 
# Isolate X and y
y = df.target
X = df.drop(columns=['target'])
X = sm.add_constant(X)
 
# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1502)
X_train.head()
 
# Logistic Regression
model = sm.Logit(y_train, X_train).fit()
print(model.summary())

Current function value: 0.344302
Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                 target   No. Observations:                  242
Model:                          Logit   Df Residuals:                      228
Method:                           MLE   Df Model:                           13
Date:                Wed, 11 Sep 2024   Pseudo R-squ.:                  0.5013
Time:                        20:10:08   Log-Likelihood:                -83.321
converged:                       True   LL-Null:                       -167.07
Covariance Type:            nonrobust   LLR p-value:                 5.942e-29
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          4.2814      2.962      1.446      0.148      -1.524      10.087
age           -0.0089      0.026     -0.336      0.737      -0.061       0.043
sex           -1.5477      0.546     -2.833      0.005      -2.619      -0.477
cp             0.7983      0.215      3.707      0.000       0.376       1.220
trestbps      -0.0243      0.012     -2.017      0.044      -0.048      -0.001
chol          -0.0022      0.004     -0.486      0.627      -0.011       0.007
fbs            0.3692      0.611      0.605      0.545      -0.827       1.566
restecg        0.5868      0.397      1.479      0.139      -0.191       1.364
thalach        0.0200      0.012      1.719      0.086      -0.003       0.043
exang         -1.0266      0.473     -2.171      0.030      -1.954      -0.100
oldpeak       -0.6749      0.238     -2.833      0.005      -1.142      -0.208
slope          0.7704      0.402      1.915      0.056      -0.018       1.559
ca            -0.7568      0.206     -3.677      0.000      -1.160      -0.353
thal          -1.0750      0.343     -3.133      0.002      -1.747      -0.403
==============================================================================
 
def interpret_logistic_regression(model):
    """
    Interprets the coefficients of a logistic regression model fitted using statsmodels.
    The interpretation is provided in terms of the percentage increase in odds for the event to happen.
    The function automatically detects binary variables and provides information about statistical significance.
    """

    variables = model.model.exog_names[1:]  # Exclude the constant
    params = model.params[1:]  # Exclude the constant
    pvalues = model.pvalues[1:]  # Exclude the constant

    # Identify binary variables by checking if the min and max of the exogenous variable are 0 and 1
    binary_vars = [var for var, values in zip(variables, model.model.exog[:, 1:].T) if min(values) == 0 and max(values) == 1]

    for variable, coef, pvalue in zip(variables, params, pvalues):
        print('-' * 50)

        # Calculate percentage increase in odds
        percentage_increase = (np.exp(coef) - 1) * 100

        if variable in binary_vars:
            print(f"For the binary variable '{variable}':")
            print(f"  - If this variable changes from 0 to 1, the odds of the event happening increase by {percentage_increase:.2f}%.")
        else:
            print(f"For the continuous variable '{variable}':")
            print(f"  - A unit increase in this variable increases the odds of the event happening by {percentage_increase:.2f}%.")

        # Check for statistical significance
        if pvalue < 0.05:
            print("  - This variable is statistically significant at the 0.05 level.")
        else:
            print("  - This variable is not statistically significant at the 0.05 level.")

    print('-' * 50)

# Apply the function:
interpret_logistic_regression(model)
 --------------------------------------------------
For the continuous variable 'age':
  - A unit increase in this variable increases the odds of the event happening by -0.88%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the binary variable 'sex':
  - If this variable changes from 0 to 1, the odds of the event happening increase by -78.73%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'cp':
  - A unit increase in this variable increases the odds of the event happening by 122.18%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'trestbps':
  - A unit increase in this variable increases the odds of the event happening by -2.40%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'chol':
  - A unit increase in this variable increases the odds of the event happening by -0.22%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the binary variable 'fbs':
  - If this variable changes from 0 to 1, the odds of the event happening increase by 44.66%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'restecg':
  - A unit increase in this variable increases the odds of the event happening by 79.81%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'thalach':
  - A unit increase in this variable increases the odds of the event happening by 2.02%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the binary variable 'exang':
  - If this variable changes from 0 to 1, the odds of the event happening increase by -64.18%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'oldpeak':
  - A unit increase in this variable increases the odds of the event happening by -49.08%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'slope':
  - A unit increase in this variable increases the odds of the event happening by 116.07%.
  - This variable is not statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'ca':
  - A unit increase in this variable increases the odds of the event happening by -53.08%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
For the continuous variable 'thal':
  - A unit increase in this variable increases the odds of the event happening by -65.87%.
  - This variable is statistically significant at the 0.05 level.
--------------------------------------------------
Predictions

# Predictions
predictions = model.predict(X_test)
predictions = np.where(predictions > 0.5, 1, 0)
predictions
 array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
       1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
 
# Check if y is imbalanced -> threshold of 35%
y_test.mean()
 0.5737704918032787
 
# Confusion Matrix
print(confusion_matrix(y_test, predictions))
 [[19  7]
 [ 3 32]]
 
# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
               precision    recall  f1-score   support

           0       0.86      0.73      0.79        26
           1       0.82      0.91      0.86        35

    accuracy                           0.84        61
   macro avg       0.84      0.82      0.83        61
weighted avg       0.84      0.84      0.83        61

def evaluate_classification_model(y_true, y_pred):
    """
    Evaluates a classification model by computing the accuracy, F1-score, sensitivity, and specificity.
    """

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:")
    print("  - Measures the proportion of correctly classified instances out of the total instances.")
    print(f"  - Result: {accuracy:.2f}\n")

    # F1-Score
    f1 = f1_score(y_true, y_pred)
    print("F1-Score:")
    print("  - Harmonic mean of precision and recall, balances both false positives and false negatives.")
    print(f"  - Result: {f1:.2f}\n")

    # Sensitivity (Recall)
    sensitivity = tp / (tp + fn)
    print("Sensitivity (Recall):")
    print("  - Measures the proportion of actual positives that are correctly identified as such.")
    print(f"  - Result: {sensitivity:.2f}\n")

    # Specificity
    specificity = tn / (tn + fp)
    print("Specificity:")
    print("  - Measures the proportion of actual negatives that are correctly identified as such.")
    print(f"  - Result: {specificity:.2f}\n")
 
# Apply the function
evaluate_classification_model(y_test, predictions)
 Accuracy:
  - Measures the proportion of correctly classified instances out of the total instances.
  - Result: 0.84

F1-Score:
  - Harmonic mean of precision and recall, balances both false positives and false negatives.
  - Result: 0.86

Sensitivity (Recall):
  - Measures the proportion of actual positives that are correctly identified as such.
  - Result: 0.91

Specificity:
  - Measures the proportion of actual negatives that are correctly identified as such.
  - Result: 0.73

Hyperparameter tuning

# Define the modelmodel = LogisticRegression(max_iter=1000)# Set up hyperparameter gridparam_grid = {    'C': [0.001, 0.01, 0.1, 1, 10, 100],   # Regularization strength    'penalty': ['l1', 'l2'],               # Type of regularization    'solver': ['liblinear', 'saga'],        # Solver options    'class_weight': [None, 'balanced']      # Handle class imbalance}# Setup GridSearchCVgrid_search = GridSearchCV(estimator=model,                           param_grid=param_grid,                           cv=5,               # 5-fold cross-validation                           scoring='accuracy', # You can use other metrics too                           verbose=1,                           n_jobs=-1)# Fit grid searchgrid_search.fit(X_train, y_train)# Get the best parametersprint(f"Best Parameters: {grid_search.best_params_}")# Predict and evaluate with the best modelbest_model = grid_search.best_estimator_y_pred = best_model.predict(X_test)# Apply the functionevaluate_classification_model(y_test, y_pred)

 Fitting 5 folds for each of 48 candidates, totalling 240 fits
Best Parameters: {'C': 1, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'liblinear'}
Accuracy:
  - Measures the proportion of correctly classified instances out of the total instances.
  - Result: 0.84

F1-Score:
  - Harmonic mean of precision and recall, balances both false positives and false negatives.
  - Result: 0.86

Sensitivity (Recall):
  - Measures the proportion of actual positives that are correctly identified as such.
  - Result: 0.91

Specificity:
  - Measures the proportion of actual negatives that are correctly identified as such.
  - Result: 0.73

# After finding the best parameters using the logistic regression model, we have not improved.
# We could try XGBoost model in order to see if there is any improvement.

