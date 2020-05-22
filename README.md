# importing libraries for data handling and analysis
import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import ExcelWriter
from pandas import ExcelFile
from openpyxl import load_workbook
import numpy as np
from scipy.stats import norm, skew
from scipy import stats
import statsmodels.api as sm

# importing libraries for data visualisations
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
%matplotlib inline
color = sns.color_palette()

# Read Excel file
df_sourcefile = pd.read_excel(
    'proof.xlsx', sheet_name=0)
print("Shape of dataframe is: {}".format(df_sourcefile.shape))

# Make a copy of the original sourcefile
df_HR = df_sourcefile.copy()
df = df_sourcefile.copy()

(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'Yes', 'satisfaction_level'])
print(
    'Ex-employees: average satisfaction level = {:.1f} years old and standard deviation = {:.1f}'.format(mu, sigma))
(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'No', 'satisfaction_level'])
print('Current employees: average satisfaction level = {:.1f} years old and standard deviation = {:.1f}'.format(
    mu, sigma))



plt.figure(figsize=(5,5))
sns.barplot(x='Attrition', y='satisfaction_level', data= df_HR);

(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'Yes', 'last_evaluation'])
print(
    'Ex-exmployees: average last evaluation = {:.2f} years old and standard deviation = {:.1f}'.format(mu, sigma))
(mu, sigma) = norm.fit(df_HR.loc[df_HR['Attrition'] == 'No', 'last_evaluation'])
print('Current exmployees: average last evaluation = {:.2f} years old and standard deviation = {:.1f}'.format(
    mu, sigma))

plt.figure(figsize=(5,5))
sns.barplot(x='Attrition', y='last_evaluation', data= df);

(mu, sigma) = norm.fit(df.loc[df['Attrition'] == 'Yes', 'average_montly_hours'])
print(
    'Ex-exmployees: average average monthly hours = {:.2f} years old and standard deviation = {:.1f}'.format(mu, sigma))
(mu, sigma) = norm.fit(df.loc[df['Attrition'] == 'No', 'average_montly_hours'])
print('Current exmployees: average average monthly hours = {:.2f} years old and standard deviation = {:.1f}'.format(
    mu, sigma))

plt.figure(figsize=(5,5))
sns.barplot(x='Attrition', y='average_montly_hours', data= df);

print("Percentage of Current Employees is {:.1f}% and of Ex-employees is: {:.1f}%".format(
    df[df['Attrition'] == 'No'].shape[0] / df.shape[0]*100,
    df[df['Attrition'] == 'Yes'].shape[0] / df.shape[0]*100))

df['salary']=df['salary'].replace(['low','medium'],[1,2])
df['dept']=df['dept'].replace(['sales'],[1])

# Find correlations with the target and sort
df_trans = df.copy()
df_trans['Target'] = df_trans['Attrition'].apply(
    lambda x: 0 if x == 'No' else 1)
df_trans = df_trans.drop(
    ['Attrition','time_spend_company','Work_accident','promotion_last_5years'], axis=1)
correlations = df_trans.corr()['Target'].sort_values()
print('Most Positive Correlations: \n', correlations.nlargest(5))
print('\nMost Negative Correlations: \n', correlations.head(5))

features=['number_project','time_spend_company','Work_accident','Attrition', 'promotion_last_5years','dept','salary']

fig=plt.subplots(figsize=(15,30))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = df, hue='Attrition',)
    plt.xticks(rotation=90)
    plt.title("No. of employee")
    
# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
df['salary']=le.fit_transform(df['salary'])
df['dept ']=le.fit_transform(df['dept'])
df['Attrition']=df['Attrition'].replace(['Yes','No'],[1,2])


#Spliting data into Feature and
X=df[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'dept ', 'salary']]
y=df['Attrition']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)

y_pred


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


# Read Excel file
likely_leaver = pd.read_excel(
    'proof.xlsx', sheet_name=1)
print("Shape of dataframe is: {}".format(likely_leaver.shape))


#Predict the response for test dataset
y_pred2 = gb.predict(likely_leaver)

y_pred2
