### Importing Libraries & Dataset ###

import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt

dataset = pd.read_csv('new_churn_data.csv')

# Removing the Rows having NaN as the Value
dataset = dataset[pd.notnull(dataset['age'])]
dataset.isna().sum()

## Data Preparation
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])

# One-Hot Encoding
dataset.housing.value_counts()
dataset = pd.get_dummies(dataset)
dataset.columns
dataset = dataset.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])

# Splitting the Dataset into the Training Set & the Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    dataset.drop(columns = 'churn'),
    dataset['churn'],
    test_size = 0.2,
    random_state = 0
)

# Balacing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else :
    higher = neg_index
    lower = pos_index
    
random.seed(0)
higher = np.random.choice(higher, size = len(lower))
lower = np.asarray(lower)
new_indices = np.concatenate((lower, higher))

x_train = x_train.loc[new_indices, ]
y_train = y_train[new_indices]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.transform(x_test))
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values
x_train = x_train2
x_test = x_test2

### Model Building ###

# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0)
cls.fit(x_train, y_train)

# Predicting Test Set
y_pred = cls.predict(x_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cls, X = x_train, y = y_train, cv =10)

# Analyzing Coefficients
pd.concat([
    pd.DataFrame(x_train.columns, columns = ['Features']),
    pd.DataFrame(np.transpose(cls.coef_), columns = ['Coefficients']),
], axis = 1)

### Feature Selection ###
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# To check the Number of Total Columns
x_train.shape

# Model to Test
cls = LogisticRegression()
rfe = RFE(cls, 20)
rfe = rfe.fit(x_train, y_train)

# Summarize the Selection of Attributes
print(rfe.support_)
print(rfe.ranking_)
x_train.columns[rfe.support_]

# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0)
cls.fit(x_train[x_train.columns[rfe.support_]], y_train)

# Predicting Test Set
y_pred = cls.predict(x_test[x_test.columns[rfe.support_]])

# Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cls, X = x_train[x_train.columns[rfe.support_]], y = y_train, cv =10)

# Analyzing Coefficients
pd.concat([
    pd.DataFrame(x_train.columns[rfe.support_], columns = ['Features']),
    pd.DataFrame(np.transpose(cls.coef_), columns = ['Coefficients']),
], axis = 1)

### End of Model ###

# Formatting Final Results
final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop = True)