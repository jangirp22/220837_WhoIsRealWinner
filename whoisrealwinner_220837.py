import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re

# Taking both the data-sets as input.

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Encoding all columns with non-integral values.

string_column_1 = df_train['Party']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(string_column_1)
df_train['Party'] = encoded_values

string_column_2 = df_train['state']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(string_column_2)
df_train['state'] = encoded_values

string_column_3 = df_train['Total Assets']
def extract_numeric(value):
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    else:
        return None
numeric_values = string_column_3.apply(extract_numeric)
df_train['Total Assets'] = numeric_values

string_column_4 = df_train['Liabilities']
def extract_numeric(value):
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    else:
        return None
numeric_values = string_column_4.apply(extract_numeric)
df_train['Liabilities'] = numeric_values

string_column_5 = df_test['Party']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(string_column_5)
df_test['Party'] = encoded_values

string_column_6 = df_test['state']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(string_column_6)
df_test['state'] = encoded_values

string_column_7 = df_test['Total Assets']
def extract_numeric(value):
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    else:
        return None
numeric_values = string_column_7.apply(extract_numeric)
df_test['Total Assets'] = numeric_values

string_column_8 = df_test['Liabilities']
def extract_numeric(value):
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    else:
        return None
numeric_values = string_column_8.apply(extract_numeric)
df_test['Liabilities'] = numeric_values

#Printing total rows of Train and Test Data-set.

print('Number of observations in training data : ',len(df_train))
print('Number of observations in testing data : ',len(df_test))

#Including Training features.

features = df_train.columns[3:8]

#Encoding the column of Education by integeral values.

string_column_9 = df_train['Education']
label_encoder = LabelEncoder()
encoded_values = label_encoder.fit_transform(string_column_9)
df_train['Education'] = encoded_values

y = df_train['Education']

clf = RandomForestClassifier(n_estimators = 100, random_state = 42)

clf.fit(df_train[features], y)

pred_train = clf.predict(df_train[features])
pred_test = clf.predict(df_test[features])

pred_train_names = label_encoder.inverse_transform(pred_train)
pred_test_names = label_encoder.inverse_transform(pred_test)

#Converting encoded values back to original values.

df_test['Predicted Education'] = pred_test_names
df_train['Predicted Education'] = pred_train_names
df_train['Education'] = string_column_9
df_train['Party'] = string_column_1
df_train['State'] = string_column_2
df_test['Party'] = string_column_5
df_test['State'] = string_column_6
df_test['Total Assets'] = string_column_7
df_test['Liabilities'] = string_column_8

accuracy = np.mean(df_train['Education'] == df_train['Predicted Education'])

df_test['Predicted Education'].to_csv('submission_220837.csv', index=True, index_label = 'ID', header=['Education'])

print(df_train.head())
print(df_test.head())

#Accuracy on train data-set.

print(accuracy*100)

# Histogram between no. of criminal cases vs party.

plt.figure(figsize=(10, 6))
criminal_cases_by_party = df_train.groupby('Party')['Criminal Case'].sum().sort_values(ascending=False)
criminal_cases_plot = criminal_cases_by_party.plot(kind='bar', color='skyblue')

plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

for i in criminal_cases_plot.patches:
    percentage = '{:.1f}%'.format(100 * i.get_height() / df_train['Criminal Case'].sum())
    x = i.get_x() + i.get_width() / 2 - 0.15
    y = i.get_height() + 0.2
    plt.annotate(percentage, (x, y), fontsize=10, ha='center')

plt.title('Number of Criminal Cases by Party')
plt.xlabel('Party')
plt.ylabel('Number of Criminal Cases')
plt.tight_layout()
plt.show()

# Histogram between wealth (Total assets - Liabilities) vs party.

df_train['Wealth'] = df_train['Total Assets'] - df_train['Liabilities']

plt.figure(figsize=(10, 6))
wealth_by_party = df_train.groupby('Party')['Wealth'].mean().sort_values(ascending=False)
wealth_plot = wealth_by_party.plot(kind='bar', color='salmon')

plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

for i in wealth_plot.patches:
    percentage = '{:.1f}%'.format(100 * i.get_height() / df_train['Wealth'].sum())
    x = i.get_x() + i.get_width() / 2 - 0.15
    y = i.get_height() + 0.2
    plt.annotate(percentage, (x, y), fontsize=10, ha='center')

plt.title('Wealth by Party')
plt.xlabel('Party')
plt.ylabel('Wealth (in crores)')
plt.tight_layout()
plt.show()

df_train['Total Assets'] = string_column_3
df_train['Liabilities'] = string_column_4
