# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:39:34 2022

@author: Thomas Grubb
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('dementia_dataset.csv', sep=',')

print(data)
data.info()

print(data.dtypes)

data.loc[:, ["Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]]


data.isna().sum()
print(data.SES)
data['SES'].fillna(data['SES'].median(), inplace = True)
data['MMSE'].fillna(data['MMSE'].median(), inplace = True) 


data.info()
print(data)



print(data.Group)

ax=sns.countplot(data=data, x= data.Group)
plt.title("Dementia Count" , size=40)
plt.xlabel('Dementia')
plt.ylabel ('Number')
plt.show()


corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

#Data PreProcessing 
data['Group'] = data['Group'].replace(['Converted'], ['Demented'])
data['M/F'] = data['M/F'].replace(['M', 'F'], [0,1])
data['Group'] = data['Group'].replace(['Nondemented', 'Demented'], [0,1])

data.shape



data['SES'].describe()
data['MMSE'].describe()
data['EDUC'].describe() #Outliers Present
data['CDR'].describe()
data['eTIV'].describe() #Outliers present
data['nWBV'].describe()
data['ASF'].describe()

data['EDUC'].describe() #Outliers Present <8 20< 
plt.boxplot(data['EDUC'])
plt.show()
Q1 = np.percentile(data['EDUC'], 25, interpolation = 'midpoint')
Q3 = np.percentile(data['EDUC'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
print(IQR)

upper = np.where(data['EDUC'] >= (Q3+1.5*IQR))
lower = np.where(data['EDUC'] <= (Q1-1.5*IQR))

data = data[~((data['EDUC'] < (Q1 - 1.5 * IQR))|(data['EDUC'] > (Q3 + 1.5 * IQR)))]
data.shape
data.isna().sum()
data['EDUC'].describe() #Outliers Present <8 20< 
print(data['EDUC'])


data['eTIV'].describe() #Outliers present
plt.boxplot(data['eTIV'])
plt.show()
Q1 = np.percentile(data['eTIV'], 25, interpolation = 'midpoint')
Q3 = np.percentile(data['eTIV'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
print(IQR)
upper = np.where(data['eTIV'] >= (Q3+1.5*IQR))
lower = np.where(data['eTIV'] <= (Q1-1.5*IQR))

data = data[~((data['eTIV'] < (Q1 - 1.5 * IQR))|(data['eTIV'] > (Q3 + 1.5 * IQR)))]
data.shape
data['eTIV'].describe() #Outliers present

boxplot = ["M/F", "SES", "MMSE", "EDUC", "CDR", "nWBV", "ASF"]

xbox = data[boxplot].values
plt.boxplot(xbox)
plt.show()

features = ["M/F", "Age", "SES", "MMSE", "EDUC", "CDR", "eTIV", "nWBV", "ASF"]
x = data[features].values

y = data['Group'].values
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size= 0.75, random_state=1)

#Support Vector Models
clf=svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
precision = metrics.precision_score(y_test, prediction)
recall = metrics.recall_score(y_test, prediction)
f1 = metrics.f1_score(y_test, prediction)
print('The accuracy is:', accuracy)
print('The precision is:', precision)
print('The Recall is:', recall)
print('The f1 is:', f1)
print(classification_report(y_test, prediction))
y_test.shape
prediction.shape
cm = confusion_matrix(y_test, prediction)
print(cm)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix using SVM')
plt.show()


fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)

plt.title('50% SVM ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



#Decision Tree Model
clf2 = DecisionTreeClassifier(random_state=1)
clf2.fit(x_train, y_train)

prediction = clf2.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
precision = metrics.precision_score(y_test, prediction)
recall = metrics.recall_score(y_test, prediction)
f1 = metrics.f1_score(y_test, prediction)
print('The accuracy is:', accuracy)
print('The precision is:', precision)
print('The Recall is:', recall)
print('The f1 is:', f1)

y_test.shape
prediction.shape
cm2 = confusion_matrix(y_test, prediction)
print(cm2)
sns.heatmap(cm2, annot=True)
plt.title('Confusion Matrix using Decision Tree')
plt.show()

fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)

plt.title('25% Decision Tree ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#K-Nearest Neighbour Model 
clf3 = KNeighborsClassifier(algorithm='auto', n_neighbors=5)
clf3.fit(x_train, y_train)

prediction = clf3.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
precision = metrics.precision_score(y_test, prediction)
recall = metrics.recall_score(y_test, prediction)
f1 = metrics.f1_score(y_test, prediction)
print('The accuracy is:', accuracy)
print('The precision is:', precision)
print('The Recall is:', recall)
print('The f1 is:', f1)

y_test.shape
prediction.shape
cm3 = confusion_matrix(y_test, prediction)
print(cm3)
sns.heatmap(cm3, annot=True)
plt.title('Confusion Matrix using K Neighbours')
plt.show()

fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)

plt.title('25% K-Nearest ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1.05])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()