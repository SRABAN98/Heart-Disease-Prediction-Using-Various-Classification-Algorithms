#importing the necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importing the dataset

dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Resume Project - Heart Disease Prediction\framingham.csv")


#recognizing "x" and "y" from the dataset

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#imputing the missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x)
x = imputer.transform(x)


#splitting the dataset into training and testing phase

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=0)


#feature scalling the independent variables for improving the model accuracy

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#training the different possible classification model with same dataset at parameter tuning to compare the accuracy score
#the model which is having highest accuracy ,we will go for the deployment of that model


#1.LOGISTIC REGRESSION
#--------------------------------


#training of the dataset

from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression()
classifier_LR.fit(x_train,y_train)


#predicting the test set results

y_pred_LR = classifier_LR.predict(x_test)


#evaluating the confusion matrix

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test,y_pred_LR)
print(cm_LR)


#accuracy of the model

from sklearn.metrics import accuracy_score
print('Accuracy of LOGISTIC REGRESSION model :' ,accuracy_score(y_test,y_pred_LR))



#2.SUPPORT VECTOR CLASSIFIRS
#--------------------------------


#training of the dataset

from sklearn.svm import SVC
classifier_SVC = SVC()
classifier_SVC.fit(x_train,y_train)


#predicting the test set results

y_pred_SVC = classifier_SVC.predict(x_test)


#evaluating the confusion matrix

from sklearn.metrics import confusion_matrix
cm_SVC = confusion_matrix(y_test,y_pred_SVC)
print(cm_SVC)


#accuracy of the model

from sklearn.metrics import accuracy_score
print('Accuracy of SUPPORT VECTOR CLASSIFIER model :' ,accuracy_score(y_test,y_pred_SVC))



#3.K-Nearest Neighbours
#--------------------------------


#training of the dataset

from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier()
classifier_KNN.fit(x_train,y_train)


#predicting the test set results

y_pred_KNN = classifier_KNN.predict(x_test)


#evaluating the confusion matrix

from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test,y_pred_KNN)
print(cm_KNN)


#accuracy of the model

from sklearn.metrics import accuracy_score
print('Accuracy of K-NEAREST NEIGHBOURS model :' ,accuracy_score(y_test,y_pred_KNN))



#4.BERNOULLI NAIVE BAYES
#--------------------------------


#training of the dataset

from sklearn.naive_bayes import BernoulliNB
classifier_BNB = BernoulliNB()
classifier_BNB.fit(x_train,y_train)


#predicting the test set results

y_pred_BNB = classifier_BNB.predict(x_test)


#evaluating the confusion matrix

from sklearn.metrics import confusion_matrix
cm_BNB = confusion_matrix(y_test,y_pred_BNB)
print(cm_BNB)


#accuracy of the model

from sklearn.metrics import accuracy_score
print('Accuracy of BERNOULLI NAIVE BAYES model :' ,accuracy_score(y_test,y_pred_BNB))



#5.GAUSSIAN NAIVE BAYES
#--------------------------------


#training of the dataset

from sklearn.naive_bayes import GaussianNB
classifier_GNB = GaussianNB()
classifier_GNB.fit(x_train,y_train)


#predicting the test set results

y_pred_GNB = classifier_GNB.predict(x_test)


#evaluating the confusion matrix

from sklearn.metrics import confusion_matrix
cm_GNB = confusion_matrix(y_test,y_pred_GNB)
print(cm_GNB)


#accuracy of the model

from sklearn.metrics import accuracy_score
print('Accuracy of GAUSSIAN NAIVE BAYES model :' ,accuracy_score(y_test,y_pred_GNB))



#From the above model building we came to know that,the accuracy of
# LR > SVC > KNN > GNB > BNB
#Thus we will go for the deployment of the LOGISTIC REGRESSION model .
#Here,We can't build the model using Multinomial Naive Bayes, because we can't use multinomial Naive Bayes where the Dependent Variable is Binary..
#Here,We can't build the model using Tree Algorithms (like Decission Tree and Random Forest) and Boosting Algorithms (like ADABoost and Gradient Boosting),because these algorithms doesn't required feature scalling..and here we have applied feature scalling in order to scale the data.....
