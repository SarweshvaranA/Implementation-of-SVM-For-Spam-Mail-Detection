# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries and load the spam.csv dataset with the correct encoding.
2. Extract the message texts into x and the labels (spam/ham) into y.
3. Split the data into training and testing sets using an 80-20 ratio.
4. Convert the text data into numerical vectors using CountVectorizer.
5. Initialize a Support Vector Machine (SVM) classifier.
6. Train the SVM model using the training data.
7. Predict the labels for the test data using the trained model.
8. Evaluate model performance using accuracy, confusion matrix, and classification report.

## Program & Output:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: SARWESHVARAN A
RegisterNumber:  212223230198
```
```python
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
```
![image](https://github.com/user-attachments/assets/238928b6-3165-466e-9897-540c9a307c45)

```python
data.shape
```
![image](https://github.com/user-attachments/assets/ea081387-7384-40f5-9656-721bf42b2ce5)

```python
x=data['v2'].values
y=data['v1'].values
x.shape

```
![image](https://github.com/user-attachments/assets/36457989-ce2b-4887-9fca-2e0c5f1d81a4)

```python
y.shape
```
![image](https://github.com/user-attachments/assets/e5357457-3789-4f3b-9ed1-a0f87cfa1b1a)

```python

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

```
![image](https://github.com/user-attachments/assets/fd7d3eb0-eb26-4187-bb45-51fab2762c0b)

```python
x_train.shape
```
![image](https://github.com/user-attachments/assets/2b77eb22-5bc6-4298-801f-ca839d8bfa46)

```python
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

```
![image](https://github.com/user-attachments/assets/7b63fb3f-0203-4657-8a79-29645ae77e72)

```python
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc
```
![image](https://github.com/user-attachments/assets/29b84e5f-5ea1-4277-926b-f0acabb201ea)

```python
con=confusion_matrix(y_test,y_pred)
print(con)

```
![image](https://github.com/user-attachments/assets/625862b1-0203-4cf9-a029-7ba84c17574b)

```python
cl=classification_report(y_test,y_pred)
print(cl)
```
![image](https://github.com/user-attachments/assets/b244fe5b-3d12-4d25-946a-97c1505b8932)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
