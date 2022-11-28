# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Apply new unknown values 


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: A.sharmila
RegisterNumber: 212221230094 
import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

## Original Data:

![D1](https://user-images.githubusercontent.com/94506182/200760286-33249055-5b86-4773-8ba7-31921250b84a.png)

## Label Encoder:

![m2](https://user-images.githubusercontent.com/94506182/200760520-d2c1f023-e840-460d-a342-b32e0027e262.png)

## X:

![m3](https://user-images.githubusercontent.com/94506182/200760768-62a9ad2d-32ea-4f77-9e9f-cd897c27978f.png)

## Y:

![y](https://user-images.githubusercontent.com/94506182/200760906-359de198-2b13-4632-b526-a75f9f34d885.png)

## Y_prediction:

![yp](https://user-images.githubusercontent.com/94506182/200761051-203f3076-ec3a-465c-87ca-8bf168c6f2b9.png)

## Accuracy:

![Acc](https://user-images.githubusercontent.com/94506182/200761179-2c75c938-f3aa-411c-8220-04576223ec3e.png)

## Cofusion:

![AR](https://user-images.githubusercontent.com/94506182/200761284-c66d45e2-e29e-4b9a-ba85-eeae3d24dec8.png)

## Classification:

![P0](https://user-images.githubusercontent.com/94506182/200761425-60962295-4c98-4df2-be6b-973b1a0e66c4.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
