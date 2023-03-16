import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
#import the data set and evaluate it
df=pd.read_csv("insurance.csv")
print(df.head())


#find the correlation among the features
#corr= df.corr()
#print(corr)
#convert the data frame into a numpy matrix
dfn=df.to_numpy()
print(dfn.shape)

#split the data 
x=dfn[:,[0,1,3,4]] #Indexing all numerical values in the dataframe
y=dfn[:,3] #target value, in this case the chighest value of the day

#split the data in training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)
#trained model
clf = LogisticRegression(lr=0.001)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)
# user input

while True:
    print("type in just numbers  /n")
    age = float(input("How old are you? "))
    bmi = float(input("What is your bmi? "))
    children = float(input("How many kids do you have?  ")) 
    charges = float(input ("Your insurance monhtly charges:  "))

    x_user=[[age, bmi, children, charges]]

    y_s = clf.predict(x_user)

    if y_s==1:
            print("stop smoking homes (smoker)")
    else:
         print("you doing good broski (No smoker)")
    
    user =input("Wanna make another prediction hommie?")

    if user=="n":
        break