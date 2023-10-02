#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/


#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

outlook = {"Sunny":1, "Overcast":2, "Rain":3}
temp = {"Hot":1,"Mild":2,"Cool":3}
humidity = {"High":1, "Normal":2}
wind = {"Weak":1, "Strong":2}
label = {"No":1, "Yes":2}

#reading the training data in a csv file
#--> add your Python code here
dbTraining = []
X = []
Y = []

#reading the training data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append(row)

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2,2], ...]]
#--> add your Python code here


            X.append([outlook[row[1]], temp[row[2]], humidity[row[3]], wind[row[4]]])
       

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

            Y.append(label[row[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here

dbTesting = []
X = []


    #reading the training data in a csv file
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTesting.append(row)
                X.append([outlook[row[1]], temp[row[2]], humidity[row[3]], wind[row[4]]])


#printing the header os the solution

print(f"{'Day' : <5}{'Outlook':^10}{'Temperature':^15}{'Humidity':^15}{'Wind':^10}{'PlayTennis':^15}{'Confidence':>10}")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here

pred = clf.predict_proba(X)
count = 0

label = {1:"No", 2:"Yes"}
for row in dbTesting:
    inst = list(pred[count])

    if max(inst) >= 0.75:
         print(f"{row[0]:<5}{row[1]:^10}{row[2]:^15}{row[3]:^15}{row[4]:^10}{label[inst.index(max(inst)) + 1]:^15}{'{num:.2f}'.format(num=max(inst)):^10}")
    count += 1
