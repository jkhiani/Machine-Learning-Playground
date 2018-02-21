import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import random
from matplotlib import style
from collections import Counter
from math import sqrt

style.use('fivethirtyeight')

def kNearestNeighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups')
    if k%2==0:
        warnings.warn('K is an even value and can cause issues')

    #Calculate distance of newSet to ALL other points  
    distances = []
    for group in data:
        for features in data[group]:
            distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([distance,group])

    #Take the k most
    votes = []
    for x in sorted(distances)[:k]:
        votes.append(x[1])
    
    #Take the most
    votesResult = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k

    return votesResult, confidence

###Create 2 classes k,r with 3 points each
##dataSet = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
##
###Want to determine wether newSet goes in k or r 
##newSet = [5,7]
##result = kNearestNeighbors(dataSet, newSet, k=3)
##print(result)
##
##for x in dataSet:
##    for y in dataSet[x]:
##        plt.scatter(y[0], y[1], s=50, color=x)

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
fullData = df.astype(float).values.tolist()

random.shuffle(fullData)
testSize = 0.2
trainingSet = {2:[], 4:[]}
testSet = {2:[], 4:[]}
trainData = fullData[:-int(testSize*len(fullData))]
testData = fullData[-int(testSize*len(fullData)):]

#Create the dictionaries using the data given
for x in trainData:
    trainingSet[x[-1]].append(x[:-1])

for x in testData:
    testSet[x[-1]].append(x[:-1])

correct = 0
total = 0

for group in testSet:
    for data in testSet[group]:
        vote, confidence = kNearestNeighbors(trainingSet, data ,k=5)
        if group==vote:
            correct+=1
        else:
            print ('Confidence when wrong:', confidence)
        total +=1

print ('Accuracy:', correct/total)

##plt.scatter(newSet[0], newSet[1], s=100, color=result)
##plt.show()
