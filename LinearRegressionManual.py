from statistics import mean
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import random

style.use('fivethirtyeight')

#Generate 2 random data sets
def createDataSet(length, variance, step = 2, correlation = False):
    val = 1
    ySet = []
    
    for x in range (length):
        y = val+random.randrange(-variance, variance)
        ySet.append(y)
        if (correlation and correlation == 'positive'):
            val+=step
        elif (correlation and correlation == 'negative'):
            val-=step

    xSet = [x for x in range(len(ySet))]
        
    return np.array(xSet, dtype=np.float64), np.array(ySet, dtype=np.float64)

#Calculate values for m for y=mx+b
def bestFitSlope(xSet, ySet):    
    numerator = ((mean(xSet) * mean(ySet)) - mean(xSet*ySet))
    denominator = (mean(xSet))**2 - mean(xSet**2)

    m = numerator/denominator
    
    return m

#Calculate b for y=mx+b
def intercept(xSet, ySet, m):
    b = mean(ySet)-m*mean(xSet)
    return b

#Calculate r squared (0.0->1.0)
def squaredError(ySetOrig, ySetLine):
    return sum((ySetLine-ySetOrig)**2)

def calcRSquared(ySetOrig, ySetLine):
    yMeanLine = [mean(ySetOrig) for y in ySetOrig]
    squaredErrorRegression = squaredError(ySetOrig, ySetLine)
    squaredErrorMean = squaredError(ySetOrig, yMeanLine)
    return (1-(squaredErrorRegression/squaredErrorMean))

#Test data set, params are length, variance, step, correlation
xSet, ySet = createDataSet(100, 50, 2, correlation='positive')
m = bestFitSlope(xSet,ySet)
b = intercept(xSet,ySet,m)

#Modify predict_x to see what value you are "predicting" for
predictX = 9
predictY = (m*predictX)+b

regressionLine = [(m*x)+b for x in xSet]
rSquared = calcRSquared(ySet, regressionLine)
print(rSquared)

#Draw
plt.scatter(predictX, predictY, color = 'g')
plt.scatter(xSet,ySet)
plt.plot(xSet,regressionLine)
plt.show()
