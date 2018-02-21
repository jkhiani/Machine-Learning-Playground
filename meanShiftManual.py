import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

x = np.array([[[1,2],
               [1.5,1.8],
               [5,8],
               [8,8],
               [1,0.6],
               [9,11],
               [8,2],
               [10,2],
               [9,3]]])

colors = 10*['g','r','c','b','k']

class meanShift:
    def __init__(self, radius=4):
        self.radius = radius

    def fit(self,data):
        centroids = {}

        for x in range(len(data)):
            centroids[x] = data[x]

        while True:
            newCentroids = []

            for x in centroids:
                inBandwidth = []
                centroid = centroids[x]

                for featureSet in data:
                    if np.linalg.norm(featureSet-centroid) < self.radius:
                        inBandwidth.append(featureSet)

                newCentroid = np.average(inBandwidth,axis=0)
                newCentroids.append(newCentroid)

            uniques = sorted(list(set(newCentroids)))

            prevCentroids = dict(centroids)
            currCentroids = {}

            for x in range(len(uniques)):
                centroids[x] = np.array(uniques[x])

            optimized = True

            for x in centroids:
                if not np.array_equal(centroids[x], prevCentroids[x]):
                    optimized = False

                if not optimized:
                    break

            if optimized:
                break

            if prevCentroids == currCentroids:
                break;

        self.centroids = centroids

    def predict(self, data):
        pass

clf = meanShift()
clf.fit(x)

centroids = clf.centroids

plt.scatter(x[:,0], x[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()

        

        
