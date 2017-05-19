import numpy as np

from KMeans import KMeans

X = np.array([[0.5, 2],
                [1, 2],
                [1, 4.5],
                [1, 6],
                [1.5, 1.8],
                [2, 8],
                [2.5, 3],
                [5, 8 ],
                [8, 8],
                [1, 0.6],
                [9,11],
                [10, 13]])

c = KMeans(k=3)
c.fit(X)

print('Centroids: ', c.centroids)
