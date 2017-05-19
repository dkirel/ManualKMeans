import numpy as np

from matplotlib import style, pyplot as plt

style.use('ggplot')
colors = 10*['b', 'g', 'r', 'c', 'm', 'y', 'w']

class KMeans:

    def __init__(self, k, epsilon=0.001, max_iterations=1000, visualize=True):
        self.k = k
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.visualize = visualize

    def fit(self, X):
        # Randomly select initial centroids
        centroids = X[:self.k, :]

        for iteration in range(self.max_iterations):
            # Calculate distances from centroids
            distances = [[np.linalg.norm(x - c) for c in centroids] for x in X]

            # Classify featuresets based on distance from centroids
            classifications = {i: [] for i in range(self.k)}
            for i, d in enumerate(distances):
                classifications[np.argmin(d)].append(X[i, :])

            # Calculate new centroid mean of each class
            prev_centroids = np.copy(centroids)
            centroids = np.array([np.average(x_k, axis=0) for k, x_k in classifications.items()])

            """
            print('Iteration: ', iteration)
            print('Distances: ', distances)
            print('Classifications: ', classifications)
            print('Prev: ', prev_centroids)
            print('Curr: ', centroids)
            """

            # Check for convergence
            pct_change = (centroids - prev_centroids)/prev_centroids

            if not np.any(pct_change > self.epsilon):
                break

        # Save centroids
        self.centroids = centroids

        # Show plot of training data and centroids
        if self.visualize:
            for c in centroids:
                plt.scatter(c[0], c[1], marker="o", color="k", s=150, linewidths=5)

            for c in classifications:
                color = colors[c]
                for x in classifications[c]:
                    plt.scatter(x[0], x[1], marker="x", color=color, s=150, linewidths=5)
                    
            plt.show()
            

    def predict(self, x):
        distances = [np.sqrt(((X - c)**2).sum(axis=1)) for c in centroids]
        return np.argmin(distances)
