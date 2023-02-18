import numpy as np

def distEuclidean(a, b):
    dist = np.linalg.norm(a-b)
    return dist

class KNearestNeighbor:
    def __init__(self, k_neighbors = 1):
        self.k_neighbors = k_neighbors
        self.distance = distEuclidean
        self.X = None

    def fit(self, x, labels):
        # combine instances and their labels
        self.X = np.hstack((x,labels))

    def kneighbors(self, Xp):
        predicted_labels = np.zeros((len(Xp), 1))
        for i,xp in enumerate(Xp):
            neighbors = []
            # find distances of instances
            dists = {self.distance(xp, x[:-1]): x for x in self.X}
            for key in sorted(dists.keys())[:self.k_neighbors]:
                # Add the labels of the k nearest neighbors
                label = int(dists[key][-1])
                neighbors.append(label)
            neighbors = np.array(neighbors).flatten()
            # get the most common label
            counts = np.bincount(neighbors)
            label = np.argmax(counts)
            # label = max(set(neighbors), key = neighbors.count)
            predicted_labels[i] = label
        return predicted_labels

