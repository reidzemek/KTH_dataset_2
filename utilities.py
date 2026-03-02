import numpy as np

def farthest_point_sampling(points, k):
    N = points.shape[0]
    centroids = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Deterministic first point: farthest from the center
    center = points.mean(axis=0)
    farthest = np.argmax(np.linalg.norm(points - center, axis=1))

    for i in range(k):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = np.argmax(distances)

    return points[centroids]