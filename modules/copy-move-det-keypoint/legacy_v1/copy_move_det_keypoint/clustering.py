import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def cluster_keypoints(keypoints, image_shape, conn_neighbor_rate=0.1, dist_thresh_rate=0.003, cpu_count=-1):
    """
    Clusterizes the given keypoints according to their (x, y) positions within the source image.
    
    Args:
        keypoints: numpy array of shape (N, 2) containing (x, y) coordinates.
        image_shape: tuple (height, width) of the source image.
        conn_neighbor_rate: Rate in [0.0, 1.0] to define the number of keypoints used to lock connectivity.
        dist_thresh_rate: Rate in [0.0, 1.0] to define the distance threshold.
        cpu_count: Number of CPU cores used in clustering. -1 to use all cores.
        
    Returns:
        list: List of lists, where each inner list contains the indices of the clustered keypoints.
    """
    if len(keypoints) == 0:
        return []

    if len(keypoints) == 1:
        return [[0]]

    # Calculate parameters
    nb_count = int(round(len(keypoints) * conn_neighbor_rate))
    # Heuristic for distance threshold based on image area
    dist_thresh = image_shape[0] * image_shape[1] * dist_thresh_rate

    # Perform clustering
    if nb_count > 0:
        # Use connectivity constraint if enough neighbors
        forced_conn = kneighbors_graph(X=keypoints, n_neighbors=nb_count, n_jobs=cpu_count)
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            connectivity=forced_conn, 
            distance_threshold=dist_thresh,
            linkage='ward' # Default linkage
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=dist_thresh,
            linkage='ward'
        )

    clustering.fit(keypoints)
    labels = clustering.labels_

    # Group indices by label
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return list(clusters.values())
