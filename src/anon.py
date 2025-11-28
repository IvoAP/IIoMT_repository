import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans


def anonimization(data, alpha: float = 1.0):
    """
    Anonymize the data using the PPM-Anon idea with a perturbation control parameter alpha.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix with shape (n_samples, n_features).
    alpha : float, optional (default=1.0)
        Perturbation level in [0, 1]:
            - alpha = 0.0 -> no perturbation (data is reconstructed exactly).
            - alpha = 1.0 -> maximum perturbation (original PPM-Anon behavior with fully shuffled eigenvectors).
            - 0.0 < alpha < 1.0 -> partial perturbation (interpolation between original and shuffled eigenvectors).

    Returns
    -------
    data_anonymized : np.ndarray
        Anonymized data with the same shape as the original data.
    """

    # --- 1) Compute the mean of each column (feature-wise mean) ---
    mean = np.array(np.mean(data, axis=0).T)

    # --- 2) Center data by subtracting the mean ---
    data_centered = data - mean

    # --- 3) Compute covariance matrix of the centered data ---
    # rowvar=False -> each column represents a variable (feature)
    cov_matrix = np.cov(data_centered, rowvar=False)

    # --- 4) Eigen-decomposition of the covariance matrix ---
    # la.eigh is used for symmetric (Hermitian) matrices like covariance matrices
    evals, evecs = la.eigh(cov_matrix)

    # --- 5) Sort eigenvalues and eigenvectors in descending order of eigenvalues ---
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    # --- 6) (Optional) variance retained just for reference/analysis ---
    variance_retained = np.cumsum(evals) / np.sum(evals)
    # Note: variance_retained is not used directly in the anonymization,
    # but could be useful if you want to decide how many components to keep.

    # --- 7) Project data into the eigenvector basis (PCA-like transform) ---
    # shape: (n_samples, n_features) -> (n_samples, n_features) in the eigen-space
    data_transformed = np.dot(evecs.T, data_centered.T).T

    # --- 8) Create a shuffled version of the eigenvectors (full perturbation basis) ---
    # We copy and transpose so that each row in new_evecs represents one eigenvector.
    shuffled_evecs = evecs.copy().T
    for i in range(len(shuffled_evecs)):
        # This shuffles the entries of each eigenvector, breaking the original structure
        np.random.shuffle(shuffled_evecs[i])
    shuffled_evecs = np.array(shuffled_evecs).T  # back to shape (n_features, n_features)

    # --- 9) Interpolate between original and shuffled eigenvectors using alpha ---
    # alpha controls how far we move from the original basis to the shuffled basis.
    # alpha = 0.0 -> new_evecs = original evecs  (no perturbation)
    # alpha = 1.0 -> new_evecs = shuffled_evecs (maximum perturbation)
    # 0 < alpha < 1 -> convex combination of both bases
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in the range [0, 1].")

    new_evecs = (1.0 - alpha) * evecs + alpha * shuffled_evecs

    # --- 10) Map data back to the original feature space using the new eigenvectors ---
    data_original_dimension = np.dot(data_transformed, new_evecs.T)
    data_original_dimension += mean  # add the mean back to undo centering

    return data_original_dimension


def find_clusters(X, k: int):
    """
    Cluster the data into k clusters using KMeans.

    Parameters
    ----------
    X : np.ndarray
        Input data with shape (n_samples, n_features).
    k : int
        Number of clusters.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample (shape: (n_samples,)).
    """
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.labels_


def anonimization_clustering(data, y, k: int, alpha: float = 1.0):
    """
    Cluster-based PPM-Anon anonymization with controllable perturbation level.

    The idea:
        1) Split the data into k clusters.
        2) Apply the anonymization procedure independently inside each cluster.
        3) Concatenate all anonymized clusters back into a single dataset.

    Parameters
    ----------
    data : np.ndarray
        Original data matrix with shape (n_samples, n_features).
    y : np.ndarray
        Labels (or any target vector) associated with the samples (shape: (n_samples,)).
        The labels are only permuted according to the cluster order to keep alignment with data.
    k : int
        Number of clusters.
    alpha : float, optional (default=1.0)
        Perturbation level forwarded to the `anonimization` function.

    Returns
    -------
    data_anonymized : np.ndarray
        Anonymized data with the same shape as the original data.
    y_in_new_order : np.ndarray
        Labels reordered to match the anonymized data.
    """

    # --- 1) Generate k data clusters using KMeans ---
    clusters = find_clusters(data, k)

    # --- 2) Bucketize sample indices by cluster label ---
    indices = dict()
    for i in range(len(clusters)):
        if clusters[i] not in indices.keys():
            indices[clusters[i]] = []
        indices[clusters[i]].append(i)

    data_anonymized, y_in_new_order = None, None

    # --- 3) Anonymize each cluster independently using the PPM-Anon mechanism ---
    for cluster_label, idx_list in indices.items():
        cluster_data = data[idx_list]
        cluster_labels = y[idx_list]

        # Apply anonymization with given alpha
        cluster_data_anon = anonimization(cluster_data, alpha=alpha)

        if data_anonymized is None and y_in_new_order is None:
            # First cluster initializes the arrays
            data_anonymized = cluster_data_anon
            y_in_new_order = cluster_labels
        else:
            # Concatenate anonymized cluster data and corresponding labels
            data_anonymized = np.concatenate((data_anonymized, cluster_data_anon), axis=0)
            y_in_new_order = np.concatenate((y_in_new_order, cluster_labels), axis=0)

    return data_anonymized, y_in_new_order
