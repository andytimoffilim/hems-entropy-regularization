import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import faiss
from typing import Tuple, List, Dict
from scipy.special import entr


class EmbeddingAnalyzer:
    """Analyze embedding distributions and compute geometry metrics."""

    def __init__(self, embedding_dim: int, num_centroids: int = 256):
        self.embedding_dim = embedding_dim
        self.num_centroids = num_centroids
        self.centroids = None
        self.partition = None

    def extract_embeddings(self, model, data_loader, device='cuda'):
        """Extract embeddings and labels from data loader."""
        model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(device)
                embeddings, _ = model(data)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        return embeddings, labels

    def fit_quantization(self, embeddings: np.ndarray):
        """Fit spherical k-means centroids for quantization."""
        # Normalize embeddings to unit sphere
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Use FAISS for efficient spherical k-means
        dim = embeddings.shape[1]

        # Convert to float32 for FAISS
        embeddings = embeddings.astype(np.float32)

        # Create index
        index = faiss.index_factory(dim, f"IVF{self.num_centroids},Flat", faiss.METRIC_INNER_PRODUCT)

        # Train index
        index.train(embeddings)
        index.add(embeddings)

        # Get centroids
        centroids = index.reconstruct_n(0, index.ntotal)

        # Find nearest centroid for each point
        distances, assignments = index.search(embeddings, 1)

        self.centroids = centroids
        self.assignments = assignments.flatten()

        return self.assignments

    def compute_quantization_entropy(self, embeddings: np.ndarray) -> float:
        """Compute empirical quantization entropy H_hat_M."""
        if self.centroids is None:
            self.fit_quantization(embeddings)

        # Count occupancy
        unique, counts = np.unique(self.assignments, return_counts=True)
        p_j = counts / len(self.assignments)

        # Compute entropy
        entropy = -np.sum(p_j * np.log(p_j + 1e-10))

        return entropy

    def compute_effective_support(self, embeddings: np.ndarray, eta: float = 0.05) -> int:
        """Compute effective support K_hat_M(eta)."""
        if self.centroids is None:
            self.fit_quantization(embeddings)

        # Count occupancy
        unique, counts = np.unique(self.assignments, return_counts=True)
        p_j = counts / len(self.assignments)

        # Sort probabilities in descending order
        sorted_p = np.sort(p_j)[::-1]

        # Find minimum number of cells to cover (1 - eta) mass
        cumsum = np.cumsum(sorted_p)
        k = np.searchsorted(cumsum, 1 - eta) + 1

        return int(k)

    def compute_rare_cell_mass(self, embeddings: np.ndarray, a: float = 0.005) -> float:
        """Compute rare cell mass tau_hat_M(a)."""
        if self.centroids is None:
            self.fit_quantization(embeddings)

        # Count occupancy
        unique, counts = np.unique(self.assignments, return_counts=True)
        p_j = counts / len(self.assignments)

        # Sum mass of cells with p_j < a
        rare_mass = np.sum(p_j[p_j < a])

        return rare_mass

    def compute_coverage(self, train_embeddings: np.ndarray, test_embeddings: np.ndarray) -> float:
        """Compute empirical distribution-weighted coverage R_hat."""
        # L2-normalize
        train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

        # Use FAISS for efficient nearest neighbor search
        dim = train_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(train_embeddings.astype(np.float32))

        distances, _ = index.search(test_embeddings.astype(np.float32), 1)
        mean_distance = np.mean(distances)

        return mean_distance

    def compute_class_conditional_coverage(self, train_embeddings: np.ndarray, train_labels: np.ndarray,
                                           test_embeddings: np.ndarray, test_labels: np.ndarray) -> Dict:
        """Compute class-conditional coverage."""
        coverage_dict = {}

        for class_id in np.unique(test_labels):
            # Filter embeddings for this class
            train_mask = train_labels == class_id
            test_mask = test_labels == class_id

            if np.sum(train_mask) > 0 and np.sum(test_mask) > 0:
                train_emb = train_embeddings[train_mask]
                test_emb = test_embeddings[test_mask]

                coverage = self.compute_coverage(train_emb, test_emb)
                coverage_dict[class_id] = coverage

        return coverage_dict

    def compute_effective_rank(self, embeddings: np.ndarray) -> float:
        """Compute effective rank from covariance matrix."""
        # Center embeddings
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)

        # Compute covariance
        cov = np.cov(centered.T)

        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zero eigenvalues

        # Normalize
        normalized_eigenvalues = eigenvalues / eigenvalues.sum()

        # Compute effective rank (exponential of Shannon entropy)
        effective_rank = np.exp(-np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues + 1e-10)))

        return effective_rank

    def compute_pairwise_similarities(self, embeddings: np.ndarray) -> Dict:
        """Compute pairwise cosine similarities."""
        # L2-normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Exclude diagonal (self-similarity)
        np.fill_diagonal(sim_matrix, np.nan)

        return {
            'mean': np.nanmean(sim_matrix),
            'std': np.nanstd(sim_matrix),
            'max': np.nanmax(sim_matrix),
            'min': np.nanmin(sim_matrix)
        }

    def compute_all_metrics(self, train_embeddings: np.ndarray, test_embeddings: np.ndarray,
                            train_labels: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """Compute all geometry metrics."""
        metrics = {}

        # Fit quantization on train embeddings
        self.fit_quantization(train_embeddings)

        # Quantization metrics
        metrics['H_hat_M'] = self.compute_quantization_entropy(test_embeddings)
        metrics['K_hat_M_eta_0.05'] = self.compute_effective_support(test_embeddings, eta=0.05)
        metrics['K_hat_M_eta_0.10'] = self.compute_effective_support(test_embeddings, eta=0.10)
        metrics['tau_hat_M_0.001'] = self.compute_rare_cell_mass(test_embeddings, a=0.001)
        metrics['tau_hat_M_0.005'] = self.compute_rare_cell_mass(test_embeddings, a=0.005)
        metrics['tau_hat_M_0.01'] = self.compute_rare_cell_mass(test_embeddings, a=0.01)

        # Coverage metrics
        metrics['R_hat'] = self.compute_coverage(train_embeddings, test_embeddings)
        class_coverage = self.compute_class_conditional_coverage(
            train_embeddings, train_labels, test_embeddings, test_labels
        )
        metrics['R_hat_class_mean'] = np.mean(list(class_coverage.values()))

        # Effective rank
        metrics['effective_rank'] = self.compute_effective_rank(test_embeddings)

        # Pairwise similarities
        sim_metrics = self.compute_pairwise_similarities(test_embeddings)
        metrics['mean_pairwise_sim'] = sim_metrics['mean']
        metrics['std_pairwise_sim'] = sim_metrics['std']

        # Class-wise dispersion (within-class distances)
        class_dispersions = []
        for class_id in np.unique(test_labels):
            class_embeddings = test_embeddings[test_labels == class_id]
            if len(class_embeddings) > 1:
                # Compute mean pairwise distance within class
                dist_matrix = pairwise_distances(class_embeddings)
                class_dispersions.append(np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)]))

        metrics['mean_within_class_dispersion'] = np.mean(class_dispersions) if class_dispersions else 0.0

        # Inter-class geometry (centroid distances)
        centroids = []
        for class_id in np.unique(test_labels):
            class_embeddings = test_embeddings[test_labels == class_id]
            centroids.append(np.mean(class_embeddings, axis=0))

        centroids = np.array(centroids)
        centroid_distances = pairwise_distances(centroids)
        metrics['mean_inter_class_distance'] = np.mean(
            centroid_distances[np.triu_indices_from(centroid_distances, k=1)])

        # Ratio
        if metrics['mean_within_class_dispersion'] > 0:
            metrics['inter_intra_ratio'] = metrics['mean_inter_class_distance'] / metrics[
                'mean_within_class_dispersion']
        else:
            metrics['inter_intra_ratio'] = float('inf')

        return metrics