from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:

    @staticmethod
    def show_2d_scatter(X, y, title="PCA visualization of sequence embeddings"):
        # Apply PCA to reduce dimensionality to 2 components
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Plotting the PCA result
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.xlabel("PCA component 1")
        plt.ylabel("PCA component 2")
        plt.show()

    @staticmethod
    def show_3d_scatter(X, y, title="3D PCA visualization of sequence embeddings"):
        # Apply PCA to reduce dimensionality to 2 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        # Plotting the PCA result in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        ax.set_title(title)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.set_zlabel("PCA component 3")
        plt.show()
