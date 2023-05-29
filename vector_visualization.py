import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS

def visualize_word_distances(word_list, word_embeddings):
    # Calculate pairwise cosine distances between word embeddings
    distance_matrix = cosine_distances(word_embeddings)

    # Apply Multidimensional Scaling (MDS) to reduce the dimensionality
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    word_coordinates = mds.fit_transform(distance_matrix)

    # Plot word embeddings
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(word_list):
        x, y = word_coordinates[i, :]
        plt.scatter(x, y, marker='o', color='b')
        plt.text(x, y, word, fontsize=12, ha='center', va='center')

    plt.title('Word Embedding Distances')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
    
if __name__ == "__main__":  
    word_list = ['apple', 'banana', 'orange', 'grape', 'pineapple']
    word_embeddings = np.array([[0.5, 0.2, 0.1, 0.9],
                                [0.3, 0.4, 0.2, 0.8],
                                [0.6, 0.7, 0.4, 0.1],
                                [0.9, 0.2, 0.5, 0.3],
                                [0.2, 0.8, 0.6, 0.4]])
    
    visualize_word_distances(word_list, word_embeddings)
