import tensorflow as tf
from kmeans import choose_random_centroids, create_samples, plot_clusters

if __name__ == '__main__':
    n_clusters = 2
    n_samples_per_cluster = 2
    n_features = 2
    seed = 0

    # Set seed for reproducibility
    tf.set_random_seed(seed)

    # Create samples
    centroids, all_samples = create_samples(
        n_clusters, n_samples_per_cluster, n_features)
    # Initial centroids guess
    initial_centroids = choose_random_centroids(all_samples, n_clusters)

    with tf.Session() as session:
        centroids, all_samples, initial_centroids = session.run(
            [centroids, all_samples, initial_centroids])

    print(all_samples)
    print(initial_centroids)

    # plot_clusters(all_samples, centroids, n_samples_per_cluster)
