import tensorflow as tf
from kmeans import create_samples, plot_clusters

if __name__ == '__main__':
    n_clusters = 5
    n_samples_per_cluster = 500
    n_features = 2
    seed = 0

    # Set seed for reproducibility
    tf.set_random_seed(seed)
    centroids, all_samples = create_samples(
        n_clusters, n_samples_per_cluster, n_features)

    model = tf.initialize_all_variables()
    with tf.Session() as session:
        centroids, all_samples = session.run([centroids, all_samples])

    plot_clusters(all_samples, centroids, n_samples_per_cluster)
