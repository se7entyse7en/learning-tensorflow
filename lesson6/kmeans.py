import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def create_samples(n_clusters, n_samples_per_cluster, n_features,
                   min_value=-100, max_value=100, stddev=5.0):
    centroids = []
    all_samples = []
    mean = (min_value + max_value) / 2
    for i in range(n_clusters):
        # Create centroid
        centroid = tf.random_uniform(
            (1, n_features), minval=min_value, maxval=max_value,
            dtype=tf.float64, name='cluster_centroid_{i}'.format(i=i))
        # Create normally distribute samples
        samples = tf.random_normal(
            (n_samples_per_cluster, n_features), mean=mean, stddev=stddev,
            dtype=tf.float64, name='cluster_samples_{i}'.format(i=i))
        # Make the samples normally distributed around the centroid
        samples += centroid

        centroids.append(centroid)
        all_samples.append(samples)

    # Concat all the centroids
    centroids = tf.concat(0, centroids, name='centroids')
    # Concat all the samples
    all_samples = tf.concat(0, all_samples, name='samples')

    return centroids, all_samples


def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    # Plot out the different clusters
    # Choose a different color for each cluster
    color = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i, centroid in enumerate(centroids):
        # Grab just the samples for the given cluster and plot them out with a
        # new color
        start = i * n_samples_per_cluster
        end = (i+1) * n_samples_per_cluster
        samples = all_samples[start:end]
        plt.scatter(samples[:, 0], samples[:, 1], c=color[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker='x',
                 color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x",
                 color=color[i], mew=5)

    plt.show()
