import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf

filename = 'MarshOrchid.jpg'
raw_image = mpimg.imread(filename)

image = tf.placeholder(tf.float64, [None, None, 3])

# Calculate image shape
shape = tf.shape(image)
height = shape[0]
width = shape[1]
depth = shape[2]

# Calculate the mean RGB value for each pixel
gray_scaled = tf.reduce_mean(image, 2, keep_dims=True)

# Adjust shape from (heigth, width, 1) to (heigth, width, depth) using
# broadasting
gray_scaled = gray_scaled * tf.ones((height, width, depth), dtype=tf.float64)

# Adjust dtype for imshow
gray_scaled = tf.cast(gray_scaled, tf.uint8)

with tf.Session() as session:
    result = session.run(gray_scaled, feed_dict={image: raw_image})

plt.imshow(result)
plt.show()
