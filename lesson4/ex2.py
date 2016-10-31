import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf

filename = 'MarshOrchid.jpg'
raw_image = mpimg.imread(filename)

image = tf.placeholder('uint8', [None, None, 3])

# Calculate image shape
shape = tf.shape(image)
height = shape[0]
width = shape[1]
depth = shape[2]

# Just ignore if not divisible by 2
half_height = tf.div(height, 2)
half_width = tf.div(width, 2)
slices_size = [half_height, half_width, -1]

# Split images in 4 corners
slice_top_left = tf.slice(image, [0, 0, 0], slices_size)
slice_top_right = tf.slice(image, [0, half_width, 0], slices_size)
slice_bottom_left = tf.slice(image, [half_height, 0, 0], slices_size)
slice_bottom_right = tf.slice(image, [half_height, half_width, 0], slices_size)

# Paste the 4 corners again
pasted_top = tf.concat(1, [slice_top_left, slice_top_right])
pasted_bottom = tf.concat(1, [slice_bottom_left, slice_bottom_right])
pasted = tf.concat(0, [pasted_top, pasted_bottom])

with tf.Session() as session:
    result = session.run(pasted, feed_dict={image: raw_image})

plt.imshow(result)
plt.show()
