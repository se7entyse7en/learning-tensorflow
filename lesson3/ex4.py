import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

filename = 'MarshOrchid.jpg'
image = mpimg.imread(filename)

x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    shape = tf.shape(x)
    height, width, depth = session.run(shape)
    # Slice the left half
    left_sliced = tf.slice(x, [0, 0, 0], [height, int(width / 2), depth])
    # Mirror pixels along the vertical axis of the left half
    left_mirrored_sliced = tf.reverse_sequence(
        left_sliced, np.ones(height) * int(width / 2), 1, batch_dim=0)
    # Paste the two slices to obtain the left half mirrored on the right half
    pasted = tf.concat(1, [left_sliced, left_mirrored_sliced])
    result = session.run(pasted)


plt.imshow(result)
plt.show()
