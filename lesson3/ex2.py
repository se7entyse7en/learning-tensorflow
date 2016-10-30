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
    # Rotate 90 degrees counter-clockwise
    x = tf.transpose(x, perm=[1, 0, 2])
    # Mirror pixels along the vertical axis to obtain a global 90 degrees
    # clockwise rotation
    x = tf.reverse_sequence(x, np.ones(width) * height, 1, batch_dim=0)
    result = session.run(x)


plt.imshow(result)
plt.show()
