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
    # Mirror pixels along the horizontal axis to obtain a mirrored image
    # clockwise rotation
    x = tf.reverse_sequence(x, np.ones(width) * height, 0, batch_dim=1)
    # Mirror pixels along the vertical axis to obtain a global 180 degrees
    # rotation
    x = tf.reverse_sequence(x, np.ones(height) * width, 1, batch_dim=0)
    result = session.run(x)


plt.imshow(result)
plt.show()
