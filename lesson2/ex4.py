import numpy as np

import tensorflow as tf

x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    for i in range(5):
        session.run(model)
        x = x + np.random.randint(1000)
        average = x / (i + 1)
        print(session.run(average))
