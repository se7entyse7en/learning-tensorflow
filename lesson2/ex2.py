import numpy as np

import tensorflow as tf

data = np.random.randint(1000, size=10000)
x = tf.constant(data, name='x')
y = tf.Variable(x + 5, name='y')

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))
