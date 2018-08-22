import tensorflow as tf

hello = tf.constant('test')
session = tf.Session()
print(session.run(hello))
