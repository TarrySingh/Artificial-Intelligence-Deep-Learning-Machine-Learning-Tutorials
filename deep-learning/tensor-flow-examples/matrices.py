import tensorflow as tf
import numpy as np
sess = tf.Session()

# identity_matrix = tf.diag([1.0, 1.0, 1.0])
# a = tf.truncated_normal([2, 3])
# b = tf.fill([2,3], 5.0)
# c = tf.random_uniform([3,2])
# d = tf.convert_to_tensor(np.array([[1.,2.,3.], [-3.,-7.,-1.],[0.,5.,-2.]]))
#
# print(sess.run(identity_matrix))


m1 = [[1.0, 2.0],
      [3.0, 4.0]]

m2 = np.array([[1.0, 2.0],
               [3.0, 4.0]], dtype=np.float32)

m3 = tf.constant([[1.0, 2.0],
                  [3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))


t1 = tf.convert_to_tensor(m1, dtype=tf.float32)
t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

print('Type t1 is:', type(t1))
print('Type t2 is:', type(t1))
print('Type t3 is:', type(t1))
