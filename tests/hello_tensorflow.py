# TensorFlow=1.x
import tensorflow as tf


# Create a Tensor.
hello = tf.constant("hello tensorflow")
print(hello)

# To access a Tensor value, call numpy().
print(hello.numpy())
