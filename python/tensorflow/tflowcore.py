#!/usr/sbin/python

import tensorflow as tf

### Step 1: Build computational graph

# Constant Tensors
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

# Session is needed to run a computational graph of tensors
sess = tf.Session()
print(sess.run([node1, node2]))

# More complicated computations combine operation nodes with tensor nodes
node3 = tf.add(node1, node2)
print("node3: ", node3)
print(sess.run(node3))

# Parameterizing graphs allow external inputs known as 'placeholders'
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # same as adder_node = tf.add(a, b)

# Can use parameterized node similar to a function
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1.2,3], b: [2,4]}))

# Can make new complex node using other node
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Now to machine learning
# Variables are not initialized until specifically initialized
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

# Our first lil model
print(sess.run(linear_model, {x:[1,2,3,4]}))

# Now lets evaluate if the model is any good
# This needs 2 things to do this: 'y' placeholder with training data, and a loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# loss(cost) function: + = loss higher is worse, 0 = perfect
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1,2,3,4], y:[0,-1,-2,-3]}))

# This would be perfect if we change the variables W=-1 and b=1
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


# Use tf.train API
# Optimizers slowly change each variable in order to minimize loss.  Simplest one is 'gradient descent'
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[-0,-1,-2,-3]})

print(sess.run([W, b]))