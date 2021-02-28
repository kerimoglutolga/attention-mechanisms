import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# attention pooling: nadayara-watson kernel regression

# generate the dataset from the following function:
# y_i = 2 * sin(x_i) + x_i ** 0.8 + epsilon (with epsilo being random noise)

n = 50
x_train = tf.sort(tf.random.uniform((n,), 0, 5))

def f(x):
    return 2 * tf.math.sin(x)  + x

y_train = f(x_train) + tf.random.normal(shape=(n,), mean=0.0, stddev=0.1)
x_test = tf.range(0, 5, 0.1, dtype=tf.float32)
y_test = f(x_test)

# an example with average pooling as our attention mechanism
y_hat = tf.repeat(tf.reduce_sum(y_train) / n, len(x_test))


plt.plot(x_test, y_hat, label = "pred")
plt.plot(x_test, y_test, label="truth", color="black")
plt.scatter(x_train, y_train, label =  "training data", color = "orange")
plt.legend()
plt.title("Average Pooling")
plt.show()
# nonparametric attention pooling with a gaussian kernel
X_repeat = tf.reshape(tf.repeat(x_test, n), (-1, n))
attention_weights = tf.nn.softmax(-(X_repeat - x_train)** 2 / 2)
y_hat = tf.linalg.matvec(attention_weights, y_train)

plt.title("Nonparametric Attention Pooling")
plt.plot(x_test, y_hat, label = "pred")
plt.plot(x_test, y_test, label="truth", color="black")
plt.scatter(x_train, y_train, label =  "training-data", color = "orange")
plt.legend()
plt.title("Nonparametric Attention Pooling")
plt.show()

# parametric attention pooling
class NWKernelRegression(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(tf.random.uniform((1,), 0, 1))

    def __call__(self, queries, keys, values):
         queries = tf.reshape(tf.repeat(queries, keys.shape[0]), (-1, keys.shape[0]))
         self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w) ** 2 / 2)
         return tf.linalg.matvec(self.attention_weights, values)


net = NWKernelRegression()
optimizer = tf.keras.optimizers.SGD(lr=0.5)
losses = []
for epoch in range(100):
    with tf.GradientTape() as tape:
        loss = tf.keras.losses.MSE(y_test, net(x_train, x_train, y_train))
        params = net.trainable_variables
    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(grads, params))
    losses.append(loss.numpy())
y_hat = net(x_test, x_train, y_train)
plt.title("Parametric Attention Pooling")
plt.plot(x_test, y_hat, label = "pred")
plt.plot(x_test, y_test, label="truth", color="black")
plt.scatter(x_train, y_train, label =  "training-data", color = "orange")
plt.legend()
plt.show()
