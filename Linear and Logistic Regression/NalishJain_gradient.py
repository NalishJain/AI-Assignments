import numpy as np

# Generate data
X = np.arange(-20, 20, 0.1)
np.random.shuffle(X)
eps = np.random.rand(400) * 10
y = 23 * X + 43 + eps
bs = np.ones((400,1))
X = X.reshape((400,1))
y = y.reshape((400, 1))
X = np.concatenate((bs, X), axis=1)

w = np.random.rand(2)


# # Learning rate
learning_rate = 0.0001

# Number of iterations
iterations = 100

for i in range(iterations):
    for j in range(400):
        y_pred = np.matmul(X[j], w)
        w -= learning_rate*(y_pred - y[j])*(X[j].T)

print(f'w = {np.round(w[1], 0)}, b = {np.round(w[0], 0)}')