import numpy as np
import matplotlib.pyplot as plt

# load data
def load_data(filename):
    data = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split(',')
        col_num = len(lineArr)
        temp = []
        for i in range(col_num):
            temp.append(int(lineArr[i]))
        data.append(temp)
    return np.array(data)

data = load_data('ex1data2.txt')
print(data.shape)
print(data[:5])

X = data[:,:-1]
y = data[:,-1:]
print(X.shape)
print(y.shape)
print(X[:5])
print(y[:5])

# 定义特征缩放函数，因为每个特征取值不同，并且差别很大
def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1,X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


X_norm, mu, sigma = featureNormalize(X)
num_train = X_norm.shape[0]
one = np.ones((num_train,1))
X = np.hstack((one, X_norm))
W = np.zeros((X.shape[1], 1))

# compute cost
def compute_cost(X_test, y_test, theta):
    num_X = X_test.shape[0]
    cost = 0.5 * np.sum(np.square(X_test.dot(theta) - y_test)) / num_X
    return cost
# 计算梯度下降
def gradient_descent(X_test, y_test, theta, alpha=0.005, iter=1500):
    J_history = []
    num_X = X_test.shape[0]
    for i in range(iter):
        theta = theta - alpha * X_test.T.dot(X_test.dot(theta)-y_test)/num_X
        cost = compute_cost(X_test, y_test, theta)
        J_history.append(cost)
    return theta, J_history
# test
print('run gradient descent')
theta, J_history = gradient_descent(X, y, W)
print('Theta computed from gradient descent:\n', theta)

# 绘制cost曲线,可以调节
plt.plot(J_history, color='b')
plt.xlabel('iters')
plt.ylabel('j_cost')
plt.title('cost variety')
plt.show()

# 预测
x_t = ([[1650,3]] - mu) / sigma
X_test = np.hstack((np.ones((1,1)),x_t))
predict = X_test.dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent)')
print(predict)

#直接使用公式求解最佳theta,不用梯度下降
XX = data[:,:-1]
yy = data[:,-1:]
m = XX.shape[0]

one = np.ones((m,1))
XX = np.hstack((one,data[:,:-1]))
def normalEquation(X_train, y_train):
    w = np.zeros((X_train.shape[0],1))
    w = ((np.linalg.pinv(X_train.T.dot(X_train))).dot(X_train.T)).dot(y_train)
    return w
w = normalEquation(XX, yy)
print(w)
x_t = [[1650,3]]
X_test = np.hstack((np.ones((1,1)),x_t))
predict = X_test.dot(w)

print(predict)


