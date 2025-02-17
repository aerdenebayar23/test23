from google.colab import drive
drive.mount('/content/drive')

import numpy as np

class Linear_Layer:
    def __init__(self, in_dim, out_dim, alpha=0.01, Theta=None, bias=None):
        self.alpha = alpha
        fac = 1  # Example: setting fac to 1
        if Theta is None:
            self.Theta = np.random.randn(in_dim, out_dim) / fac
        else:
            self.Theta = Theta
        if bias is None:
            self.bias = np.random.randn(out_dim) / fac
        else:
            self.bias = bias
        # Initialize grad and grad_bias to zero
        self.grad = np.zeros_like(self.Theta)
        self.grad_bias = np.zeros_like(self.bias)

    def forward_pass(self, X):
        self.X = X
        self.z = np.dot(X, self.Theta) + self.bias  # Linear transformation
        return self.z

    def backprop(self, grad):
        print(f"ReLU Backprop: self.X.shape = {self.X.shape}, grad.shape = {grad.shape}")
        dTheta = np.dot(self.X.T, grad) / self.X.shape[0]
        dbias = np.sum(grad, axis=0) / self.X.shape[0]

        # Update grad and grad_bias instead of directly updating Theta and bias
        self.grad = dTheta
        self.grad_bias = dbias

        # Calculate and return the gradient for the previous layer
        grad_prev = np.dot(grad, self.Theta.T)
        return grad_prev

    def applying_sgd(self):
        self.Theta = self.Theta - (self.alpha * self.grad)
        self.bias = self.bias - (self.alpha * self.grad_bias)
    def count_parameters(model):
        return sum(p.size for p in model.parameters())

import numpy as np

class ConvLayer:
    def __init__(self, num_filters=8, filter_dim=3, stride=1, pad=1, alpha=0.01):
        self.num_filters = num_filters
        self.filter_dim = filter_dim
        self.stride = stride
        self.filter = np.random.randn(self.filter_dim, self.filter_dim)
        self.filter = self.filter / self.filter.sum()  # Normalize filter
        self.bias = np.random.rand() / 10  # Small random bias
        self.pad = pad
        self.alpha = alpha
        self.filters = np.random.randn(num_filters, filter_dim, filter_dim) / np.sqrt(filter_dim * filter_dim)
        self.biases = np.random.randn(num_filters) / 10  # Жижиг санамсаргүй налуу утгууд

    def convolving(self, X, fil):
        """Ганц кернелд convolution хийх"""
        dimen_x = (X.shape[0] - self.filter_dim) // self.stride + 1
        dimen_y = (X.shape[1] - self.filter_dim) // self.stride + 1
        z = np.zeros((dimen_x, dimen_y))

        for i in range(dimen_x):
            for j in range(dimen_y):
                patch = X[i * self.stride:i * self.stride + self.filter_dim,
                          j * self.stride:j * self.stride + self.filter_dim]
                z[i, j] = np.sum(patch * fil)

        return z
    def forward_pass(self, X):
        """Олон кернелээр convolution хийх"""
        self.X = X
        batch_size, height, width = X.shape
        output_height = (height - self.filter_dim) // self.stride + 1
        output_width = (width - self.filter_dim) // self.stride + 1
        self.z = np.zeros((batch_size, self.num_filters, output_height, output_width))

        for b in range(batch_size):
            for f in range(self.num_filters):
                self.z[b, f] = self.convolving(self.X[b], self.filters[f]) + self.biases[f]

        return self.z

    def backprop(self, grad_z):
        """Олон кернелд зориулж backprop хийх"""
        batch_size, num_filters, grad_h, grad_w = grad_z.shape
        self.grads = np.zeros_like(self.X)  # Оролт руу буцаах градиент
        self.grad_filters = np.zeros_like(self.filters)
        self.grad_biases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for f in range(self.num_filters):
                # Гаралтын градиентийг жинтэй convolution хийх
                filter_flipped = np.flip(self.filters[f], axis=(0, 1))
                self.grads[b] += self.convolving(np.pad(grad_z[b, f], ((1, 1), (1, 1)), 'constant'), filter_flipped)

                # Кернелүүдийн градиент тооцоолох
                for i in range(self.filter_dim):
                    for j in range(self.filter_dim):
                        self.grad_filters[f, i, j] += np.sum(
                            grad_z[:, f, :, :] * self.X[:, i:i+grad_h, j:j+grad_w]
                        )

                # Налууны градиент
                self.grad_biases[f] += np.sum(grad_z[:, f, :, :])

        return self.grads

    def applying_sgd(self):
        # Update filter and bias using SGD
        self.filter = self.filter - (self.alpha * self.grad_filter)
        self.bias = self.bias - (self.alpha * self.grad_bias)

    def change_alpha(self):
        # Reduce learning rate by a factor of 10
        self.alpha = self.alpha / 10


class Pooling:
    def __init__(self, pool_dim=2, stride=2):
        self.pool_dim = pool_dim
        self.stride = stride

    def forward_pass(self, data):
        (q, p, t) = data.shape
        z_x = int((p - self.pool_dim) / self.stride) + 1
        z_y = int((t - self.pool_dim) / self.stride) + 1
        after_pool = np.zeros((q, z_x, z_y))
        for ii in range(0, q):
            liss = []
            for i in range(0, p, self.stride):
                for j in range(0, t, self.stride):
                    if (i + self.pool_dim <= p) and (j + self.pool_dim <= t):
                        temp = data[ii, i:i + self.pool_dim, j:j + self.pool_dim]
                        temp_1 = np.max(temp)
                        liss.append(temp_1)
            liss = np.asarray(liss)
            liss = liss.reshape((z_x, z_y))
            after_pool[ii] = liss
            del liss
        return after_pool

    def backprop(self, pooled):
        (a, b, c) = pooled.shape
        cheated = np.zeros((a, 2 * b, 2 * c))
        for k in range(0, a):
            pooled_transpose_re = pooled[k].reshape((b * c))
            count = 0
            for i in range(0, 2 * b, self.stride):
                for j in range(0, 2 * c, self.stride):
                    cheated[k, i:i + self.stride, j:j + self.stride] = pooled_transpose_re[count]
                    count = count + 1
        return cheated

    def applying_sgd(self):
        pass


class softmax:
    def __init__(self):
        pass
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    def forward_pass(self, z):
        self.z =  z
        (p,t) = self.z.shape
        self.a = np.zeros((p,t))
        for i in range(0,p):
            for ii in range(0,t):
                self.a[i,ii] = None #
        return self.a
    def backprop(self, Y):
        y = self.expansion(Y)
        self.grad = (self.a - y)
        return self.grad
    def applying_sgd(self):
        pass

class relu:
    def __init__(self):
        pass
    def forward_pass(self, z):
        if (len(z.shape) == 3):
            z_temp = z.reshape((z.shape[0], z.shape[1]*z.shape[2]))
            z_temp_1 = self.forward_pass(z_temp)
            self.a_1 = z_temp_1.reshape((z.shape[0], z.shape[1], z.shape[2]))
            return (self.a_1)
        else:
            (p,t) = z.shape
            self.a = np.zeros((p,t))
            for i in range(0,p):
                for ii in range(0,t):
                        self.a[i,ii] = max([0,z[i,ii]])
            return self.a
    def derivative(self, a):
        if a>0:
            return 1
        else:
            return 0
    def backprop(self, grad_previous):
        if (len(grad_previous.shape)==3):
            (d, p, t) = grad_previous.shape
            self.grad = np.zeros((d, p, t))
            for i in range(d):
                for ii in range(p):
                    for iii in range(t):
                        self.grad[i, ii, iii] = (grad_previous[i, ii, iii] * self.derivative(self.a_1[i, ii, iii]))
            return (self.grad)
        else:
            (p,t) = grad_previous.shape
            self.grad = np.zeros((p,t))
            for i in range(p):
                for ii in range(t):
                    self.grad[i,ii] = grad_previous[i,ii] * self.derivative(self.a[i,ii])
            return (self.grad)
    def applying_sgd(self):
        pass

class padding():
    def __init__(self, pad = 1):
        self.pad = pad
    def forward_pass(self, data):
        X = np.pad(data , ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),'constant', constant_values=0)
        return X
    def backprop(self, y):
        return (y[:, 1:(y.shape[1]-1),1:(y.shape[2]-1)])
    def applying_sgd(self):
        pass

class reshaping:
    def __init__(self):
        pass
    def forward_pass(self, a):
        self.shape_a = a.shape
        self.final_a = a.reshape(self.shape_a[0], self.shape_a[1]*self.shape_a[2])
        return self.final_a
    def backprop(self, q):
        return (q.reshape(self.shape_a[0], self.shape_a[1], self.shape_a[2]))
    def applying_sgd(self):
        pass

class cross_entropy:
    def __init__(self):
        pass
    def expansion(self, t):
        (a,) = t.shape
        Y = np.zeros((a,10))
        for i in range(0,a):
            Y[i,t[i]] = 1
        return Y
    def loss(self, A, Y):
        exp_Y = self.expansion(Y)
        (u,i) = A.shape
        loss_matrix = np.zeros((u,i))
        for j in range(u):
            for jj in range(i):
                if exp_Y[j,jj] == 0:
                    loss_matrix[j,jj] = np.log(1 - A[j,jj])
                else:
                    loss_matrix[j,jj] = np.log(A[j,jj])
        return ((-(loss_matrix.sum()))/u)

class accuracy:
    def __init__(self):
        pass
    def value(self, out, Y):
        self.out = np.argmax(out, axis=1)
        p = self.out.shape[0]
        total = 0
        for i in range(p):
            if Y[i]==self.out[i]:
                total += 1
        return total/p

class ConvNet:
    def __init__(self, Network):
        self.Network = Network
    def forward_pass(self, X):
        n = X
        for i in self.Network:
            n = i.forward_pass(n)
            # print(n.shape) #
        return n
    def backprop(self, Y):
        m = Y
        count = 1
        for i in (reversed(self.Network)):
            m = i.backprop(m)
    def applying_sgd(self):
        for i in self.Network:
            i.applying_sgd()

from tensorflow.keras.datasets import mnist

(Xtr, Ytr), (Xte, Yte) = mnist.load_data()
X_training = Xtr[:1000,:,:]
Y_training = Ytr[:1000]
X_training = X_training/255
al = 0.3
stopper = 85.0

complete_NN = ConvNet([
                        padding(),
                        ConvLayer(),
                        Pooling(),
                        relu(),
                        padding(),
                        ConvLayer(),
                        Pooling(),
                        relu(),
                        ConvLayer(),
                        relu(),
                        reshaping(),
                        Linear_Layer(5*5, 24, alpha = al),
                        relu(),
                        Linear_Layer(24, 10, alpha = al),
                        softmax()])

CE = cross_entropy()
acc = accuracy()
epochs = 3
broke = 0
batches = 100
for i in range(epochs):
    k = 0
    for j in range(batches, 1001, batches):
        out = complete_NN.forward_pass(X_training[k:j])
        print("epoch:{} \t batch: {} \t loss: \t {}".format(i+1, int(j/batches), CE.loss(out, Y_training[k:j])), end="\t")
        accur = acc.value(out, Y_training[k:j])*100
        print("accuracy: {}".format(accur))
        if accur >= stopper:
            broke = 1
            break
        complete_NN.backprop(Y_training[k:j])
        complete_NN.applying_sgd()
        k = j
    if broke == 1:
        break

out = complete_NN.forward_pass(X_training)
print("The final loss is {}".format(CE.loss(out, Y_training)))
print("The final accuracy on train set is {}".format(acc.value(out, Y_training)*100))

Xtest = Xte/255
out_1 = complete_NN.forward_pass(Xtest)
print("The accuracy on test set is {}".format(acc.value(out_1, Yte)*100))