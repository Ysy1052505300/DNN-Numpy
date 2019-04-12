import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        size = in_features * out_features
        w = np.random.normal(0, 0.01, size).reshape(in_features, out_features)
        b = np.zeros((1, out_features))
        self.w = np.concatenate([w, b], axis=0)
        self.input = None
        self.output = None

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        # print(self.w.shape, "w size")
        self.input = x
        ones = np.ones((1, x.shape[1]))
        x = np.concatenate([x, ones], axis=0)
        self.output = np.dot(self.w.T, x)
        # print('forward', self.output.shape)
        return self.output

    def backward(self, dout, i=None):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        d_pre, n = self.input.shape
        d_aft, n = dout.shape
        temp_d = np.zeros((n, d_pre + 1, d_aft))
        input_T = self.input.T
        dout_T = dout.T
        for i in range(n):
            grad1 = input_T[i].reshape((d_pre, 1))
            grad2 = dout_T[i].reshape((1, d_aft))
            w = np.dot(grad1, grad2)
            b = grad2
            temp_d[i] = np.concatenate([w, b], axis=0)
        # print("backward", temp_d.shape)
        return temp_d




class ReLU(object):
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.input = x
        output = x.copy()
        row, column = output.shape
        for i in range(row):
            for j in range(column):
                if output[i][j] < 0:
                    output[i][j] = 0
        self.output = output
        # print('forward', self.output.shape)
        return output

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        d, n = self.output.shape
        temp_d = np.zeros((d, n))
        for i in range(n):
            for j in range(d):
                if self.output[j][i] > 0:
                    temp_d[j][i] = 1
        d_pre, n = temp_d.shape
        d_aft, n = dout.shape
        tempd_T = temp_d.T
        dout_T = dout.T
        dx = np.zeros((n, d_pre))
        for i in range(n):
            grad1 = tempd_T[i].reshape((d_pre, 1))
            grad2 = dout_T[i].reshape((1, d_aft))
            dx[i] = np.sum(np.dot(grad1, grad2), axis=1)
        # print('backward', dx.T.shape)
        return dx.T


class SoftMax(object):
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        self.input = x
        xT = x.T
        output = np.zeros(x.shape)
        d, n = output.shape
        for i in range(n):
            b = xT[i].max()
            exp = np.exp(xT[i] - b)
            sum = np.sum(exp)
            for j in range(d):
                output[j][i] = exp[j]/sum
        self.output = output
        # print('forward', self.output.shape)
        return output

    def backward(self, dout, index):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        d, n = self.output.shape
        temp_d = np.zeros((d, n))
        for i in range(n):
            for j in range(d):
                if index[0][i] == j:
                    temp_d[j][i] = self.output[j][i] * (1 - self.output[j][i])
                else:
                    temp_d[j][i] = - self.output[j][i] * self.output[j][i]
        dx = np.zeros((n, d))
        for i in range(n):
            dx[i] = temp_d.T[i] * dout[0][i]
        # print('backward', dx.T.shape)
        return dx.T


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        d, n = x.shape
        output = np.zeros((1, n))
        for i in range(n):
            temp = 0
            for j in range(d):
                temp = temp + y[j][i] * np.log(x[j][i])
            output[0][i] = -temp
        # print('forward', output.shape)
        return output

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        d, n = x.shape
        dx = np.zeros((1, n))
        index = np.zeros((1, n))
        for i in range(n):
            for j in range(d):
                if y[j][i] == 1:
                    dx[0][i] = - 1 / (x[j][i])
                    index[0][i] = j
        # print('backward', dx.shape)
        return dx, index
