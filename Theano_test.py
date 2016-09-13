"""
    use theano to train the neural net work
"""
'''
# about shared varible
x = T.iscalar('x')
sh = shared(0)
f = function([x], sh**2, updates=[(sh, sh+x)])

# about calculating gradient
x = T.dscalar('x')
y = x**3
qy = T.grad(y, x)
f = function([x], qy)
f(4)
print(pp(qy))


# about a neuron forward propagatin

x = T.vector('x')
#w = T.vector('w')
#b = T.dscalar('b')

w = theano.shared(np.array([1, 1]))
b = theano.shared(-1.5)

z = T.dot(w, x) + b
a = ifelse(T.lt(z, 0), 0, 1)

#neuron = theano.function([x, w, b], a)  #todo TypeError: Outputs must be theano Variable or Out instances.
                                        #todo module object is not callable

neuron = theano.function([x], a)

# test

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

#weights = [1, 1]
#bias = -1.5

for i in range(len(inputs)):
    t = inputs[i]
    #out = neuron(t, w, b)
    out = neuron(t)
    print(out)

# back propagation

x = T.matrix('x')
w = theano.shared(np.array([random(), random()]))

#print(w.get_value())

b = theano.shared(1.0)

learning_rate = 0.01
z = T.dot(x, w) + b
a = 1/(1+T.exp(-z))

a_hat = T.vector('a_hat')  # the supposed output

cost = -(a_hat*T.log(a) + (1-a_hat)*T.log(1-a)).sum()
# about cost function there is a question in it

dw, db = T.grad(cost, [w, b]) # derivative w and derivative b

#full_batch gradient descent
train = function(
    inputs=[x, a_hat],
    outpus=[a, cost],
    updates=[
        [w, w-learning_rate*dw],
        [b, b-learning_rate*db]
    ]
)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [0, 0, 0, 1]

cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
'''
'''

# use theano to realize the forward propagation

def sigmoid(z):
    return 1/(1+np.exp(-z))

def convertToMatrix(x):  # x is a list convert this list to a matrix
    mx = np.zeros(len(x), 1)
    for i in range(len(x)):
        mx[i, 0] = x[i]
    return mx

sizes = [4, 5, 6]
w = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
weights = theano.shared(w)
b = [np.random.randn(y, 1) for y in sizes[1:]]
biases = theano.shared(b)
#print(weights.get_value())
weight = weights.get_value()
bias = biases.get_value()

x = T.vector('x')
a = convertToMatrix(x)
for w, b in weight, bias:
    z = np.dot(w, a) + b
    a = sigmoid(z)

f = function(
    inputs=[x],
    outputs=[a]
)

# cautious here

for w, b in weight, bias:
    z = np.dot() + b
for singlew in weight:
    print(singlew)
    print(singlew.shape)

for singlebias in bias:
    print(singlebias)
    print(singlebias.shape)

class neural_net_work:
    def __init__(self, sizes):
        self.weights = [np.random.randn(y, x)
                        for y, x in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(y ,1)
                       for y in sizes[1:] ]
        weights = theano.shared(self.weights)
        biases = theano.shared(self.biases)  # useless ?
        self.layernum = len(sizes)
'''
import numpy as np
import theano.tensor as T
from theano import function
from theano import pp
from theano import shared
import theano
from theano.ifelse import ifelse
from random import random
import load_data
import timeit

class BP:
    """
    in this class of bp net work the parameter is not a list
    the weights and biases is sepreated.
    and the input vector come from load_data don't need to be handled
    and the return value is a function of the function is used
    :parameter sizes the shape of sizes should be 1*3 and it is a list
    """
    def __init__(self, sizes):
        self.layers = len(sizes)
        weights1 = np.random.randn(sizes[0], sizes[1])
        weights2 = np.random.randn(sizes[1], sizes[2])
        biases1 = np.random.randn(1, sizes[1])
        biases2 = np.random.randn(1, sizes[2])

        self.weights1 = theano.shared(name='weights1', value=weights1.astype(theano.config.floatX))
        self.weights2 = theano.shared(name='weights2', value=weights2.astype(theano.config.floatX))
        self.biases1 = theano.shared(name='biases1', value=biases1.astype(theano.config.floatX))
        self.biases2 = theano.shared(name='biases2', value=biases2.astype(theano.config.floatX))

    def forwardprop(self):
        """
        this function has no parameter
        :return: a theano function that could output the prediction
         according to the input into this function
        """
        x = T.vector('x')
        hide = T.nnet.sigmoid(T.dot(x, self.weights1) + self.biases1)
        output = T.nnet.sigmoid(T.dot(hide, self.weights2) + self.biases2)
        prediction = T.argmax(output)
        predict = function(inputs=[x], outputs=prediction)
        return predict

    def backprop(self):
        """
        this function has no parameter
        :return: a theano function that could output the derivative of weights and biases
        """
        x = T.vector('x')
        y = T.vector('y')

        output = self.forwardprop().predict(x)

        error = y * T.log(output) - (1. - y) * (1. - T.log(1. - output))

        dw1 = T.grad(error, self.weights1)
        dw2 = T.grad(error, self.weights2)
        db1 = T.grad(error, self.biases1)
        db2 = T.grad(error, self.biases2)

        get_grad = function(inputs=[x, y], outputs=[dw1, dw2, db1, db2])
        return get_grad

    def updates(self, mini_batch, eta):
        n = len(mini_batch)
        delta_w1 = np.zeros(self.weights1.get_value().shape)
        delta_w2 = np.zeros(self.weights2.get_value().shape)
        delta_b1 = np.zeros(self.biases1.get_value().shape)
        delta_b2 = np.zeros(self.biases2.get_value().shape)

        for x, y in mini_batch:
            get_grad = self.backprop()
            dw1, dw2, db1, db2 = get_grad(x, y)
            delta_w1 += dw1
            delta_w2 += dw2
            delta_b1 += db1
            delta_b2 += db2

        temp_w1 = self.weights1.get_value() - (eta * 1.0 / n) * delta_w1
        temp_w2 = self.weights2.get_value() - (eta * 1.0 / n) * delta_w2
        temp_b1 = self.biases1.get_value() - (eta * 1.0 / n) * delta_b1
        temp_b2 = self.biases2.get_value() - (eta * 1.0 / n) * delta_b2

        self.weights1.set_value(temp_w1)
        self.weights2.set_value(temp_w2)
        self.biases1.set_value(temp_b1)
        self.biases2.set_value(temp_b2)

    def SGD(self, training_data, mini_batch_size, epoch, eta):
        n = len(training_data)
        for i in range(epoch):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updates(mini_batch, eta)

    def evaluate(self, test_data):
        error = 0.0
        for x, y in test_data:
            predict = self.forwardprop()
            output = predict(x)
            real_res = np.argmax(y)
            if output != real_res:
                error += 1.0
            print('output: {}  real num: {}'.format(output, real_res))
        return 1.0 - error/len(test_data)



'''
def convert(x):
    pass
sizes = [3, 4, 2]
x = T.vector('x')
y = T.vector('y')
weights1 = np.random.randn(sizes[0], sizes[1])
weights2 = np.random.randn(sizes[1], sizes[2])
biases1 = np.random.randn(1, sizes[1])
biases2 = np.random.randn(1, sizes[2])
# weights and biases should be shared variable
#res = T.dot(x, weights) + biases
hide = T.dot(x, weights1) + biases1  # get [[]]
res = T.dot(hide, weights2) + biases2
sub = res - y
#get_res = function(inputs=[x], outputs=hide)
#get_res = function([x], res)# and get the result
get_res = function(inputs=[x, y], outputs=[res, sub]) # and sub is alright
r, s = get_res([1, 2, 3], [1, 1])
#r = get_res((1, 2, 3))  # tuple or list is alright
print(r)
print(s)
print(r.shape)
print(s.shape)
'''
'''
 when training this bp net work, the parameters weights and
 biases should be stored into the pickle file
'''
import load_data

class bp_network:
    """
    compared with the class BP ,the bp_network is more extensible
    and more easy to train the neural net work
    cause the layer number can be changed by input of the
    different length of the parameters
    the length of the list can be also used as a parameter
    when training the neural net work, the parameters weights and biases need
     to be loaded into a pickle file for the next training, so it can accelerate
     the training speed.
    """
    def __init__(self, sizes):
        self.layers = len(sizes)
        weights = [np.random.randn(x, y)
                   for x, y in zip(sizes[:-1], sizes[1:])]
        biases = [np.random.randn(1, y)
                  for y in sizes[1:]]

        self.weights = [theano.shared(w)#, value=w.astype(theano.config.floatX))
                        for w in weights] # backtrace when the node is created
        self.biases = [theano.shared(b)#, value=b.astype(theano.config.floatX))
                       for b in biases]
        # add the phrase value=w.astype(theano.config.floatX) killed an error

    def forwardprop(self):  # this function has no problem
        """
        :return: predict funcion
        """
        x = T.vector('x')
        a = x
        for w, b in zip(self.weights, self.biases):
            a = T.nnet.sigmoid(T.dot(a, w) + b)
        prediction = a
        predict = function(inputs=[x], outputs=prediction)
        return predict

    def get_weights(self, x, y):
        l = [None]*len(y)
        for i in range(0, len(y)):
            l[i] = 1.0
        l = np.array(l)
        #x = T.vector('x')
        #y = T.vector('y')
        predict = self.forwardprop()
        #print('ok')
        res = predict(x)
        #r = np.argmax(res)
        #print('r'.format(r))
        print(res)
        print(l-res)
        error = T.sum(y * T.log(res) - (l - y) * T.log(l - res))
        print(error)
        #dw = []
        #for i in range(0, self.layers-1):
        #    dw.append(T.grad(error, self.weights[i])) # TODO:
        #weights = self.weights.get_value()
        #dw = T.grad(error, self.weights[0])
        #dw = T.grad(error, self.weights[0].get_value())
        self.weights[0].get_value()
        print('OK')
        dw = T.grad(cost=error, wrt=self.weights[0].get_value()) # error occured here
        #dw = [T.grad(error, w) for w in self.weights]
        get_dw = function(inputs=[x, y], outputs=dw)
        return get_dw

    def get_biases(self, x, y):
        l = [None]*len(y)
        for i in range(0, len(y)):
            l[i] = 1.0
        l = np.array(l)
        #x = T.vector('x')
        #y = T.vector('y')
        predict = self.forwardprop()
        res = predict(x)
        error = T.sum(y * T.log(res) - (l - y) * T.log(l - res))
        db = [T.grad(error, b) for b in self.biases]
        # TODO: grad method it used only by non-differentiable operator float64 or matrix
        # how to vectorize the output
        get_db = function(inputs=[x, y], outputs=db)
        # can't it be calculated in batch ?
        return get_db

    def update(self, mini_batch, eta):
        delta_w = [np.zeros(w.get_value().shape) # todo the differences with w.shape
                   for w in self.weights]
        delta_b = [np.zeros(b.get_value().shape)
                   for b in self.biases]

        for x, y in mini_batch:
            get_weights = self.get_weights(x, y)
            get_biases = self.get_biases(x, y)
            derivative_w, derivative_b = get_weights(x, y), get_biases(x, y)
            delta_w = [deltaw+derivativew
                       for deltaw, derivativew in zip(delta_w, derivative_w)]
            delta_b = [deltab+derivativeb
                       for deltab, derivativeb in zip(delta_b, derivative_b)]

        tempw = [w - (eta*1.0/len(mini_batch)) * dw
                 for w, dw in zip(self.weights, delta_w)]
        tempb = [b - (eta*1.0/len(mini_batch)) * db
                 for b, db in zip(self.biases, delta_b)]

        self.weights = [w.set_value(tw)
                        for w, tw in zip(self.weights, tempw)]
        self.biases = [b.set_value(tb)
                       for b, tb in zip(self.biases, tempb)]
        print('a mini_batch updte over')

    def SGD(self, training_data, mini_batch_size, epoch, eta):
        n = len(training_data)
        for i in range(epoch):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
            print('SGD {} over'.format(i))

    def evaluate(self, test_data):
        accuracy = 0.0
        error = 0
        for x, y in test_data:
            predict = self.forwardprop()
            output = predict(x)
            output_res = np.argmax(output)
            real_res = np.argmax(y)
            print('output:{}'.format(output_res))
            print('real num:{}'.format(real_res))
            if output_res != real_res:
                error += 1
        accuracy = 1.0 - error*1.0/len(test_data)
        return accuracy
# test

#bp = bp_network([784, 30, 10])
#training_data = load_data.get_training_data(10)
#bp.SGD(training_data, 10, 1, 0.3)

# involve all the function above into one, maybe will make it right

class a_simple_test:
    def __init__(self, sizes):
        param = [p for p in sizes]
        self.layers = len(sizes)
        self.param = [theano.shared(p)
                      for p in param]

    def a_additional_function(self):
        pass

    def test(self):
        x = T.dscalar('x')
        #y = T.dscalar('y')
        #z = T.dscalar('z')
        res = 1.0
        for i in range(0, self.layers):
            res *= self.param[i]
        res *= x
        dp = [T.grad(res, p) for p in self.param]
        f = function(inputs=[x], outputs=dp)
        print('ok')
        return f

a_test = a_simple_test([30, 10, 5])
f = a_test.test()
print(f(3))
































