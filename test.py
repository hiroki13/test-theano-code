__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T


def main():
    import test_sequence
    conv()


def sum():
    e = np.asarray([[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]], dtype=theano.config.floatX)
    w = T.fmatrix('w')
    y = T.sum(w)
    f = theano.function(inputs=[w], outputs=y)
    print f(e)


def extract_with_index():
    def forward(b_t, b_tm1, c_emb):
        c_tmp = c_emb[b_tm1: b_t]
        return T.max(c_tmp, axis=0), b_t

    e = np.asarray([[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]], dtype=theano.config.floatX)
    g = np.asarray([1, 2, 6], dtype='int32')

    w = T.fmatrix('w')
    i = T.iscalar('i')
    j = T.iscalar('j')
    k = T.ivector('k')

    y = w[i: j]
#    f = theano.function(inputs=[w, i, j], outputs=y)
#    print f(e, 0, 2)

    [c, _], _ = theano.scan(fn=forward,
                            sequences=[k],
                            outputs_info=[None, T.cast(0, 'int32')],
                            non_sequences=[w])
    f = theano.function(inputs=[w, k], outputs=c)
    print f(e, g)


def conv():
    from theano.tensor.nnet.conv import conv2d
#    from theano.tensor.signal.conv import conv2d
#    e = np.asarray([[[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]]], dtype=theano.config.floatX)
#    e = np.asarray([[[[1., 2.]], [[3., 4.]], [[5, 6]], [[5., 6.]], [[7., 8.]], [[9, 10]]]], dtype=theano.config.floatX)
#    e = np.asarray([[[[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]]]], dtype=theano.config.floatX)
#    e = np.asarray([[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]], dtype=theano.config.floatX)

#    e = np.asarray([[1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 9, 10]], dtype=theano.config.floatX)
#    e = np.asarray([[1, 1, 1, 1, 1], [1, 1, 2, 1, 1]], dtype=theano.config.floatX)
    e = np.ones(shape=(1, 1, 5, 10), dtype=theano.config.floatX)

#    x = np.asarray([[1, 1], [1, 1], [1, 1]], dtype=theano.config.floatX)
#    x = np.asarray([[[1], [1]], [[2], [1]]], dtype=theano.config.floatX)
#    x = np.asarray([[1, 1, 1], [2, 1, 1]], dtype=theano.config.floatX)  # 1D: 2, 2D: 3
    x = np.ones(shape=(6, 1, 1, 10), dtype=theano.config.floatX)  # 1D: resulting emb dim, 2D: char emb dim * window
#    x = np.asarray([[[1, 1], [1, 1], [1, 1]], [[1, 2], [1, 1], [1, 1]]], dtype=theano.config.floatX)
#    x = np.asarray([[[1, 1]], [[2, 1]]], dtype=theano.config.floatX)
    zero = theano.shared(np.zeros(shape=(1, 2), dtype=theano.config.floatX))

    w = T.ftensor4('w')
    v = T.ftensor4('v')
#    w = T.fmatrix('w')
#    v = T.ftensor3('v')
#    v = T.fmatrix('v')

#    u = T.concatenate([zero, w, zero], axis=1)
    y = conv2d(input=w, filters=v)
    y = y.reshape((y.shape[1], y.shape[2]))
#    y = conv2d(input=w, filters=v)
#    y = conv2d(input=u.T, filters=v.dimshuffle(0, 1, 'x'), subsample=(3, 1))
#    y = T.max(c.reshape((c.shape[0], c.shape[1])), axis=1)
#    y = c.reshape((c.shape[0], c.shape[1]))
#    y = u.T
    f = theano.function(inputs=[w, v], outputs=y)
#    f = theano.function(inputs=[w], outputs=w.T)

    print f(e, x)
#    print f(e)


def conv_batch():
    from theano.tensor.signal.conv import conv2d
    e = np.asarray([[[1., 2.], [3., 4.], [5, 6]], [[5., 6.], [7., 8.], [9, 10]]], dtype=theano.config.floatX)
#    x = np.asarray([[1., 2.], [3., 1.]], dtype=theano.config.floatX)
    x = np.asarray([[1, 2], [1, 1]], dtype=theano.config.floatX)

    w = T.ftensor3('w')
    v = T.fmatrix('v')

    y = conv2d(input=w, filters=v)
    f = theano.function(inputs=[w, v], outputs=y)

    print f(e, x)


def sigmoid():
    x = T.fvector()
    y = T.mean(T.log(T.nnet.sigmoid(x)))
    f = theano.function(inputs=[x], outputs=y)
    print f([10000., 1000.])


def mean():
    e = np.asarray([[[2, 4], [5, 1]], [[3, 5], [4, 6]]], dtype='float32')
    w = T.ftensor3('w')

#    y = T.mean(w, axis=[1, 2], keepdims=True)
    y = T.argmax(w, axis=2)
    f = theano.function(inputs=[w], outputs=y)

    print f(e)
#    print f([[[2, 4], [2, 1]], [1, 2, 3]])


def multiply():
    e1 = np.asarray([2, 4], dtype='int32')
    e2 = np.asarray([[1, 2]], dtype='int32')
    w = T.ivector('v1')
    v = T.imatrix('v2')

    y = v * w
    f = theano.function(inputs=[v, w], outputs=y)

    print f(e2, e1)


def repeat():
    e = np.asarray([[[2, 4], [5, 1]]], dtype='int32')
    w = T.itensor3('w')

    y = T.repeat(w, T.cast(w.shape[1], dtype='int32'), 0)
    f = theano.function(inputs=[w], outputs=y)

    print f(e)


def tensor_max():
    e = np.asarray([[[2, 4], [5, 1]], [[3, 5], [4, 6]]], dtype='int32')
    w = T.itensor3('w')

    y = T.repeat(T.max(w, axis=1, keepdims=True), 3, 1)
    f = theano.function(inputs=[w], outputs=y)

    print f(e)


def add():
    e1 = np.asarray([[2, 4], [5, 1], [3, 5], [4, 6]], dtype='int32')
#    e2 = np.asarray([[100, 100], [10, 10]], dtype='int32')
    e2 = np.asarray([1, 2], dtype='int32')
    w = T.imatrix('w')
#    v1 = T.imatrix('v1')
    v1 = T.ivector('v1')

#    y = T.max_and_argmax(w, 1)
    y = v1 * T.ones_like(w)
    f = theano.function(inputs=[w, v1], outputs=y)

    print f(e1, e2)


def dot():
    e1 = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='int32')
    e2 = np.asarray([[1, 2], [3, 1]], dtype='int32')
    w = T.itensor3('w')
    v = T.imatrix('v')

    y = T.batched_dot(v, w.dimshuffle(0, 2, 1))
    u = w.T
    f = theano.function(inputs=[v, w], outputs=y)
    f2 = theano.function(inputs=[w], outputs=u)

    print f(e2, e1)
#    print f2(e1)


def extract_with_matrix():
    e1 = np.asarray([[2, 4], [7, 1], [3, 5], [4, 6]], dtype='int32')
    e2 = np.asarray([1, 3], dtype='int32')
    e3 = np.asarray([0, 1, 0], dtype='int32')
    w = T.imatrix('w')
    v1 = T.ivector('v1')
    v2 = T.imatrix('v2')

#    y = w[0][v1]
    y = T.sum(w, 1)
    f = theano.function(inputs=[w], outputs=y)

    print f(e1, [[1, 0, 1], [1, 1, 0], [0, 1, 0]])


def test_logsumexp():
    E = np.asarray([[2, 4], [2, 1], [3, 5], [4, 2]], dtype='float32')
    E2 = np.asarray([[2, 4, 3], [2, 1, 5], [100, 5, 3], [4, 2, 6]], dtype='float32')
    W = T.fmatrix('W')
    y = logsumexp(W, 1)
    f = theano.function(inputs=[W], outputs=[y])

    yv = logsumexp_v(y)
    fv = theano.function(inputs=[W], outputs=[yv])

    y2 = T.log(T.sum(T.exp(W), 1))
    f2 = theano.function(inputs=[W], outputs=[y2])

    print 'f',
    print f(E)
    print 'f',
    print f(E2)
    print 'fv',
    print fv(E)
    print 'fv',
    print fv(E2)
#    print 'f2',
#    print f2(E)
#    print 'f2',
#    print f2(E2)


def logsumexp_v(a):
    b = T.max(a)
    return b + T.log(T.sum(T.exp(a-b)))


def logsumexp(x, axis=None):
    x_max = T.max(x, axis=axis).reshape((x.shape[0], 1))
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max.flatten()


def test_exp():
    i = T.fscalar()
#    y = T.exp(i)
    y = T.log(i)
#    y = T.nnet.sigmoid(i)
    f = theano.function(inputs=[i], outputs=[y])

    a = 5 - 100
    print f(a)


def test_subtensor():
    from collections import OrderedDict
    E = theano.shared(np.asarray([[2, 4], [2, 1], [3, 5], [4, 2]], dtype='float32'))
    E2 = theano.shared(np.asarray([[2, 4], [2, 1], [3, 5], [4, 2]], dtype='float32'))
    c = np.asarray([2, 1], dtype='int32')
    u = np.asarray([[1, 2], [2, 4]], dtype='float32')
    c2 = np.asarray([[0, 0, 1, 0], [0, 1, 0, 0]], dtype='float32')
    u2 = np.asarray([[1, 2], [2, 4]], dtype='float32')

    x = T.ivector('x')
    W = T.fmatrix('w')
    s = E[x]
    y = T.nnet.softmax(T.dot(s, W)).flatten()

    x2 = T.fmatrix('x2')
    W2 = T.fmatrix('w2')
    s2 = T.dot(x2, E2)
    y2 = T.nnet.softmax(T.dot(s2, W2)).flatten()

    updates = OrderedDict()
    f = theano.function(inputs=[x, W], outputs=y)
    sub = theano.function(inputs=[x], outputs=s)
    cost = - T.log(y[1])
    gy = T.grad(cost, s)
    updates[E] = T.inc_subtensor(s, -0.1 * gy)
    g = theano.function([x, W], gy, updates=updates)

    updates2 = OrderedDict()
    f2 = theano.function(inputs=[x2, W2], outputs=y2)
    sub2 = theano.function(inputs=[x2], outputs=s2)
    cost2 = - T.log(y2[1])
    gy2 = T.grad(cost2, E2)
    updates2[E2] = E2 - 0.1 * gy2
    g2 = theano.function([x2, W2], gy2, updates=updates2)

    print f(c, u)
    print sub(c)
    print g(c, u)
    print E.get_value(True)
    print

    print f2(c2, u2)
    print sub2(c2)
    print g2(c2, u2)
    print E2.get_value(True)


def test_check_gradient():
    u = np.asarray([[1, 2], [2, 4]], dtype=theano.config.floatX)
    c = np.asarray([2, 1], dtype=theano.config.floatX)
    d = np.asarray([1, 0], dtype=theano.config.floatX)

    x = T.fvector('x')
    W = T.fmatrix('w')
    y = T.nnet.softmax(T.dot(x, W)).flatten()
    f = theano.function(inputs=[x, W], outputs=y)
    print 'Prob: %s' % str(f(c, u))

    cost = T.sum((d - y) ** 2)
#    cost = - T.log(y[1])
    gy = T.grad(cost, x)
    g = theano.function([x, W], gy)
    print 'Gold Grad: %s' % str(g(c, u))

    ep = 0.0001
    cost_f = theano.function(inputs=[x, W], outputs=cost)
    aprox_grad = []

    for i in xrange(c.shape[0]):
        zero = np.zeros(c.shape, dtype=theano.config.floatX)
        zero[i] = 1.
        e = ep * zero
        c1 = c + e
        c2 = c - e

        j1 = cost_f(c1, u)
        j2 = cost_f(c2, u)

        g_pred = (j1 - j2) / (2. * ep)
        aprox_grad.append(g_pred)

    print 'Aprox Grad: %s' % str(aprox_grad)


def check_gradient(x, d, cost_f, ep=0.001):
    aprox_grad = []
    print x[0][:7]
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            zero = np.zeros(x.shape, dtype=theano.config.floatX)
            zero[i][j] = 1.
            e = ep * zero
            x1 = x + e
            x2 = x - e

            _, j1 = cost_f(x1, d)
            _, j2 = cost_f(x2, d)

            g_pred = (j1 - j2) / (2. * ep)
            aprox_grad.append(g_pred)
            if len(aprox_grad) > 6:
                break
        break

    print 'Aprox Grad: %s' % str(aprox_grad)


def test_debug_print():
    from theano.printing import debugprint
    x = T.dscalar('x')
    y = x ** 2
    gy = T.grad(y, x)
    debugprint(gy)  # print out the gradient prior to optimization
    f = theano.function([x], gy)
    debugprint(f.maker.fgraph.outputs[0])


def test_grad_clip():
    W = T.fmatrix()
    t = 2.
    y = T.switch(T.abs_(W) > t, t / T.abs_(W) * W, W)

    f = theano.function(inputs=[W], outputs=[y])
    w = [[1, -3], [-4, 1]]
    print f(w)


def test_lookup():
    a = T.itensor3()
    b = T.ivector()
    y = a[0][T.arange(b.shape[0]), b]
    f = theano.function(inputs=[a, b], outputs=[y])

    u = [[[1, 2], [2, 4]], [[3, 1], [2, 1]]]
    c = [0, 1]

    print f(u, c)


def test_dot():
    a = T.itensor3()
    W = T.imatrix()
    y = T.dot(a, W)
    f = theano.function(inputs=[a, W], outputs=[y])
    u = [[[1, 2], [2, 4]], [[3, 1], [2, 1]]]
    w = [[1, 1], [1, 1]]
    print f(u, w)


def test_vitabi_learning():
    x = T.fmatrix()
    d = T.ivector()
    size_x = 3
    size_y = 2

    l = lstm(x, d, size_x, size_y)
    f = theano.function(
        inputs=[x, d],
        outputs=[l.cost, l.pred_y],
#        outputs=[l.Z],
        updates=l.updates
    )

    data_x = np.asarray([[0.2, 0.9, 0.3], [0.2, 0.8, 0.1], [0.1, 0.6, 0.3]], dtype=theano.config.floatX)
    data_y = np.asarray([0, 0, 0], dtype='int32')
    epoch = 10
    for i in xrange(epoch):
        print f(data_x, data_y)


def test_vitabi():
    x = T.fmatrix()
    d = T.ivector()
    size_x = 3
    size_y = 2

    l = lstm(x, d, size_x, size_y)
    f = theano.function(
        inputs=[x, d],
        outputs=[l.Z, l.d_score, l.p, l.pred_y]
    )

    data_x = np.asarray([[0.2, 0.9, 0.3], [0.2, 0.8, 0.1], [0.1, 0.6, 0.3]], dtype=theano.config.floatX)
    data_y = []
    data_y.append(np.asarray([0, 0, 0], dtype='int32'))
    data_y.append(np.asarray([0, 0, 1], dtype='int32'))
    data_y.append(np.asarray([0, 1, 0], dtype='int32'))
    data_y.append(np.asarray([1, 0, 0], dtype='int32'))
    data_y.append(np.asarray([0, 1, 1], dtype='int32'))
    data_y.append(np.asarray([1, 0, 1], dtype='int32'))
    data_y.append(np.asarray([1, 1, 0], dtype='int32'))
    data_y.append(np.asarray([1, 1, 1], dtype='int32'))

    y = [f(data_x, data) for data in data_y]
#    z = np.sum([d[1] for d in y])
    p = [np.exp(d[2]) for d in y]
    total = np.sum(p)

    print 'Pred Z: %f' % y[0][0]
    print 'Pred Probs: ',
    for d in p:
        print d,
    print
    print 'Total Pred Probs: %f' % total
    print 'Best Path: %s' % str(y[0][3])


class lstm(object):
    def __init__(self, x, d, size_x=3, size_y=2):
        self.x = x
        self.d = d
        self.size_x = size_x
        self.size_y = size_y

        a = [[2, 1], [5, 3], [4, 2]]
        b = [[1, 3, 2], [2, 1, 5]]

        self.W = theano.shared(0.1 * np.asarray(a, dtype=theano.config.floatX))
        self.W_t = theano.shared(0.1 * np.asarray(b, dtype=theano.config.floatX))
        self.params = [self.W, self.W_t]

        self.scores0 = theano.shared(np.zeros(self.size_y+1, dtype=theano.config.floatX))
        self.score0 = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        self.node0 = theano.shared(np.asarray(self.size_y, dtype='int32'))

        self.emit = T.dot(self.x, self.W)
        self.Z = self.z(self.emit)
        self.d_score = self.get_d_score(self.emit)
        self.p = self.d_score - self.Z
        self.pred_y = self.vitabi(self.emit)

        self.lr = 0.1
        self.cost = - self.p
        self.grads = T.grad(self.cost, self.params)
        self.updates = [(p, p - self.lr * g) for p, g in zip(self.params, self.grads)]

    def get_d_score(self, emit):
        def d_step(d_t, e_t, d_prev, score_prev, trans):
            score = score_prev + trans[d_t][d_prev] + e_t[d_t]  # score_prev: 1D: current_tags, 2D: prev_tags
            return d_t, score

        [_, d_scores], _ = theano.scan(fn=d_step,
                                  sequences=[self.d, emit],
                                  outputs_info=[self.node0, self.score0],
                                  non_sequences=self.W_t)
        d_score = d_scores[-1]
        return d_score

    def z(self, emit):
        def z_step(e_t, score_prev, trans):
            score = logsumexp(score_prev + trans, axis=1) + e_t
            return score

        z_scores, _ = theano.scan(fn=z_step,
                                  sequences=[emit],
                                  outputs_info=[self.scores0],
                                  non_sequences=self.W_t)
        z_score = logsumexp_v(z_scores[-1])
        return z_score

    def vitabi(self, emit):
        def forward(e_t, score_prev, trans):
            score = score_prev + trans + e_t.dimshuffle(0, 'x')
            max_scores_t = T.max(score, axis=1)
            max_nodes_t = T.cast(T.argmax(score, axis=1), dtype='int32')
            return max_scores_t, max_nodes_t

        def backward(nodes_t, max_node_t):
            return nodes_t[max_node_t]

        [max_scores, max_nodes], _ = theano.scan(fn=forward,
                                                 sequences=[emit],
                                                 outputs_info=[self.scores0, None],
                                                 non_sequences=self.W_t)
        max_node = T.cast(T.argmax(max_scores[-1]), dtype='int32')

        nodes, _ = theano.scan(fn=backward,
                               sequences=max_nodes[::-1],
                               outputs_info=max_node)
        return T.concatenate([nodes[::-1].dimshuffle('x', 0), max_node.reshape((1, 1))], 1).flatten()

