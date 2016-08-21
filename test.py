import numpy as np
import theano
import theano.tensor as T


def main():
    lookup()


def index():
    cands = T.imatrix('cands')
    h_in = np.asarray([[1, 0], [0, 1]], dtype='int32')

    y = T.dot(cands, T.arange(cands.shape[-1]))
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(h_in)


def norm():
#    cands = T.imatrix('cands')
    cands = T.ivector('cands')
#    h_in = np.asarray([[1, 2], [3, 4]], dtype='int32')
    h_in = np.asarray([2, 4], dtype='int32')

    y = cands.norm(2, axis=0)
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(h_in)


def outer():
#    cands = T.itensor3('cands')
    cands = T.imatrix('cands')
#    cands = T.ivector('cands')
#    h_in = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='int32')
    h_in = np.asarray([[1, 2], [3, 4]], dtype='int32')
#    h_in = np.asarray([2, 4], dtype='int32')
#    h_in = np.asarray([[1, 2, 3], [2, 1, 3]], dtype='int32')

    y = T.outer(cands, cands)
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(h_in)


def sort():
#    cands = T.itensor3('cands')
    cands = T.imatrix('cands')
#    h_in = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='int32')
#    h_in = np.asarray([[1, 2], [3, 4]], dtype='int32')
    h_in = np.asarray([[4, 2], [3, 1]], dtype='int32')

    y = T.sort(cands, axis=1)
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(h_in)


def flip():
#    cands = T.itensor3('cands')
    cands = T.imatrix('cands')
#    h_in = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='int32')
    h_in = np.asarray([[1, 2], [3, 4]], dtype='int32')

    y = cands[::-1]
    f = theano.function(inputs=[cands], outputs=[y], on_unused_input='ignore')
    print f(h_in)


def attention_origin():
    query = T.imatrix('query')
    cands = T.itensor3('cands')

    d = 2
    W1_c = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_c = theano.shared(np.ones((d, d), dtype='int32'))
    W1_h = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_h = theano.shared(np.ones((d, d), dtype='int32'))
    w    = theano.shared(np.ones((d,), dtype='float32'))
    W2_r = theano.shared(np.random.randint(-1, 1, (d, d)))
    W2_h = theano.shared(np.random.randint(-1, 1, (d, d)))
#    W2_r = theano.shared(np.ones((d, d), dtype='float32'))
#    W2_h = theano.shared(np.ones((d, d), dtype='float32'))

    q_in = np.asarray([[1, 2]], dtype='int32')
#    q_in = np.ones((1, 2), dtype='int32')
    C_in = np.ones((1, 3, 2), dtype='int32')

    def forward(h_before, C):
        # C is batch * len * d
        # h is batch * d

        M = T.dot(C, W1_c) + T.dot(h_before, W1_h).dimshuffle((0, 'x', 1))

        # batch*len*1
        alpha = T.nnet.softmax(T.dot(M, w))
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))

        # batch * d
        r = T.sum(C * alpha, axis=1)

        # batch * d
        h_after = T.dot(r, W2_r) + T.dot(h_before, W2_h)

        return h_after, r, alpha.reshape((alpha.shape[0], alpha.shape[1])), M

    y, a, b, c = forward(query, cands)
    f = theano.function(inputs=[query, cands], outputs=[y, a, b, c], on_unused_input='ignore')
    print f(q_in, C_in)


def attention_q():
    query = T.itensor3('query')
    cands = T.itensor3('cands')

    d = 2
    W1_c = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_c = theano.shared(np.ones((d, d), dtype='int32'))
    W1_h = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_h = theano.shared(np.ones((d, d), dtype='int32'))
    w    = theano.shared(np.ones((d,), dtype='float32'))
    W2_r = theano.shared(np.random.randint(-1, 1, (d, d)))
    W2_h = theano.shared(np.random.randint(-1, 1, (d, d)))
#    W2_r = theano.shared(np.ones((d, d), dtype='float32'))
#    W2_h = theano.shared(np.ones((d, d), dtype='float32'))

#    q_in = np.asarray([[[1, 2], [3, 4], [5, 6]]], dtype='int32')
    q_in = np.ones((1, 3, 2), dtype='int32')
#    C_in = np.ones((1, 3, 2), dtype='int32')
#    C_in = np.ones((4, 3, 3, 2), dtype='int32')
    C_in = np.asarray(np.random.randint(-2, 2, (1, 3, 2)), dtype='int32')

    def forward(query, cands, eps=1e-8):
        # cands: 1D: n_queries, 2D: n_cands-1, 3D: dim_h
        # query: 1D: n_queries, 2D: n_words, 3D: dim_h
        # mask: 1D: n_queries, 2D: n_cands, 3D: n_words

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words, 4D: dim_h
        M = T.dot(query, W1_c).dimshuffle(0, 'x', 1, 2) + T.dot(cands, W1_h).dimshuffle(0, 1, 'x', 2)

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        alpha = T.nnet.softmax(T.dot(M, w).reshape((cands.shape[0] * cands.shape[1], query.shape[1])))
        alpha = alpha.reshape((cands.shape[0], cands.shape[1], query.shape[1], 1))

        # 1D: n_queries, 2D: n_cands-1, 3D: n_words
        r = T.sum(query.dimshuffle((0, 'x', 1, 2)) * alpha, axis=2)  # 4 * 3 * 2

        # 1D: n_queries, 2D: n_cands, 3D: dim_h
        h_after = T.dot(r, W2_r)  # 4 * 3 * 2
#        return h_after, h_after
        return h_after, r, alpha.reshape((alpha.shape[0], alpha.shape[1], alpha.shape[2])), M

    y, a, b, c = forward(query, cands)
    f = theano.function(inputs=[query, cands], outputs=[y, a, b, c], on_unused_input='ignore')
    print f(q_in, C_in)


def seq_attention():
    # x: 1D: n_words, 2D: Batch, 3D n_h
    h = T.itensor3('x')

    d = 2
    W1_c = theano.shared(np.ones((d, d), dtype='int32'))
    W1_h = theano.shared(np.ones((d, d), dtype='int32'))
    w    = theano.shared(np.ones((d,), dtype='int32'))
    W2_r = theano.shared(np.ones((d, d), dtype='int32'))
    W2_h = theano.shared(np.ones((d, d), dtype='int32'))

    h_in = np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype='int32')

    y, _ = theano.scan(fn=one_attention,
                       sequences=h,
                       outputs_info=None,
                       non_sequences=[h, W1_c, W1_h, w, W2_r, W2_h]
                       )

    f = theano.function(inputs=[h], outputs=[y])
    print f(h_in)


def one_attention(h, C, W1_c, W1_h, w, W2_r, W2_h, eps=1e-8):
    M = T.dot(C, W1_c) + T.dot(h, W1_h).dimshuffle(0, 'x', 'x', 1)  # 4 * 3 * 3 * 2

    # batch * len * 1
    alpha = T.dot(M, w)  # 4 * 3 * 3
    alpha /= T.sum(alpha, axis=2, keepdims=True) + eps
    alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], alpha.shape[2], 1))

    # batch * d
    r = T.sum(C * alpha, axis=2)  # 4 * 3 * 2

    # batch * d
    # 4 * 3 * 2
#    return T.tanh(T.dot(r, W2_r) + T.dot(h, W2_h))
    return T.dot(r, W2_r) + T.dot(h, W2_h)


def attention():
#    q = T.fmatrix('q')
#    C = T.ftensor4('C')
    q = T.imatrix('q')
    C = T.itensor4('C')

    d = 2
    W1_c = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_c = theano.shared(np.ones((d, d), dtype='int32'))
    W1_h = theano.shared(np.random.randint(-3, 3, (d, d)))
#    W1_h = theano.shared(np.ones((d, d), dtype='int32'))
    w    = theano.shared(np.ones((d,), dtype='float32'))
    W2_r = theano.shared(np.random.randint(-1, 1, (d, d)))
    W2_h = theano.shared(np.random.randint(-1, 1, (d, d)))
#    W2_r = theano.shared(np.ones((d, d), dtype='float32'))
#    W2_h = theano.shared(np.ones((d, d), dtype='float32'))

#    q_in = np.asarray([[1, 2], [3, 4], [-1, -2], [-3, -4]], dtype='int32')
    q_in = np.asarray([[1, 2]], dtype='int32')
#    q_in = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype='float32')
    C_in = np.ones((1, 3, 3, 2), dtype='int32')
#    C_in = np.ones((4, 3, 3, 2), dtype='int32')
#    C_in = np.asarray(np.random.randint(-2, 2, (1, 3, 3, 2)), dtype='int32')

    def forward(h_before, _C, eps=1e-8):
        # C: n_queries * n_cands * n_words * dim_h
        # h: n_queries * dim_h

#        M = T.tanh(T.dot(_C, W1_c) + T.dot(h_before, W1_h).dimshuffle(0, 'x', 'x', 1))
        M = T.dot(_C, W1_c) + T.dot(h_before, W1_h).dimshuffle(0, 'x', 'x', 1)  # 4 * 3 * 3 * 2
#        M = T.dot(h_before, W1_h).dimshuffle(0, 'x', 'x', 1)

        # batch * len * 1
        alpha = T.exp(T.dot(M, w))  # 4 * 3 * 3
#        alpha = T.nnet.softmax(T.dot(M, w))  # 4 * 3 * 3
        alpha /= T.sum(alpha, axis=2, keepdims=True) + eps
#        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], alpha.shape[2], 1))

        # batch * d
#        r = T.sum(_C * alpha, axis=1)
        r_in = _C * alpha
        r = T.sum(r_in, axis=1)  # 4 * 3 * 2

        # batch * d
        h_after = T.dot(r, W2_r) + T.dot(h_before, W2_h).dimshuffle((0, 'x', 1))  # 4 * 3 * 2
#        return h_after
        return h_after, r, alpha, M

    y, a, b, m = forward(q, C)
    f = theano.function(inputs=[q, C], outputs=[y, a, b, m], on_unused_input='ignore')
    print f(q_in, C_in)


def softmax_3d():
    C_in = np.ones((3, 3, 2), dtype='float32')
    m = T.ftensor3()

    z = T.sum(m, axis=2, keepdims=True)
    y = m / z

    f = theano.function(inputs=[m], outputs=[y], on_unused_input='ignore')
    print f(C_in)


def add_b():
    w = T.imatrix('w')
    a = T.imatrix('a')
    y = w + a.repeat(4, 0)
    f = theano.function(inputs=[w, a], outputs=[y])

    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype='int32')
    b = np.asarray([[2, 1]], dtype='int32')
    print f(e, b)


def zero_pad_gate():
    dim_emb = 2
    window = 1
#    w = T.imatrix('w')
    w = T.itensor3('w')
    zero = T.zeros((1, 1, dim_emb * window), dtype=theano.config.floatX)

#    y = T.eq(w, zero)
    y = T.eq(T.sum(T.eq(w, zero), 2, keepdims=True), 0) * w
    f = theano.function(inputs=[w], outputs=[y])

    e = np.asarray([[[2, 4], [0, 0]], [[3, 2], [4, 1]]], dtype='int32')
    print f(e)

#    return T.eq(T.sum(T.eq(matrix, self.zero), 2, keepdims=True), dim_emb * window)

def double_roop():
    a = T.fmatrix('a')
    r = T.zeros(shape=(5, 2), dtype=theano.config.floatX)

    def recursive(t, s):
        def forward(_t, seq):
#            z_t = zero[:seq.shape[0]]
#            h_t = T.set_subtensor(z_t, seq)
            return seq[_t] + 1

        u, _ = theano.scan(fn=forward,
                                sequences=T.arange(s[t].shape[0]),
                                outputs_info=None,
                                non_sequences=s[t])
#                                n_steps=s.shape[0]-1)
        return u

    y, _ = theano.scan(fn=recursive,
                       sequences=[T.arange(a.shape[0])],
                       outputs_info=None,
                       non_sequences=a)

#    y = recursive(a)
    f = theano.function(inputs=[a], outputs=[y])
    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype=theano.config.floatX)
    print f(e)


def grnn_one_gate():
    np.random.seed(0)
    matrix = T.ftensor3('a')
    n_d = 2

    W = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 2, n_d)))
    U = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 3, n_d * 3)))
    G = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 2, n_d * 2)))

    eps = 1e-8

    def step(m):
        h_l = m[:, 0]
        h_r = m[:, 1]

        # 1D: batch, 2D: 2 * n_d
        r = T.nnet.sigmoid(T.dot(T.concatenate([h_l, h_r], axis=1), G))
        half = r.shape[1]/2
        # 1D: batch, 2D: n_d
        r_l = r[:, :half]
        r_r = r[:, half:]

        # 1D: batch, 2D: n_d
        h_hat = T.tanh(T.dot(T.concatenate([r_l * h_l, r_r * h_r], axis=1), W))

        # 1D: batch, 2D: 3 * n_d
        z_hat = T.exp(T.dot(T.concatenate([h_hat, h_l, h_r], axis=1), U))
        # 1D: batch, 2D: 3 (h_hat, h_l, h_r), 3D: n_d
        z_hat = z_hat.reshape((z_hat.shape[0], 3, z_hat.shape[1] / 3))

        # 1D: batch, 2D: n_d
        Z = T.sum(z_hat, axis=1)
        # 1D: batch, 2D: 3, 3D: n_d
        Z = T.repeat(Z, repeats=3, axis=1).reshape((Z.shape[0], n_d, 3)).dimshuffle((0, 2, 1))

        z = z_hat / Z + eps

        h = h_hat * z[:, 0] + h_l * z[:, 1] + h_r * z[:, 2]

        seq_sub_tensor = h
        return seq_sub_tensor

    y = step(matrix)
    f = theano.function(inputs=[matrix],
                        outputs=[y])
#    e = np.asarray([[[2, 4], [2, 1], [3, 2], [4, 1]], [[1, 3], [2, 3], [2, 1], [3, 2]]], dtype=theano.config.floatX)
    e = np.ones((2, 4, n_d), dtype='float32')
    print f(e)


def grnn_batch_gate():
    matrix = T.ftensor3('a')
    n_d = 2

#    W = theano.shared(np.ones((n_e * 2, n_d), dtype='float32'))
    W = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 2, n_d)))
    U = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 3, n_d * 3)))
    G = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 2, n_d * 2)))


#    zero = T.zeros(shape=matrix.shape, dtype=theano.config.floatX)
    zero = T.zeros(shape=(matrix.shape[0], matrix.shape[1], n_d), dtype=theano.config.floatX)
    eps = 1e-8

    def recursive(t, m):
        def step(_t, _m):
            h_l = _m[:, _t]
            h_r = _m[:, _t + 1]

            # 1D: batch, 2D: 2 * n_d
            r = T.nnet.sigmoid(T.dot(T.concatenate([h_l, h_r], axis=1), G))
            half = r.shape[1]/2
            # 1D: batch, 2D: n_d
            r_l = r[:, :half]
            r_r = r[:, half:]

            # 1D: batch, 2D: n_d
            h_hat = T.tanh(T.dot(T.concatenate([r_l * h_l, r_r * h_r], axis=1), W))

            # 1D: batch, 2D: 3 * n_d
            z_hat = T.exp(T.dot(T.concatenate([h_hat, h_l, h_r], axis=1), U))
            # 1D: batch, 2D: 3 (h_hat, h_l, h_r), 3D: n_d
            z_hat = z_hat.reshape((z_hat.shape[0], 3, z_hat.shape[1] / 3))

            # 1D: batch, 2D: n_d
            Z = T.sum(z_hat, axis=1)
#            Z = T.sum(z_hat, axis=1, keepdims=True)
            # 1D: batch, 2D: 3, 3D: n_d
            Z = T.repeat(Z, repeats=3, axis=1).reshape((Z.shape[0], n_d, 3)).dimshuffle((0, 2, 1))
#            Z = T.repeat(Z, repeats=3, axis=2).dimshuffle((0, 2, 1))

            z = z_hat / Z + eps

            h = h_hat * z[:, 0] + h_l * z[:, 1] + h_r * z[:, 2]

            seq_sub_tensor = h
            return _m, seq_sub_tensor

        [_, u], _ = theano.scan(fn=step,
                                sequences=T.arange(m.shape[1] - t - 1),
                                outputs_info=[m, None])

        return T.set_subtensor(zero[:, :m.shape[1] - t - 1], u.dimshuffle((1, 0, 2)))

    y, _ = theano.scan(fn=recursive,
                       sequences=T.arange(matrix.shape[1]-1),
                       outputs_info=matrix)

    params = [W, G]
    cost = T.mean(y)
    updates = [(W, W - T.grad(cost=cost, wrt=W)), (G, G - T.grad(cost=cost, wrt=G)), (U, U - T.grad(cost=cost, wrt=U))]

    y = y[-1][:, 0]
    f = theano.function(inputs=[matrix],
                        outputs=[y],
                        updates=updates)

    e = np.asarray([[[2, 4], [2, 1], [3, 2], [4, 1]], [[1, 3], [2, 3], [2, 1], [3, 2]]], dtype=theano.config.floatX)
    for i in xrange(5):
        print i
        print f(e)


def grnn_batch():
    matrix = T.ftensor3('a')
    n_e = 2
    n_d = 2

#    W = theano.shared(np.ones((n_e * 2, n_d), dtype='float32'))
    W = theano.shared(np.random.uniform(low=-1., high=1., size=(n_d * 2, n_d)))


#    zero = T.zeros(shape=matrix.shape, dtype=theano.config.floatX)
    zero = T.zeros(shape=(matrix.shape[0], matrix.shape[1], n_d), dtype=theano.config.floatX)

    def recursive(t, m):
        def step(_t, _m):
            m_t = T.concatenate([_m[:, _t], _m[:, _t + 1]], axis=1)
            seq_sub_tensor = T.dot(m_t, W)
            return _m, seq_sub_tensor

        [_, u], _ = theano.scan(fn=step,
                                sequences=T.arange(m.shape[1] - t - 1),
                                outputs_info=[m, None])

        return T.set_subtensor(zero[:, :m.shape[1] - t - 1], u.dimshuffle((1, 0, 2)))

    y, _ = theano.scan(fn=recursive,
                       sequences=T.arange(matrix.shape[1]-1),
                       outputs_info=matrix)

    updates = [(W, W - T.grad(cost=T.mean(y), wrt=W))]

    y = y[-1][:, 0]
    f = theano.function(inputs=[matrix],
                        outputs=[y],
                        updates=updates)

    e = np.asarray([[[2, 4], [2, 1], [3, 2], [4, 1]], [[1, 3], [2, 3], [2, 1], [3, 2]]], dtype=theano.config.floatX)
    for i in xrange(5):
        print i
        print f(e)


def grnn():
    matrix = T.fmatrix('a')
#    W = theano.shared(np.zeros((2, 2), dtype='float32'))
    W = theano.shared(np.random.uniform(low=-1., high=1., size=(2, 2)))
    zero = T.zeros(shape=(4, 2), dtype=theano.config.floatX)

    def recursive(t, m):
        def step(_t, _m):
            seq_sub_tensor = T.dot(_m[_t] + _m[_t + 1], W)
            return _m, seq_sub_tensor

        [_, u], _ = theano.scan(fn=step,
                                sequences=T.arange(m.shape[0] - t - 1),
                                outputs_info=[m, None])

        return T.set_subtensor(zero[:m.shape[0] - t - 1], u)

    y, _ = theano.scan(fn=recursive,
                       sequences=T.arange(matrix.shape[0]-1),
                       outputs_info=matrix)

    updates = [(W, W - T.grad(cost=T.mean(y), wrt=W))]

    f = theano.function(inputs=[matrix],
                        outputs=[y],
                        updates=updates)

    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype=theano.config.floatX)
    for i in xrange(5):
        print i
        print f(e)


def refined_grnn():
    matrix = T.fmatrix('a')
    zero = T.zeros(shape=(4, 2), dtype=theano.config.floatX)

    def recursive(t, m):
        def forward(_t, _m):
            seq_sub_tensor = _m[_t] + _m[_t + 1]
            return _m, seq_sub_tensor

        [_, u], _ = theano.scan(fn=forward,
                                sequences=T.arange(m.shape[0] - t - 1),
                                outputs_info=[m, None])
        return T.set_subtensor(zero[:m.shape[0] - t - 1], u)

    y, _ = theano.scan(fn=recursive,
                       sequences=T.arange(matrix.shape[0]-1),
                       outputs_info=matrix)

    f = theano.function(inputs=[matrix], outputs=[y])
    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype=theano.config.floatX)
    print f(e)


def switch():
    w = T.imatrix('w')
    a = T.ivector('a')
#    y = T.switch(T.eq(w, a.reshape((a.shape[0], 1))), T.eq(a.reshape((a.shape[0], 1)), 1), 0)
    y = T.switch(a, 1, 0)
#    f = theano.function(inputs=[w, a], outputs=[y])
    f = theano.function(inputs=[a], outputs=[y])

    e = np.asarray([[2, 0], [2, 1], [3, 2], [2, 1]], dtype='int32')
    b = np.asarray([0, 7, 1, 1], dtype='int32')
#    print f(e, b)
    print f(b)


def eq():
    w = T.imatrix('w')
    a = T.ivector('a')
#    y = T.sum(T.eq(w, a.reshape((a.shape[0], 1))), 0)
#    y = T.eq(w, a.reshape((a.shape[0], 1)))
    y = T.neq(w, 3)
    f = theano.function(inputs=[w], outputs=[y])

    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype='int32')
    b = np.asarray([2, 1, 1, 0], dtype='int8')
    print f(e)


def printing():
    from theano import pp, tensor as T
    x = T.dscalar('x')
    y = x ** 2
    gy = T.grad(y, x)
    pp(gy)


def logsumexp_v(e):
    a = T.fvector()
    b = T.max(a)
    y = b + T.log(T.sum(T.exp(a-b)))
    f = theano.function(inputs=[a], outputs=[y])
    print e - f(e)


def softmax(e):
    a = T.fvector()
    y = T.log(T.nnet.softmax(a))
    f = theano.function(inputs=[a], outputs=[y])
    print f(e)


def logsumexp_new():
    return


def typed_list_with_shared():
    # typed_list cannot contain shared variable and numpy

    from theano.typed_list import TypedListType

    e = TypedListType(T.fvector)()
    u = T.fvector()
    o = theano.typed_list.append(e, u)
    x = [[1., 2.], [3., 4., 5.], [5., 6., 7], [5., 6.]]

    f = theano.function(inputs=[e, u], outputs=[o])
    print f([], [1., 2., 4.])


def broadcast():
    w = T.imatrix('w')
    y = w * T.ones(shape=(2, w.shape[0], w.shape[1]))
    f = theano.function(inputs=[w], outputs=[y])

    e = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype='int32')
    print f(e)


def vitabi():
    def transition(e_t, score_prev, trans):
        return e_t.T + trans + score_prev

    def forward_step(t, score_prev, e, b, trans):
        score = T.switch(t > 0, transition(e[t-1], score_prev, trans), transition(b, score_prev, trans))
        return T.max(score, axis=1), T.argmax(score, axis=1)

    def backward_step(node_t, prev_node):
        return prev_node, node_t[prev_node]

    seq = T.fmatrix()
    bos = T.fvector()
    W = theano.shared(np.asarray([[1, 0], [2, 1]], dtype='float32'))
    zero = theano.shared(np.zeros(shape=(2), dtype='float32'))

    [max_scores, max_nodes], _ = theano.scan(fn=forward_step,
                                             sequences=T.arange(seq.shape[0]+1),
                                             outputs_info=[zero, None],
                                             non_sequences=[seq, bos, W])

    y_hat_index = T.argmax(max_scores[-1])
    [path, _],  _ = theano.scan(fn=backward_step,
                                sequences=max_nodes[::-1],
                                outputs_info=[None, y_hat_index]
                                )

    y_score = max_scores[-1][y_hat_index]
    y_hat_score = max_scores[-1][0]

    f = theano.function(inputs=[seq, bos], outputs=[y_hat_score, y_score, path])

    s = np.asarray([[5, 6], [2, 4], [3, 2], [1, 2]], dtype='float32')
    b = np.asarray([200, 100], dtype='float32')

    print f(s, b)


def forward():
    def transition(e_t, score_prev, trans):
        return e_t.T + trans + score_prev

    def forward_step(t, score_prev, e, b, trans):
        score = T.switch(t > 0, transition(e[t-1], score_prev, trans), transition(b, score_prev, trans))
        return T.max(score, axis=1), T.argmax(score, axis=1)

    seq = T.fmatrix()
    bos = T.fvector()
    W = theano.shared(np.asarray([[1, 0], [2, 1]], dtype='float32'))
    zero = theano.shared(np.zeros(shape=(2), dtype='float32'))

    [scores, nodes], _ = theano.scan(fn=forward_step,
                                     sequences=T.arange(seq.shape[0]+1),
                                     outputs_info=[zero, None],
                                     non_sequences=[seq, bos, W])

    f = theano.function(inputs=[seq, bos], outputs=[scores, nodes])

    x1 = np.asarray([[5, 6], [2, 4], [3, 2], [1, 2]], dtype='float32')
    x2 = np.asarray([200, 100], dtype='float32')

    print f(x1, x2)


def backward():
    def backward_step(node_t, prev_node):
        return prev_node, node_t[prev_node]

    max_nodes = T.imatrix()
    max_scores = T.fmatrix()

    [path, _],  _ = theano.scan(fn=backward_step,
                                sequences=max_nodes[::-1],
                                outputs_info=[None, T.cast(T.argmax(max_scores[-1]), 'int32')]
                                )

    f = theano.function(inputs=[max_nodes, max_scores], outputs=path)

    n = np.asarray([[0, 1], [1, 0], [0, 1], [0, 1]], dtype='int32')
    s = np.asarray([[7, 6], [2, 1], [3, 2], [1, 2]], dtype='float32')

    print f(n, s)


def maximum():
    e1 = np.asarray([[2, 4], [2, 1], [3, 2], [4, 1]], dtype='int32')
    e2 = np.asarray([[2, 1], [3, 2], [3, 4], [2, 3]], dtype='int32')

    w = T.imatrix('w')
    a = T.imatrix('a')

    y = T.maximum(w, a)
    f = theano.function(inputs=[w, a], outputs=[y])

    print f(e1, e2)


def lookup():
    e = theano.shared(np.asarray([[2, 1], [10, 3], [3, 5], [4, 6]], dtype='int32'))
#    w = T.imatrix('w')
#    v = T.imatrix('v')
#    v = T.itensor3('v')
#    w = T.ivector('w')
    v = T.ivector('v')
    w = T.itensor3('w')

    y = w[T.arange(w.shape[0]), v]
#    y = w[:, v]
    f = theano.function(inputs=[w, v], outputs=[y])

#    print f([[1, 2], [3, 4]],
#            [[0, 1, 1], [1, 0, -1]])
    print f([[[0, 0], [0, 1], [0, 1]], [[1, 1], [1, 0], [1, -1]]],
            [1, 2])
#    print f([1, 2],
#            [0, 1, 1])
#    print f([[[5, 2], [3, 1], [2, 1]], [[1, 2], [3, 4], [5, 6]]])


def max3d():
    e = np.asarray([[[2, 1], [10, 3]], [[3, 5], [4, 6]]], dtype='int32')
    v = np.asarray([0, 1], dtype='int32')
    w = T.itensor3('w')
    a = T.ivector('a')

#    y = T.max(w, axis=1)
#    y = T.max(w[:, :, :1], axis=1)
#    y = w[T.arange(a.shape[0]), a]
    y = w[:, a]
#    y_a = y / a
#    y_r = y % a
#    y = T.max(w, axis=[1, 2])
#    f = theano.function(inputs=[w, a], outputs=[y, y_a, y_r])
    f = theano.function(inputs=[w, a], outputs=[y])

    print f(e, v)


def copy():
    e = np.asarray([[[2, 4], [5, 1]], [[3, 5], [4, 6]]], dtype='int32')
    w = T.itensor3('w')
    u = T.ones(shape=(2, w.shape[2]))

    y = T.repeat(T.max(w, axis=1, keepdims=True), 2, 1)
#    y = T.max(w, axis=1, keepdims=True) * u
    f = theano.function(inputs=[w], outputs=y)

    print f(e)


def typed_list():
    import theano.typed_list
    from theano.typed_list import TypedListType

#    e = TypedListType(T.ivector)()
    e = TypedListType(T.ivector)()
    l = theano.typed_list.length(e)
    emb = theano.shared(np.asarray([[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]], dtype=theano.config.floatX))

    def forward(l1, e1):
        u = e1[l1]
        return T.max(emb[u], axis=0)

    y, _ = theano.scan(fn=forward, sequences=T.arange(l, dtype='int64'), outputs_info=[None], non_sequences=[e])
    y = T.sum(y)

    f = theano.function(inputs=[e], outputs=y)
    print f([[1, 2], [3, 4, 5]])


def sum():
    e = np.asarray([[1., 2.], [3., 4.], [5, 6], [5., 6.], [7., 8.], [9, 10]], dtype=theano.config.floatX)
    w = T.fmatrix('w')
    y = T.sum(w, 1)
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
    e = np.ones(shape=(1, 1, 10, 10), dtype=theano.config.floatX)  # 3D: n_char, 4D: emb_dim

#    x = np.asarray([[1, 1], [1, 1], [1, 1]], dtype=theano.config.floatX)
#    x = np.asarray([[[1], [1]], [[2], [1]]], dtype=theano.config.floatX)
#    x = np.asarray([[1, 1, 1], [2, 1, 1]], dtype=theano.config.floatX)  # 1D: 2, 2D: 3
#    x = np.ones(shape=(5, 1, 3, 10), dtype=theano.config.floatX)  # 1D: resulting emb dim, 3D: n_window, 4D: char_emb_dim
    x = np.ones(shape=(5, 50), dtype=theano.config.floatX)  # 1D: resulting emb dim, 3D: n_window, 4D: char_emb_dim
#    x = np.asarray([[[1, 1], [1, 1], [1, 1]], [[1, 2], [1, 1], [1, 1]]], dtype=theano.config.floatX)
#    x = np.asarray([[[1, 1]], [[2, 1]]], dtype=theano.config.floatX)
    zero = theano.shared(np.zeros(shape=(1, 1, 2, 10), dtype=theano.config.floatX))

    w = T.ftensor4('w')
#    v = T.ftensor4('v')
    v = T.fmatrix('w')
#    w = T.fmatrix('w')
#    v = T.ftensor3('v')
#    v = T.fmatrix('v')

    s = v.reshape((v.shape[0], 1, 5, 10))
    u = T.concatenate([zero, w, zero], axis=2)
    y = conv2d(input=u, filters=s)
    y = y.reshape((y.shape[1], y.shape[2]))  # 1D: h_dim, 2D: n_window_slides
#    y = conv2d(input=w, filters=v)
#    y = conv2d(input=u.T, filters=v.dimshuffle(0, 1, 'x'), subsample=(3, 1))
#    y = T.max(y, axis=1)
#    y = c.reshape((c.shape[0], c.shape[1]))
#    y = u.T
    f = theano.function(inputs=[w, v], outputs=[u, y])
#    f = theano.function(inputs=[w], outputs=w.T)

    print f(e, x)
#    print f(e)


def conv_with_scan():
    dim_c_emb = 10
    window = 5
    h_emb = 10
    c_emb_in = np.ones(shape=(10, dim_c_emb), dtype=theano.config.floatX)  # 1D: n_char, 2D: emb_dim
    w_in = np.ones(shape=(dim_c_emb * window, h_emb), dtype=theano.config.floatX)  # 1D: resulting emb dim, 3D: n_window, 4D: char_emb_dim

    w = T.fmatrix('w')
    c_emb = T.fmatrix('v')
    zero = theano.shared(np.zeros(shape=(window/2, dim_c_emb), dtype=theano.config.floatX))
    win = T.iscalar('win')

    u = T.concatenate([zero, c_emb, zero], axis=0)
    u_in = u[:win].flatten()
    y = T.dot(u_in, w)
    f = theano.function(inputs=[c_emb, w, win], outputs=[u_in, y])

    print f(c_emb_in, w_in, window)


def conv_2d():
    from theano.tensor.signal.conv import conv2d
    e = np.ones(shape=(8, 10), dtype=theano.config.floatX)
    x = np.ones(shape=(3, 10), dtype=theano.config.floatX)
    zero = theano.shared(np.zeros(shape=(1, 10), dtype=theano.config.floatX))

    w = T.fmatrix('w')
    v = T.fmatrix('v')

    u = T.concatenate([zero, w, zero], axis=0)
    y = conv2d(input=u.T, filters=v.dimshuffle(0, 1, 'x'), subsample=(3, 1))
#    y = T.max(c.reshape((c.shape[0], c.shape[1])), axis=1)
#    y = c.reshape((c.shape[0], c.shape[1]))
#    y = u.T
    f = theano.function(inputs=[w, v], outputs=y)

    print f(e, x)


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
    e = np.asarray([[[2, 4], [5, 1], [2, 3]], [[3, 5], [4, 6], [3, 1]]], dtype='float32')
    w = T.ftensor3('w')

    y = T.concatenate([T.mean(w[:, :2], axis=1), T.mean(w[:, 1:], axis=1)], 1)
#    y = w[:, 1:]
    f = theano.function(inputs=[w], outputs=y)

    print f(e)


def multiply():
    e1 = np.asarray([[2, 4, 1], [3, 1, 2]], dtype='int32')
    e2 = np.asarray([[1, 0], [0, 1]], dtype='int32')
    w = T.imatrix('v1')
    v = T.imatrix('v2')

#    y = w * v.reshape((v.shape[0], v.shape[1], 1)) #T.repeat(v, 2, 1)
    y = w.dimshuffle(0, 'x', 1) * v.dimshuffle(0, 1, 'x') #T.repeat(v, 2, 1)
    f = theano.function(inputs=[w, v], outputs=y)

    print f(e1, e2)


def repeat():
#    e = np.asarray([[[2, 4], [5, 1], [2, 1]]], dtype='int32')
    e = np.asarray([[[2, 4], [5, 1], [2, 10]], [[20, 4], [5, 10], [20, 1]]], dtype='int32')
#    e = np.asarray([[[4], [1], [2]], [[4], [5], [1]]], dtype='int32')
#    e = np.asarray([[2, 4], [5, 1]], dtype='int32')
    w = T.itensor3('w')
#    w = T.imatrix('w')

#    y = T.repeat(w, T.cast(w.shape[1], dtype='int32'), 0)[T.arange(w.shape[1]), 1:]
#    y = T.sum(w, axis=1)
    y = T.repeat(T.sum(w, axis=1), 2, axis=1).reshape((w.shape[0], 2, 2))
#    y = T.repeat(T.repeat(w, T.cast(w.shape[1], dtype='int32'), 0)[T.arange(w.shape[1]), 1:], 2, 0)
#    y = T.repeat(w, 2, 0)
    f = theano.function(inputs=[w], outputs=y)

    print f(e)


def tensor_max():
    e = np.asarray([[[2, 4], [5, 1]], [[3, 5], [4, 6]]], dtype='int32')
    w = T.itensor3('w')

    y = T.max(w, axis=1)
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
#    y = v1 * T.ones_like(w)
    y = v1 + w
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
#    a = T.itensor3()
    W = T.imatrix()
    a = T.ones(shape=(2, W.shape[0], W.shape[1]), dtype='int32')
#    y = T.dot(a, W)
    y = a * W
    f = theano.function(inputs=[W], outputs=[y])
#    u = [[[1, 2], [2, 4]], [[3, 1], [2, 1]]]
#    w = [[1, 1], [1, 1]]
    u = [[1, 2], [2, 4]]
#    print f(u, w)
    print f(u)


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

