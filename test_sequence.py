__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T


def dynamic_sequence():
    def add(w, m):
        return w + m

    def forward(k_t, w_t):
        u = T.dot(k_t, w_t)
        m = T.max(u)
        a = T.argmax(u)
        w_t = T.switch(T.eq(T.arange(u.shape[0]), a), add(w_t, m), w_t)
        if T.eq(u.shape[0]-1, a):
            w_t = T.concatenate([w_t, T.zeros((2, 1), dtype=theano.config.floatX)], axis=1)
        return u, w_t

    wx = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=theano.config.floatX)
    kx = np.asarray([[0, 1], [1, 0]], dtype=theano.config.floatX)

    w = T.fmatrix('w')
    k = T.fmatrix('k')

    [r, s], _ = theano.scan(fn=forward,
                       sequences=[k],
                       outputs_info=[None, None],
                       non_sequences=[w.T])

    f = theano.function(inputs=[w, k], outputs=[r, s])
    print f(wx, kx)


