import theano

theano.config.floatX = 'float32'


if __name__ == '__main__':
    import test
    test.main()
