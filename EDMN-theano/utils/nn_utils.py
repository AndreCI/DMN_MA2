import theano
import theano.tensor as T
import lasagne

'''
Utils methods to create neural networks easily.
'''

def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out
    
def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])
    
    
def constant_param(value=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)
    

def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)

def GRU_update(h,x,Wr,Ur,Br,Wz,Uz,Bz,W,U,Bh):
    """ 
    Function who does the computation for any classical GRU
    In short, compute h_t = GRU(x_t, h_t-1)
    :variable z: update gate
    :variable r: reset gate
    :variable _h: potential state
    :param h: previous state
    :param x: current input
    :param params: different weights for GRU (3 W, 3 U, 3 b)
    :return new h: updated state
    """
    z = T.nnet.sigmoid(T.dot(Wz, x) + T.dot(Uz, h) + Bz)
    r = T.nnet.sigmoid(T.dot(Wr, x) + T.dot(Ur, h) + Br)
    _h = T.tanh(T.dot(W, x) + r * T.dot(U, h) + Bh)
    return z * h + (1 - z) * _h
                                                                
    