from functions import Function
from numba import jit
import np_utils
import numpy as np
from multiprocessing import Pool

# @staticmethod
@jit(nopython=True)
def conv2d(args):
    """
    Example of an accelerated function, Notice the Numba jit decorator on top.
    """
    inp_ = args[0]
    kernel = args[1]
    stride = args[2]
    padding = args[3]
    (batch_size, in_channels, H, W) = inp_.shape
    (out_channels, in_channels_t, Hk, Wk) = kernel.shape
    Hc =  int((H - Hk)/stride)+1
    Wc = int((W - Wk)/stride)+1
    conv_layer = np.zeros((batch_size, out_channels, Hc, Wc))
    for batch_i in range(batch_size):
        for o_chann_i in range(out_channels):
            for in_chann_i in range(in_channels):
                curr_ker = kernel[o_chann_i, in_chann_i, :, :]
                curr_inp = inp_[batch_i, in_chann_i, :, :]
                h_ind = 0
                while h_ind + Hk <= H:
                    w_ind = 0
                    while w_ind + Wk <= W:
                        inp_patch = curr_inp[h_ind:h_ind+Hk, w_ind:w_ind+Wk]
                        # Sum the conv_value of all the inp_channels
                        conv_layer[batch_i, o_chann_i, h_ind//stride, w_ind//stride] += np.sum(inp_patch*curr_ker)
                        w_ind+=stride
                    h_ind+=stride
    return conv_layer

def conv2d_mul(inp_, kernel, stride, padding):
    n_threads = 20
    pool = Pool(n_threads)
    step = max(1, inp_.shape[0]//n_threads)

    batches = [(inp_[i*step:i*step+step], kernel, stride, padding) for i in range(n_threads)]
    batches[-1] = (inp_[(n_threads - 1)*step:], kernel, stride, padding)
    outputs = pool.map(conv2d, batches)
    return np.vstack(outputs)


class Convolution2D(Function):

    def forward(self, stride, padding, *args):
        """
        Forward pass of the convolution operation between two four dimensional tensors.
        :param stride: Convolution stride, defaults to 1.
        :param padding: Convolution padding, defaults to 0.
        :param args: Operands of convolution operation (input(batch_size, in_channels, H, W), kernel(out_channels, in_channels, Hk, Wk)).
        :return: Output of the convolution operation.
        """
        #TODO
        parents = list(args)
        inp_ = parents[0].value
        kernel = parents[1].value
        
        (batch_size, in_channels, H, W) = inp_.shape
        (out_channels, in_channels_t, Hk, Wk) = kernel.shape
        assert in_channels == in_channels_t
        
        return conv2d((inp_, kernel, stride, padding))
        # return conv2d_mul(inp_, kernel, stride, padding)

        

    def backward(self, gradient):
        """
        Sets the gradients for operands of convolution operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        pass


class Reshape(Function):
    def forward(self, shape, *args):
        """
        Forward pass of the reshape operation on a tensor
        :param shape: tuple of required dimension.
        :param args: Input tensor to be reshaped.
        :return: reshaped tensor.
        """
        #TODO
        return None

    def backward(self, gradient):
        """
        Sets the gradient for input of reshape operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        pass


