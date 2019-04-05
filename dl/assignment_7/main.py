import numpy as np
from tensor import Tensor

if __name__=="__main__":
    a = np.random.rand(50, 1, 50, 50)
    b = np.random.rand(20, 1, 2, 2)
    # Implementation
    result_1 = Tensor(a).conv2d(Tensor(b)).value