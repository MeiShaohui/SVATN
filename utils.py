#from .common_utils import *
# from utils.denoising_utils import *
import torch
import numpy as np
from skimage._shared import *
from skimage.util import *
from skimage.metrics.simple_metrics import _as_floats
from skimage.metrics.simple_metrics import mean_squared_error
from skimage._shared.utils import * 
#from pymf.dist import l1_distance
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def compare_snr(image_true, image_test, *, data_range=None):
    """
    Compute the signal to noise ratio (PSNR) for an image.
    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im_true.", stacklevel=2)
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    
    return 10 * np.log10((np.mean(image_test ** 2, dtype=np.float64)) / err)
 

def add_noise(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    """   
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
    """
    img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)

    return img_noisy_np

        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    """   
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noisy_np


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay= 0, amsgrad=False)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    elif optimizer_type == 'RMSprop':
        print('Starting optimization with RMSprop')
        optimizer = torch.optim.RMSprop(parameters, lr=LR,  alpha=0.99,
                    eps=1e-08, weight_decay=0, momentum=0, centered=False)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False