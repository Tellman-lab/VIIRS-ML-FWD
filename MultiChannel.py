import fastai
from fastai.vision.all import *
import rasterio as rio
import numpy as np

# Fuinction to read image values, including allowing for different normalization schemes and clip values 
def readSatImage(fn, max_val, norm='gamma', gamma=0.4):
    fromValsI=[-100, max_val]
    fromValsM=[-100, max_val]

    # Read in I bands at 375 m resolution
    img = np.empty((3,256,256))
    for k, bandName in enumerate(['I1', 'I2', 'I3']):
        imgPath = fn.as_posix().replace('I1', bandName)
        with rio.open(imgPath) as r:
            data = r.read(1)
            data = FillMissingValues(data)
            img[k,::] = normalizeBand(data, fromValsI, norm=norm, gamma=gamma)

    # Read in M bands at 750 m resolution
    img2 = np.empty((3,128,128))
    for k, bandName in enumerate(['M3', 'M4', 'M11']):
        imgPath = fn.as_posix().replace('I1', bandName)
        with rio.open(imgPath) as r:
            data = r.read(1)
            data = FillMissingValues(data)
            img2[k,::] = normalizeBand(data, fromValsM, norm=norm, gamma=gamma)
            
    return [torch.from_numpy(img), torch.from_numpy(img2)]

# Function to fill missing data pixels for inference
def FillMissingValues(data):
    data = data * 1.0

    mask = data == -28672
    if np.sum(mask) == 0:
        return data

    if np.sum(~mask) == 0:
        data *= 0.0
        return data

    # Set missing values to median pixel value for inference
    # Pixels will later be set back to no data after inference
    data[mask] = np.median(data[~mask])
    return data

# Function to normalize band values, optionally with log transform
def normalizeBand(data, fromVals, norm='gamma', gamma=0.4):
    
    data = np.clip(data, fromVals[0], fromVals[1])
    if norm=='log':
        shifted = data - fromVals[0] + 1e-6  # this part basically just ensures that min=1e-6 so no log(0) or log negative
        logged = np.log(shifted)
        # Normalizes between log(1e-6) and log(16100) if using default instrumental min/max clip vals
        normed = (logged - np.log(1e-6)) / (np.log(fromVals[1] - fromVals[0] + 1e-6) - np.log(1e-6))
    elif norm=='arcsinh':
        shifted = data - fromVals[0]
        asinh = np.arcsinh(shifted)
        # scale = 1000
        # asinh_scaled = np.arcsinh(shifted / scale)
        normed = (asinh - asinh.min()) / (asinh.max() - asinh.min())
    elif norm=='log1p':
        shifted = data - fromVals[0]
        log1p = np.log1p(shifted)
        denom = log1p.max() - log1p.min()
        if denom==0:
            normed = np.zeros_like(log1p)  # Sets all values to zeros in case where log1p values are all the same for the band, could use 1 or 0.5 instead, but should only be for fringe cases anyway
        else:
            normed = (log1p - log1p.min()) / denom
    elif norm=='gamma': # gamma value <1 emphasizes low values, >1 emphasizes high values
        shifted = data - fromVals[0] 
        normed = (shifted / (fromVals[1] - fromVals[0])) ** gamma
    else:
        normed = np.interp(data, fromVals, (0., 1.))
    return np.clip(normed, 0., 1.)

def make_multi_channel_create(max_val, norm='gamma', gamma=0.4):
    def _create(fn):
        img = readSatImage(fn=fn, max_val=max_val, norm=norm, gamma=gamma)
        return [MultiChannelTensorImage(img[0]), MultiChannelTensorImage(img[1])]
    return _create

class MultiChannelTensorImage(TensorImage):
    @classmethod
    def create(cls, fn, **kwargs) ->None:
        img = readSatImage(fn=fn, max_val=max_val, norm=norm, gamma=gamma)
        return [cls(img[0]), cls(img[1])]

    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'

MultiChannelTensorImage.create = Transform(MultiChannelTensorImage.create)

def MultiChannelImageBlock(cls=MultiChannelTensorImage, chans=None):
    return TransformBlock(partial(cls.create))