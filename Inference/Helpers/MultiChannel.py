import fastai
from fastai.vision.all import *
import rasterio as rio
import numpy as np
from rasterio.enums import Resampling

# Fuinction to read image values, including allowing for different normalization schemes and clip values 
def readSatImage(fn, max_val, i_bands, m_bands, learned_downsampling=True, norm='gamma', gamma=0.4):
    fromValsI=[-100, max_val]
    fromValsM=[-100, max_val]

    # Keep a consistent ordering of available bands
    i_bands = [band for band in ['I1', 'I2', 'I3'] if band in i_bands]
    m_bands = [band for band in ['M3', 'M4', 'M11'] if band in m_bands]

    i_res = 256
    m_res = 128 if learned_downsampling else 256

    # Allocate only the requested number of bands - at either 128 or 256 pixels res
    img = np.empty((len(i_bands), i_res, i_res), dtype=np.float32)
    img2 = np.empty((len(m_bands), m_res, m_res), dtype=np.float32)
    
    # Read in I bands
    for k, bandName in enumerate(i_bands):
        imgPath = fn.as_posix().replace('I1', bandName)
        with rio.open(imgPath) as r:
            
            # Optional resampling
            if r.height == i_res and r.width == i_res:
                data = r.read(1)
            else:
                if r.height > i_res: # from fine to coarse (256 to 128), use average resampling
                    resampling_method = Resampling.average
                else: # from coarse to fine (128 to 256), use bilinear interpolation
                    resampling_method = Resampling.bilinear
                data = r.read(1, out_shape=(i_res, i_res), resampling=resampling_method)
            
            data = FillMissingValues(data)
            img[k,::] = normalizeBand(data, fromValsI, norm=norm, gamma=gamma)

    # Read in M bands
    for k, bandName in enumerate(m_bands):
        imgPath = fn.as_posix().replace('I1', bandName)
        with rio.open(imgPath) as r:

            # Optional resampling
            if r.height == m_res and r.width == m_res:
                data = r.read(1)
            else:
                if r.height > m_res: # from fine to coarse (256 to 128), use average resampling
                    resampling_method = Resampling.average
                else: # from coarse to fine (128 to 256), use bilinear interpolation
                    resampling_method = Resampling.bilinear
                data = r.read(1, out_shape=(m_res, m_res), resampling=resampling_method)
            
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

def make_multi_channel_create(max_val, i_bands, m_bands, learned_downsampling=True, norm='gamma', gamma=0.4):
    def _create(fn):
        img = readSatImage(fn=fn, max_val=max_val, i_bands=i_bands, m_bands=m_bands, learned_downsampling=learned_downsampling, norm=norm, gamma=gamma)
        return [MultiChannelTensorImage(img[0]), MultiChannelTensorImage(img[1])]
    return _create

class MultiChannelTensorImage(TensorImage):
    @classmethod
    def create(cls, fn, **kwargs) ->None:
        img = readSatImage(fn=fn, max_val=max_val, i_bands=i_bands, m_bands=m_bands, learned_downsampling=learned_downsampling, norm=norm, gamma=gamma)
        return [cls(img[0]), cls(img[1])]

    def __repr__(self): return f'{self.__class__.__name__} size={"x".join([str(d) for d in self.shape])}'

MultiChannelTensorImage.create = Transform(MultiChannelTensorImage.create)

def MultiChannelImageBlock(cls=MultiChannelTensorImage, chans=None):
    return TransformBlock(partial(cls.create))