import os
import rasterio as rio
from pathlib import Path
from affine import Affine
from rasterio.windows import Window
import numpy as np
from rasterio import MemoryFile

# Function to pad the original scene so that inference still works up to the scene edge 
# even when using a buffer for the feathering option 
def GetPaddedScene(img_path, buffer, pad_value=-28672):
    with rio.open(img_path) as src:
        profile = src.profile.copy()

        # Original dimensions
        width, height = src.width, src.height
        transform = src.transform

        # New dimensions with padding
        new_width = width + 2 * buffer
        new_height = height + 2 * buffer

        # Adjust the transform so that the original raster sits in the center
        # Move the origin `buffer` pixels up and left
        new_transform = Affine(
            transform.a,
            transform.b,
            transform.c - buffer * transform.a,  # Adjust x-origin
            transform.d,
            transform.e,
            transform.f - buffer * transform.e,  # Adjust y-origin
        )

        # Update profile for the padded dataset
        profile.update({
            "width": new_width,
            "height": new_height,
            "transform": new_transform
        })

        # Read original data
        data = src.read()

        # Fill entire padded dataset with pad_value
        pad_tile = np.full(
            (src.count, new_height, new_width),
            pad_value,
            dtype=src.dtypes[0]
        )

        # Insert original data into the center of pad_tile
        pad_tile[:, buffer:buffer + height, buffer:buffer + width] = data

    memfile = MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(pad_tile)

    return memfile


# Function to create chips, including using padding at scene edges (for scenes smaller than required 256x256 pixels)
# Additionally include buffer for feathering optiom
# *** This could be parallelized i.e. each img_path run in parallel
def CreateChips(img_path, patch_size, outputFolder, pad_value=-28672, buffer=0):
    print(f'Creating chips for {img_path}.', end='\r')
    base_name = Path(img_path).stem

    open_raster_call = lambda p: rio.open(p)
    # If using a buffer for feathering, first pad the scene outwards at the edges
    if buffer > 0:
        memfile = GetPaddedScene(img_path, buffer, pad_value)
        open_raster_call = lambda p: memfile.open()

    with open_raster_call(img_path) as dataset:
        width, height = dataset.width, dataset.height
        profile = dataset.profile

        for k_x, x in enumerate(range(0, width, patch_size-buffer)):
            for k_y, y in enumerate(range(0, height, patch_size-buffer)):
                win_width = min(patch_size, width - x)
                win_height = min(patch_size, height - y)
                window = Window(x, y, win_width, win_height)
                data = dataset.read(window=window)
                data[(data < -100) | (data > 16100)] = -28672

                # Pad to full patch size with the desired value
                padded = np.full((dataset.count, patch_size, patch_size), pad_value, dtype=dataset.dtypes[0])
                padded[:, :win_height, :win_width] = data

                # Update profile to reflect full patch size
                profile.update({
                    'width': patch_size,
                    'height': patch_size,
                    'transform': rio.windows.transform(window, dataset.transform)
                })

                patch_filename = f"{outputFolder}/{base_name}_{k_x}_{k_y}.tif"
                with rio.open(patch_filename, 'w', **profile) as dst:
                    dst.write(padded)

    print(f'Finished creating chips for {img_path}.', end='\r')
