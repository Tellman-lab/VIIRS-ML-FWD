import xarray as xr
from rasterio.enums import Resampling
import rasterio as rio
import os
import numpy as np
from rasterio.merge import merge
from rasterio import MemoryFile

# Function which runs the process to mosaic the inferred chips
def MosaicInferredChips(chipsFilesInferred, qf1Files, deleteOriginal=False, maskClouds=False, imageBuffer=0, gradient_method='linear'):
    
    # Get files and file dates
    fileDates=np.unique(list(map(lambda item: item.parents[1].stem, chipsFilesInferred)))

    # Loop through dates
    MosaicFiles=[]
    for date in fileDates:
        dateFiles=list(filter(lambda item: item.parents[1].stem==date, chipsFilesInferred))
        nDateFiles=len(dateFiles)

        targetTifPath=dateFiles[0].parents[1] / f'{date}.tif'
        open_raster_call = lambda f: xr.open_mfdataset(f, engine="rasterio")

        # If using feathering, use mosaicing method accounting for buffer
        if imageBuffer > 0:
            memfile = GetMosaicWithBuffer(dateFiles, imageBuffer, gradient_method)
            open_raster_call = lambda f: xr.open_dataset(memfile, engine='rasterio')

        # Reproject match to the original VIIRS scene including pulling back in the no data pixels
        sourceFile=list(filter(lambda item: item.parent.stem==date, qf1Files))[0]

         # Open the files to merge/mosaic, and open the original source file with correct extent and resolution/alignment
        with open_raster_call(dateFiles) as dataset:
            with xr.open_dataset(sourceFile) as source:
                dataset = dataset.rio.write_crs(source.rio.crs)
                dataset = dataset.rio.reproject_match(source)

                # Apply cloud mask
                if maskClouds:
                    viirsmask = 1 << 3
                    cloudLayer = source.band_data.values.astype('int16') & viirsmask
                    mask = (cloudLayer == 0)
                    dataset = dataset.where(mask)

                dataset.rio.nodata = -28672
                dataset = dataset.fillna(-28672)
       
        # Export
        dataset.rename({'band_data': 'inundation'}).squeeze().rio.to_raster(targetTifPath)
        with rio.open(targetTifPath,'r+') as r:
            r.nodata = -28672
        print(f'Mosaic created from {nDateFiles} files saved to {targetTifPath}.')
        MosaicFiles.append(targetTifPath)
                
    if deleteOriginal:
        for item in chipsFilesInferred:
            os.remove(item)
        os.rmdir(chipsFilesInferred[0].parent)
        print('Deleted chipped inferred intermediate files.')

    return MosaicFiles
            
# Function to mosaic inferred chips with the feathering option
def GetMosaicWithBuffer(dateFiles, imageBuffer, gradient_method):
    mask = GetMask(imageBuffer, gradient_method=gradient_method)
    datasets = [rio.open(fp) for fp in dateFiles]

    # Custom function to mosaic the chips accounting for the buffer and masked areas
    # https://rasterio.readthedocs.io/en/stable/api/rasterio.merge.html
    # merged data: current window of final matrix
    # new_data: new data to be added, same shape as merged_data
    # merged mask: boolean masks where pixels are invalid, same shape as merged_data
    # new_mask: boolean masks where pixels are invalid, same shape as merged_data
    def merge_custom(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None):#, fill_value=0):
        # initialise values with 0 where merged mask is still nodata, and new_mask has data
        merged_data[merged_mask & ~new_mask] = 0.0
        # multiply data by our custom mask
        new_data *= mask
        # fill data in the merged data where data exists
        merged_data[~new_mask] += new_data[~new_mask]


    # Perform the mosaic using the custom function
    mosaic, out_transform = merge(datasets, method=merge_custom)
    out_meta = datasets[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        }
    )

    memfile = MemoryFile()
    with memfile.open(**out_meta) as dataset:
        dataset.write(mosaic)

    for ds in datasets:
        ds.close()

    return memfile

# Function to get the mask of weights to multiply inferred pixel values by for feathering option
def GetMask(imageBuffer, gradient_method='linear'):
    mask = np.ones((256, 256))
    
    if gradient_method == 'linear':
        gradient = np.linspace(0, 1, imageBuffer + 2)[1:-1]
    elif gradient_method == 'sin':
        gradient = (np.sin(np.pi * (np.arange(0, imageBuffer + 1)) / (2 * (imageBuffer + 1))) ** 2)[1:]
    else:
        raise ValueError('Invalid gradient method for feathering.')

    mask[0:imageBuffer, :] = mask[0:imageBuffer, :] * gradient[:, np.newaxis]
    mask[-imageBuffer:, :] = mask[-imageBuffer:, :] * np.flipud(gradient)[:, np.newaxis]
    mask[:, 0:imageBuffer] = mask[:, 0:imageBuffer] * gradient[np.newaxis, :]
    mask[:, -imageBuffer:] = mask[:, -imageBuffer:] * np.flipud(gradient)[np.newaxis, :]
    
    return mask

