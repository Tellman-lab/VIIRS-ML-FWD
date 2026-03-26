from pathlib import Path
import rasterio as rio
import numpy as np
import fastai
from fastai.vision.all import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc

# Helpers
from Helpers.Model import LateFusionUNetReg as LateFusionUNetReg
from Helpers.MultiChannel import *

# Fubction to run inference, allowing for different transformations or clip values
def InferImages(items, modelPath, inDir, outDir, nbCoresDeepLearning, nbCoresTifGeneration, batchSize, device, max_val, norm='gamma', gamma=0.4, overwrite=True):

    if device not in ['cpu', 'mps']:
        device = int(device)
        torch.cuda.set_device(device)
        currentDevice=torch.cuda.current_device()
        print(f'Using device {currentDevice}.')
    else:
        torch.device(device)
        print(f'Using {device}.')

    # If do not overwrite, filter items to only items where inference does not already exist
    targetFiles = sorted(map(lambda item: Path(str(item).replace(inDir, outDir)).parent/('_'.join(item.stem.split('_')[1:])+item.suffix), items))
    
    if overwrite:
        itemsToInfer=items.copy()
    else:
        itemsToInfer = []
        for i, targetFile in enumerate(targetFiles):
            if not targetFile.exists():
                itemsToInfer.append(items[i])

    if len(itemsToInfer)==0:
        print(f'Found no files to infer, skipping inference.')
        return targetFiles
    else:
        print(f'Found {len(itemsToInfer)} files to infer remaining from a total of {len(items)} files.')

    
        
    # Create image blocks
    # ImageBlock = MultiChannelImageBlock()
    image_create_fn = make_multi_channel_create(max_val=max_val, norm=norm, gamma=gamma)
    ImageBlock = TransformBlock(type_tfms=image_create_fn)
    MaskBlock = TransformBlock(type_tfms=[lambda x: TensorMask(torch.from_numpy(np.ones((256,256))))])

    # Create data block
    db = DataBlock(blocks=(ImageBlock, MaskBlock),
               get_items = lambda x: itemsToInfer)
    dl = db.dataloaders(Path('.'), num_workers=nbCoresDeepLearning, bs=batchSize)

    # Load model
    print(f'Loading model from: {modelPath}.')
    model = LateFusionUNetReg(6, 1)
    acc_metric = [mae, mse, rmse, R2Score()]
    loss_fn = L1LossFlat()
    learn = Learner(dl, model, loss_func = loss_fn, metrics=acc_metric, opt_func=ranger)
    learn.load(modelPath)
    print(f'Loaded model.')

    # Get predictions and save to raster in batches of max 1000 files at a time
    batch_size_items=1000 #0
    for batch_idx, items_batch in enumerate(chunked(itemsToInfer, batch_size_items)):
        print(f"Inferring batch {batch_idx+1} of {len(itemsToInfer)//batch_size_items + 1}: {len(items_batch)} items.")

        # Get predictions
        test_dl = learn.dls.test_dl(items_batch)
        preds, _ = learn.get_preds(dl=test_dl)
        print(f'Got {len(preds)} predictions.')
    
        # Process predictions to raster outputs
        if nbCoresTifGeneration == 1:
            print(f'Creating inferred rasters.')
            for image_k in range(len(items_batch)):
                processResultAsRaster(image_k, items_batch, inDir, outDir, preds)
        else:
            func_part = partial(processResultAsRaster, items=items_batch, inDir=inDir, outDir=outDir, preds=preds)     
            with ProcessPoolExecutor(max_workers=min(nbCoresTifGeneration, len(items_batch))) as executor:
                futures = [executor.submit(func_part, idx) for idx in range(len(items_batch))]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Creating inferred rasters"):
                    _ = _.result()

        # Free up batch-level memory
        del test_dl, preds
        gc.collect()
        if device not in ['mps', 'cpu']:
            torch.cuda.empty_cache()

    # Return list of inferred files
    targetFilesInferred = sorted(map(lambda item: Path(str(item).replace(inDir, outDir)).parent/('_'.join(item.stem.split('_')[1:])+item.suffix), itemsToInfer))
    print(f'Inferred {len(targetFilesInferred)} files.')
    
    return targetFiles

# Function to process the predictions and save as raster tif file
def processResultAsRaster(k, items, inDir, outDir, preds):
    item = items[k]

    targetFile = Path(str(item).replace(inDir, outDir)).parent/('_'.join(item.stem.split('_')[1:])+item.suffix)
    targetFile.parent.mkdir(exist_ok=True, parents=True)

    with rio.open(item) as r:
        profile = r.profile.copy()
        profile.update(count = 1, nodata = -28672, dtype='float32')
        prediction = preds[k]
        if prediction.ndim == 3:
            prediction = prediction.squeeze(0)
        prediction[r.read(1).squeeze() == -28672] = -28672
        with rio.open(targetFile, 'w', **profile) as dst:
            dst.write(prediction, 1)
                
# Small helper function for running inf in batches
def chunked(files, n):
    for i in range(0, len(files), n):
        yield files[i:i + n]    