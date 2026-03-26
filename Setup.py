from pathlib import Path
import requests
import os
import sys
import zipfile
from IPython.utils import io
import shutil
import subprocess
import shlex
import numpy as np
from importlib import reload
import urllib.parse
import rioxarray as riox
from rasterio.warp import Resampling
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import helpers
import Helpers.Chip as Chip
reload(Chip)

# Function to download and unzip the model weights
def download(url, savePath):
    response = requests.get(url)
    with open(savePath, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved to {savePath}.")
    
def unzip(savePath):
    unzipPath=savePath.parent
    with zipfile.ZipFile(savePath, 'r') as zip_ref:
        zip_ref.extractall(unzipPath)
        print(f"File extracted to {unzipPath}.")
    os.remove(savePath)

# Function to download the data from a LAADS-DAAC order
def downloadOrder(orderNumber, outputPath, apikey, suppress=True):
    
    # Clear existing
    for item in outputPath.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    
    # Make download request
    archiveURL = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/{orderNumber}/"
    access = f"Authorization: Bearer {apikey}"
    command = f'wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3 {archiveURL} --header "{access}" -P {outputPath}'
    try:
        # Split the command into shell arguments using shlex
        args = shlex.split(command)
        if suppress:
            with open('/dev/null', 'w') as devnull:
                subprocess.run(args, check=True, stdout=devnull, stderr=devnull)
        else:
            subprocess.run(args, check=True)
        print(f'Downloaded order {orderNumber} to {outputPath}.')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading order: {e}")

    # Count downloaded files
    nfiles = len([item for item in outputPath.iterdir() if item.is_file()])
    if nfiles>0:
        print(f'Downloaded {nfiles} files.')
    else:
        print(f'Failed to download any files: *stopping*')
        sys.exit(1)


# Resample the M bands to 750 m resolution (provided at 375 m), and correct for the orignal bands not being exactly 375 m
def resampleOrder(savePath, deleteOriginal=True):

    filesToDelete=[]
    
    # I bands
    files = list(filter(lambda item: item.suffix=='.tif' and 'QF' not in item.stem and '375m' in item.stem and '_375.tif' not in item.name, savePath.iterdir())) 
    for file in files:
        with riox.open_rasterio(file) as source:
            # I bands should be exactly 375 m resolution
            fileOut = source.rio.reproject(source.rio.crs, resolution=(375,375), resampling=Resampling.nearest)
            fileOut.rio.to_raster(Path(str(file).replace('.tif','_375.tif')))
    filesToDelete.extend(files)

    # M bands
    files = list(filter(lambda item: item.suffix=='.tif' and '750m' in item.stem and '_750.tif' not in item.name, savePath.iterdir()))
    for file in files:
        with riox.open_rasterio(file) as source:
            # M bands should be exactly 750 m resolution
            fileOut = source.rio.reproject(source.rio.crs, resolution=(750,750), resampling=Resampling.nearest)
            fileOut.rio.to_raster(Path(str(file).replace('.tif','_750.tif')))
    filesToDelete.extend(files)
    
    # QF bands
    files = list(filter(lambda item: item.suffix=='.tif' and 'QF' in item.stem and '750' not in item.stem and '_375.tif' not in item.name, savePath.iterdir()))
    for file in files:
        with riox.open_rasterio(file) as source:
            # QF bands should be exactly 375 m resolution
            fileOut = source.rio.reproject(source.rio.crs, resolution=(375,375), resampling=Resampling.nearest)
            fileOut.rio.to_raster(Path(str(file).replace('.tif','_375.tif')))
    filesToDelete.extend(files)

    # Optionally delete the orignal data (now that we have the resampled version required)
    if deleteOriginal:
        for item in filesToDelete:
            if item.is_file():
                item.unlink()

# Prepare the input data as required by the inference pipeline
def prepInfInputs(dataPath, inPath):#, deleteOriginal=False):

    # Remove any items that already exist (in case was run prior and failed)
    for item in inPath.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    # Get files and find the tifs containing the required bands: I1, I2, I3, M3, M4, M11, QF1
    files=list(filter(lambda item: item.suffix=='.tif' and 'chipped' not in str(item) and 'Composite' not in str(item), Path(dataPath).glob('**/*.tif')))
    files=list(filter(containsBand, files))
    
    # Get dates if there are multiple dates, in form yyyydoy
    fileDates=np.unique(list(map(lambda item: item.stem.split('.')[1].replace('A',''), files)))
    
    # # Get multiple granules on same day if present
    # if 'mosaic' in str(files[0]):
    #     fileGranules=['mosaic']
    # else:
    #     fileGranules=np.unique(list(map(lambda item: item.stem.split('.')[2], files)))

    # Loop through dates and check band files are present
    bands=['I1', 'I2', 'I3', 'M3', 'M4', 'M11', 'QF1']#, 'QF2']
    inPathDates=[]
    for date in fileDates:

        dateFiles=list(filter(lambda item: item.stem.split('.')[1].replace('A','')==date, files))
        
        # Get multiple granules on same day if present
        if 'mosaic' in str(dateFiles[0]):
            fileGranules=['mosaic']
        else:
            fileGranules=np.unique(list(map(lambda item: item.stem.split('.')[2], dateFiles)))
        
        for granule in fileGranules:
            if granule=='mosaic':
                dateGranuleFiles=list(filter(lambda item: item.stem.split('.')[1].replace('A','')==date, dateFiles))
            else:
                dateGranuleFiles=list(filter(lambda item: item.stem.split('.')[1].replace('A','')==date and item.stem.split('.')[2]==granule, dateFiles))
        
            # Check all bands are present, including QF1 band used for cloud mask
            bandFiles=[]
            for band in bands:
                bandFile=list(filter(lambda item: band in str(item), dateGranuleFiles))
                if len(bandFile)>1:
                    # If QF1, we expect two files since at 375m and 750m res, we just want to keep the 375m one
                    if 'QF' in str(bandFile[0]):
                        bandFile=[item for item in bandFile if item.stem.split('_')[-1]=='375']
                        bandFiles.append(bandFile[0])
                    # For M band, use only the 750 m res output
                    elif 'M' in band: 
                        bandFile=[item for item in bandFile if item.stem.split('_')[-1]=='750']
                        bandFiles.append(bandFile[0])
                    elif 'I' in band: 
                        bandFile=[item for item in bandFile if item.stem.split('_')[-1]=='375']
                        bandFiles.append(bandFile[0])
                elif len(bandFile)==1:
                    bandFiles.append(bandFile[0])
                else:
                    print(f'No tif file found for band {band} for date {date} for granule {granule}: *stopping*.')
                    sys.exit(1)
                # If still too many files then stop process
                if len(bandFile)>1:
                    print(f'More than 1 tif file found for band {band} for date {date} for granule {granule}: *stopping*.')
                    sys.exit(1)
    
            # Make a copy with the naming convention correct for inference
            if granule=='mosaic':
                inPathDate=inPath/f'{date}'
            else:
                inPathDate=inPath/f'{date}_{granule}'
            inPathDate.mkdir(exist_ok=True, parents=True)
            inPathDates.append(inPathDate)
            for b, bandFile in enumerate(bandFiles):
                newFileName=inPathDate/(bands[b]+'.tif')
                shutil.copy(bandFile, newFileName)
            print(f'Prepared {len(bandFiles)} files for date {date} for granule {granule} saved to {inPathDate}.')

    # if deleteOriginal:
    #     shutil.rmtree(dataPath)

    print(f'Prepared inputs saved at {inPath} for a total of {len(inPathDates)} unique dates and granules.')
        
    return inPath

def containsBand(file):
    bands=['I1', 'I2', 'I3', 'M3', 'M4', 'M11', 'QF1']#, 'QF2']
    return any(band in str(file) for band in bands)

# Function to pre-check that all files are present for inference
def checkInfInputs(inPath):

    # Get files and file dates
    files=list(filter(lambda item: item.suffix=='.tif' and 'chipped' not in str(item) and 'composite' not in str(item), Path(inPath).glob('**/*.tif')))
    fileDates=np.unique(list(map(lambda item: item.parent.stem, files)))

    # Loop through dates
    bandFilesAllDates=[]
    for date in fileDates:
        dateFiles=list(filter(lambda item: item.parent.stem==date, files))

        # Check all bands are present, including QF1 band used for cloud mask
        bands=['I1', 'I2', 'I3', 'M3', 'M4', 'M11', 'QF1']
        bandFiles=[]
        for band in bands:
            bandFile=list(filter(lambda item: band in str(item), dateFiles))
            if len(bandFile)>1:
                print(f'More than 1 tif file found for band {band} for date {date}: *stopping*.')
                sys.exit(1)
            elif len(bandFile)==1:
                bandFiles.append(bandFile[0])
            else:
                print(f'No tif file found for band {band} for date {date}: *stopping*.')
                sys.exit(1)
        bandFilesAllDates.extend(bandFiles)

    print(f'{len(bandFilesAllDates)} of {len(bands)*len(fileDates)} required files present for {len(fileDates)} dates.')

# Create chips from original files
def chipFiles(inPath, chip=True, max_workers=4, pad_value=-28672, buffer=0):
    
    # Get files and file dates
    files=list(filter(lambda item: item.suffix=='.tif' and 'chipped' not in str(item), Path(inPath).glob('**/*.tif')))
    fileDates=np.unique(list(map(lambda item: item.parent.stem, files)))

    # Loop through dates
    chipsFilesForInf=[]
    qf1Files=[]
    for date in fileDates:
        dateFiles=list(filter(lambda item: item.parent.stem==date, files))

        # Create path for chips
        chipsPath = dateFiles[0].parent/'chipped'
        chipsPath.mkdir(parents=True, exist_ok=True)

        if chip:
            # Clear existing
            for item in chipsPath.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

            # Create chips
            tifs375=[item for item in dateFiles if 'I' in item.stem]
            tifs750=[item for item in dateFiles if 'M' in item.stem]       

            # Create chips in parallel
            jobs = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for img in tifs375:
                    jobs.append(executor.submit(chipSingleImage, img, chipsPath, 256, pad_value, buffer))
                for img in tifs750:
                    jobs.append(executor.submit(chipSingleImage, img, chipsPath, 128, pad_value, buffer//2))

                for f in tqdm(as_completed(jobs), total=len(jobs), desc=f"Chipping {date}"):
                    f.result()

        # Get chipped files for infeerences
        # Just looks for I1 files since inference code automatically gets the others from the same folder
        chipsFiles = list(chipsPath.glob('*I1*.tif'))
        chipsFilesForInf.extend(chipsFiles)

        # Record location of QF1 file for cloud mask used post inference
        qf1=[item for item in dateFiles if 'QF1' in item.stem]
        qf1Files.extend(qf1)
        print(f'Found a total of {len(qf1Files)} QF1 files across {len(fileDates)} dates.')

    print(f'Created a total of {len(chipsFilesForInf)} chips across {len(fileDates)} dates.')
    return chipsFilesForInf, qf1Files

# Accompanying function to run chipping for single image, executed in parallel
def chipSingleImage(img, chipsPath, size, pad_value=-28672, buffer=0):
    Chip.CreateChips(Path(img), size, chipsPath, pad_value=pad_value, buffer=buffer)
    return img

# New function to release order using requests *** This needs fixing
def releaseOrder(orderNumber, email):
    base_url = "https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices/releaseOrder"
    params = {
        'email': email,
        'orderId': orderNumber
    }
    releaseLink = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        response = requests.get(releaseLink)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Order {orderNumber} released successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while releasing order: {e}")