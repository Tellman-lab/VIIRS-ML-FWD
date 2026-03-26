from pathlib import Path
from importlib import reload
import os
import sys
import fastai
# print(fastai.__version__)
from fastai.vision.all import *
import argparse

# Import helper functions
import Helpers.Setup as Setup
import Helpers.Chip as Chip
import Helpers.Infer as Infer
import Helpers.Mosaic as Mosaic
reload(Setup)
reload(Chip)
reload(Infer)
reload(Mosaic)

# Get arguments passed at command line
parser = argparse.ArgumentParser()

# Required arguments 
parser.add_argument('--folderNames', type=str, nargs="+") # Names of the folders to infer: if using LAADS DAAC download, folderName must be the LAADS order number

# Optional arguments for setup (with default settings if not specified)
parser.add_argument('--rootPath', default=os.getcwd(), type=str, help='Root path for work, by default is the parent of the Inference.py code')
parser.add_argument('--modelWeightsPath', default='ModelWeights/model', type=str, help='Path/filename to the model weights, defined relative to the root path')
parser.add_argument('--laadsEmail', default='', type=str, help='Your email for NASA LAADS DAAC: only required if using LAADS DAAC download')
parser.add_argument('--laadsApikey', default='', type=str, help='Your API key token for NASA LAADS DAAC: only required if using LAADS DAAC download')

# Optional arguments for processing options (with default settings if not specified)
parser.add_argument('--downloadOrder', default='y', type=str, choices=['y', 'n'], help='Whether to download the data from a LAADS-DAAC submitted order')
parser.add_argument('--prepFiles', default='y', type=str, choices=['y', 'n'], help='Whether to prepare the files by checking the correct number of files are present')
parser.add_argument('--resampleFiles', default='y', type=str, choices=['y', 'n'], help='Whether to resample the data to exactly 375 and 750 m resolutions')
parser.add_argument('--chipFiles', default='y', type=str, choices=['y', 'n'], help='Whether to chip the files into 256 x 256 pixel chips required for inference')
parser.add_argument('--overwriteInf', default='y', type=str, choices=['y', 'n'], help='Whether to overwrite existing inferred chip files')
parser.add_argument('--maskClouds', default='n', type=str, choices=['y', 'n'], help='Whether to mask clouds in the inferrence, using the VIIRS cloud mask, default is no')
parser.add_argument('--mosaicInf', default='y', type=str, choices=['y', 'n'], help='Whether to mosaic the inferred chip files over space')
parser.add_argument('--deleteInfChips', default='n', type=str, choices=['y', 'n'], help='Whether to delete the chipped inferred files after mosaicing, default is no')
parser.add_argument('--device', default='0', type=str, help='Device for inference, e.g. cpu, mps, or gpu devices (numbered) ')
parser.add_argument('--bs', default=64, type=int, help='Batch size for inference')
parser.add_argument('--imageBuffer', default=64, type=int, help='Image buffer to use for feathering option, use 0 for no feathering')
parser.add_argument('--gradientMethod', default='linear', type=str, choices=['linear', 'sin'], help='Method for feathering, either linear or sinusoidal')

# Parse arguments
args = parser.parse_args()
print('\n----------------- Running with arguments -----------------')
print(args)

# Get all the arguments for use
folderNames=args.folderNames
email=args.laadsEmail
apikey=args.laadsApikey
imageBuffer=args.imageBuffer
gradientMethod=args.gradientMethod

# If not specified, get rootpath as parent of path of Inference.py
if Path(args.rootPath).resolve() == Path.cwd():
    try:
        scriptPath = Path(__file__).resolve()
    except NameError:
        # __file__ is not defined in notebooks
        scriptPath = Path.cwd()
    rootPath = scriptPath.parent.parent.resolve()
else:
    rootPath = Path(args.rootPath).resolve()
print(f'Using root path {rootPath}.')

# Get model weights path, by default defined by rootPath but otherwise can be any other valid path
# if args.modelWeightsPath=='ModelWeights/model':
if 'ModelWeights' in args.modelWeightsPath:
    modelPath=rootPath/args.modelWeightsPath
else:
    modelPath=Path(args.modelWeightsPath)
print(f'Using model path/file {modelPath}.')
      
# Boolean arguments for options in functions
download = args.downloadOrder == 'y'
prepare = args.prepFiles == 'y'
resample = args.resampleFiles == 'y'
chip = args.chipFiles == 'y'
overwrite = args.overwriteInf == 'y'
maskClouds = args.maskClouds == 'y'
mosaic = args.mosaicInf == 'y'
deleteInfChips = args.deleteInfChips == 'y'

# Fixed arguments
pad_value=-28672
max_val=16000
norm='gamma'
gamma=0.4


# ---------------- BEGIN LOOP OVER ORDERS ---------------- 

# Loop over all orders and run inference
print('\n----------------- Beginning loop over all folders -----------------')
print(f'Running VIIRS inference for a total of {len(folderNames)} folders.') 
for o, folderName in enumerate(folderNames):
    print('\n---------- Beginning processing ----------')
    print(f'Processing folder {folderName}, {o+1} of total {len(folderNames)} folders.')

    # ---------------- SETUP ----------------
    # Each of the steps can be omitted by providing the relevant argument with n

    # Establish paths
    dataDir, inputsDir, inferredDir,  = 'Data', 'Inputs', 'Inferred'
    dataPath = rootPath/dataDir/folderName
    inputsPath = rootPath/inputsDir/folderName
    inferredRootPath = rootPath/inferredDir
    
    dataPath.mkdir(exist_ok=True, parents=True)
    inputsPath.mkdir(exist_ok=True, parents=True)
    inferredRootPath.mkdir(exist_ok=True, parents=True)

    # If requested the data via LAADS-DAAC, download the data order 
    if download:
        print('\n---- Downloading ----')
        print(f'Downloading data from LAADS-DAAC.')
        Setup.downloadOrder(folderName, dataPath, apikey, suppress=True)        
    
    # Resample the bands to exactly 350 and 750 m resolutions
    if resample:
        print('\n---- Resampling ----')
        print(f'Resampling data to exact required resolutions.')
        Setup.resampleOrder(dataPath, deleteOriginal=True)

    # Prepare the files
    if prepare:
        print('\n---- Preparing files ----')
        print(f'Preparing inputs.')
        Setup.prepInfInputs(dataPath, inputsPath)
    
        print(f'Checking inputs were prepared successfully.')
        Setup.checkInfInputs(inputsPath)
    
    else:
        print(f'Skipping preparing inputs.')

    # Chip the data inputs
    if chip:
        print('\n---- Chipping ----')
        print(f'Creating chips.')
        chipsFilesForInf, qf1Files = Setup.chipFiles(inputsPath, chip=True, max_workers=4, pad_value=pad_value, buffer=imageBuffer)
        
    else:
        # Run chip files but without the chipping, just for the purpose of returning the file names
        print(f'Skipping chipping inputs.')    
        chipsFilesForInf, qf1Files = Setup.chipFiles(inputsPath, chip=False, max_workers=4, pad_value=pad_value, buffer=imageBuffer)
    
    # ---------------- INFERENCE ----------------
    
    # Infer chipped files
    print('\n---- Inference ----')
    print(f'Running inference.')
    chipsFilesInferred = Infer.InferImages(chipsFilesForInf, modelPath, inputsDir, inferredDir, nbCoresDeepLearning=5, nbCoresTifGeneration=10, batchSize=args.bs, device=args.device, max_val=max_val, norm=norm, gamma=gamma, overwrite=overwrite)

    # ---------------- MOSAICING ----------------
    
    # Mosaic inferred chip files
    if mosaic:
        print('\n---- Mosaicing ----')
        print(f'Mosaicing inferred chips.')
        mosaicFiles = Mosaic.MosaicInferredChips(chipsFilesInferred, qf1Files, deleteOriginal=deleteInfChips, maskClouds=maskClouds, imageBuffer=imageBuffer, gradient_method=gradientMethod)
    else:
        print(f'Skipping mosaicing.')    

    print(f'Finished processing folder {folderName}, {o+1} of total {len(folderNames)} folders.')

print('\n----------------- Finished -----------------')


