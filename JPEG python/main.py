import os
import glob
import time
import numpy as np
import cv2
from PIL import Image
from SI_MiPOD_fastlog import SI_MiPOD_fastlog
from SI_MiPOD_v0 import SI_MiPODv0
# from SI_EBS import SI_EBS
# from SI_UNIWARD import SI_UNIWARD

def extract_quant_tables(jpg_file):
    with Image.open(jpg_file) as img:
        if 'dpi' in img.info:
            del img.info['dpi']  # Remove dpi information to avoid warning
        img.save('temp.jpg', "JPEG", quality=100, optimize=True)
    
    with Image.open('temp.jpg') as img:
        quant_tables = img.quantization
    
    os.remove('temp.jpg')
    return quant_tables

data_dir = "./JPEG python/data"

# Dry-run for fair execution time comparisons
dummyJPEGstruct = {'quant_tables': extract_quant_tables(os.path.join(data_dir, 'ALASKA_50774_QF75.jpg'))}
preCover = cv2.imread(os.path.join(data_dir, 'ALASKA_50774.tif'), flags=cv2.IMREAD_UNCHANGED)
S_STRUCT, C_STRUCT, pChange, ChangeRate, Deflection = SI_MiPOD_fastlog(preCover, dummyJPEGstruct, 0.2)

imgList = glob.glob(os.path.join(data_dir, '*.jpg'))
for imgIdx, imgPath in enumerate(imgList, start=1):
    print(f'\n ***** Processing image {os.path.basename(imgPath)} *****')
    Payload = 0.4

    # Read JPEG struct from JPEG file (only to get a blank struct with Qtables)
    dummyJPEGstruct = {'quant_tables': extract_quant_tables(imgPath)}

    # Read precover images ... whose name is related with the JPEG file ...
    posFS = os.path.basename(imgPath).find('_', os.path.basename(imgPath).find('_') + 1)
    preCoverPath = os.path.join(os.path.dirname(imgPath), os.path.basename(imgPath)[:posFS] + '.tif')
    preCover = cv2.imread(preCoverPath, flags=cv2.IMREAD_UNCHANGED)

    # Get stego DCT coefficients (and Deflection, pChanges and overall ChangeRate)
    tStart = time.time()
    stegoStruct, coverStruct, pChange, ChangeRate, Deflection = SI_MiPOD_fastlog(preCover, dummyJPEGstruct, Payload)
    tEnd = time.time()

    cv2.imwrite(os.path.join('/results', os.path.basename(imgPath)), stegoStruct['coef_arrays'][0])
    cv2.imwrite(os.path.join('/results', os.path.basename(imgPath)[:posFS] + '_cover.jpg'), coverStruct['coef_arrays'][0])

    StegoDCT = stegoStruct['coef_arrays'][0]
    nbnzAC = np.sum(StegoDCT != 0) - np.sum(StegoDCT[::8, ::8] != 0)
    HNats = -pChange.flatten() * np.log(pChange.flatten()) - (1 - pChange.flatten()) * np.log(1 - pChange.flatten())
    Hbits = -pChange.flatten() * np.log2(pChange.flatten()) - (1 - pChange.flatten()) * np.log2(1 - pChange.flatten())

    print(f"\t\t\t\t\t\t\t\t Target payload = {Payload * nbnzAC:5.2f} bits")
    print(f"SI-MiPOD fastlog runs in {tEnd - tStart:2.3f} sec. \t\t\t\tActual payload : {np.nansum(Hbits):5.2f} bits = {np.nansum(HNats):5.2f} Nats (ternary entropy computed from pChanges)")

    tStart = time.time()
    stegoStruct, coverStruct, pChange, ChangeRate, Deflection = SI_MiPODv0(preCover, dummyJPEGstruct, Payload)
    tEnd = time.time()
    print(f"SI-MiPOD original (not so fast) runs in {tEnd - tStart:2.3f} sec.\t\tActual payload : {np.nansum(-pChange.flatten() * np.log2(pChange.flatten()) - (1 - pChange.flatten()) * np.log2(1 - pChange.flatten())):5.2f} bits = {np.nansum(-pChange.flatten() * np.log(pChange.flatten()) - (1 - pChange.flatten()) * np.log(1 - pChange.flatten())):5.2f} Nats (ternary entropy computed from pChanges)")

    # tStart = time.time()
    # stegoStruct, coverStruct, pChange = SI_EBS(preCover, dummyJPEGstruct, Payload)
    # tEnd = time.time()
    # print(f"SI-EBS runs in {tEnd - tStart:2.3f} sec. \t\t\t\t\tActual payload : {np.nansum(-pChange.flatten() * np.log2(pChange.flatten()) - (1 - pChange.flatten()) * np.log2(1 - pChange.flatten())):5.2f} bits = {np.nansum(-pChange.flatten() * np.log(pChange.flatten()) - (1 - pChange.flatten()) * np.log(1 - pChange.flatten())):5.2f} Nats (ternary entropy computed from pChanges)")

    # tStart = time.time()
    # stegoStruct, coverStruct, pChange = SI_UNIWARD(preCover, dummyJPEGstruct, Payload)
    # tEnd = time.time()
    # print(f"SI-UNIWARD runs in {tEnd - tStart:2.3f}. \t\t\t\t\tActual payload : {np.nansum(-pChange.flatten() * np.log2(pChange.flatten()) - (1 - pChange.flatten()) * np.log2(1 - pChange.flatten())):5.2f} bits = {np.nansum(-pChange.flatten() * np.log(pChange.flatten()) - (1 - pChange.flatten()) * np.log(1 - pChange.flatten())):5.2f} Nats (ternary entropy computed from pChanges)")