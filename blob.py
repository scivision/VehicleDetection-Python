#!/usr/bin/env python
import h5py
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import skimage.feature as skif
from matplotlib.pyplot import figure, draw, pause

def main():
    p = ArgumentParser()
    p.add_argument('infn', help='HDF5 motion file to analyze')
    p = p.parse_args()
    
    fn = Path(p.infn).expanduser()
    
    with h5py.File(fn, 'r') as f:
        mot = f['motion'][:].astype(np.uint8)
        
    bmot = mot > 15
        
    ax = figure(1).gca()
    for i,m in enumerate(bmot):
        ax.cla()
        ax.imshow(bmot[i], origin='bottom')
        
        blobs = skif.blob_doh(m, min_sigma=5, max_sigma=10000, num_sigma=20, overlap=0.1)
        # blobs = skif.blob_dog(m)
        #blobs = skif.blob_log(m, min_sigma=10, max_sigma=300, num_sigma=10, threshold=.1, overlap=1)
        
        good = (blobs[:,0]>5) & (blobs[:,1]>5) & (blobs[:,0] < mot.shape[0]-5) & (blobs[:,1] < mot.shape[1]-5)
        blobs = blobs[good, ...]
        for b in blobs:
            ax.scatter(b[1], b[0], b[2]*100)
            
        #if blobs.size > 0:
        #    print(f'frame {i}, {blobs.shape[0]} blobs, max sigma {blobs[:,2].max()}')

        
        draw()
        pause(0.05)
        

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass