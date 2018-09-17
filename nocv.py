#!/usr/bin/env python
import h5py
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from matplotlib.pyplot import figure, draw, pause
from datetime import datetime


def main():
    p = ArgumentParser()
    p.add_argument('infn', help='HDF5 motion file to analyze')
    p.add_argument('-o', '--outfn', help='write blob count stem')
    p.add_argument('-v', '--verbose', help='show debug plots', action='store_true')
    p = p.parse_args()

    verbose = p.verbose

    fn = Path(p.infn).expanduser()

    with h5py.File(fn, 'r') as f:
        mot = np.rot90(f['motion'][1025:, ...].astype(np.uint8), axes=(1, 2))

    bmot = mot > 15

    countfn = fn.parent/p.outfn if p.outfn else None
    if verbose:
        fg = figure()
        ax1, ax2 = fg.subplots(2, 1)
    else:
        ax2 = None

    Ncount = []
    for i, m in enumerate(bmot):
        if verbose:
            ax1.cla()
            ax1.imshow(m, origin='upper')
            ax1.set_title(f'frame {i}')

        N = spatial_discrim(m, i, ax2)

        Ncount.append(N)

        if countfn is not None and i and not i % 500:
            countfn = fn.parent/(p.outfn + datetime.now().isoformat() + '.h5')
            with h5py.File(countfn, 'w') as f:
                f['count'] = Ncount
                f['index'] = i
            Ncount = []


def spatial_discrim(mot: np.ndarray, i: int, ax=None) -> int:
    """
    rectangular LPF in effect
    """
    MIN = 500
    MAX = np.inf

    lane1 = mot[25:27, :].sum(axis=0)
    lane2 = mot[35:40, :].sum(axis=0)

    L = lane1.size
    iLPF = (int(L*4/9), int(L*5.2/9))

    Flane1 = np.fft.fftshift(abs(np.fft.fft(lane1))**2)
    Flane2 = np.fft.fftshift(abs(np.fft.fft(lane2))**2)

    N1 = int(MIN <= Flane1[iLPF[0]:iLPF[1]].sum() <= MAX)
    N2 = int(MIN <= Flane2[iLPF[0]:iLPF[1]].sum() <= MAX)

    if ax is not None:
        ax.cla()
        fx = range(-L//2, L//2)
        ax.plot(fx, Flane1)
        ax.plot(fx, Flane2)
        ax.set_title(f'frame {i}  counts {N1} {N2}')
        ax.set_ylim(0, 1000)
        ax.set_xlabel('Spatial Frequency bin (arbitrary units)')
        ax.set_ylabel('magnitude$^2$')
        ax.axvline(iLPF[0]-L//2, color='red', linestyle='--')
        ax.axvline(iLPF[1]-L//2, color='red', linestyle='--')

        draw()
        pause(0.1)

    return N1 + N2


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
