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
    p.add_argument('mthres', help='motion threshold', type=int)
    p.add_argument('-o', '--outfn', help='write blob count stem')
    p.add_argument('-v', '--verbose', help='show debug plots', action='store_true')
    p.add_argument('-s', help='preview save name')
    p = p.parse_args()

    verbose = p.verbose

    fn = Path(p.infn).expanduser()

    with h5py.File(fn, 'r') as f:
        mot = np.rot90(f['motion'][1025:, ...].astype(np.uint8), axes=(1, 2))

    bmot = mot > p.mthres

    countfn = fn.parent/p.outfn if p.outfn else None
    if verbose:
        fg = figure(figsize=(8, 8))
        ax1, ax2 = fg.subplots(2, 1)
        fg.suptitle('FFT-based approach - no OpenCV')
    else:
        ax1 = ax2 = None

    Ncount = []
    for i, m in enumerate(bmot):
        if verbose:
            ax1.cla()
            ax1.imshow(mot[i], origin='upper')
            ax1.set_title('h.264 difference frames')

        N = spatial_discrim(m, ax1, ax2)

        Ncount.append(N)

        if countfn is not None and i and not i % 500:
            countfn = fn.parent/(p.outfn + datetime.now().isoformat() + '.h5')
            with h5py.File(countfn, 'w') as f:
                f['count'] = Ncount
                f['index'] = i
            Ncount = []

        if p.s:
            fg.savefig(p.s+f'{i:04d}.png', bbox_inches='tight', dpi=100)


def spatial_discrim(mot: np.ndarray, ax1=None, ax2=None) -> int:
    """
    rectangular LPF in effect
    """
    MIN = 500
    MAX = np.inf

    ilanes = [(25, 27),
              (35, 40)]

    lane1 = mot[ilanes[0][0]:ilanes[0][1], :].sum(axis=0)
    lane2 = mot[ilanes[1][0]:ilanes[1][1], :].sum(axis=0)

    L = lane1.size
    iLPF = (int(L*4/9), int(L*5.2/9))

    Flane1 = np.fft.fftshift(abs(np.fft.fft(lane1))**2)
    Flane2 = np.fft.fftshift(abs(np.fft.fft(lane2))**2)

    N1 = int(MIN <= Flane1[iLPF[0]:iLPF[1]].sum() <= MAX)
    N2 = int(MIN <= Flane2[iLPF[0]:iLPF[1]].sum() <= MAX)

    if ax2 is not None:
        ax2.cla()
        fx = range(-L//2, L//2)
        ax2.plot(fx, Flane1)
        ax2.plot(fx, Flane2)
        ax2.set_title(f'lane counts {N1} {N2}')
        ax2.set_ylim(0, 1000)
        ax2.set_xlabel('Spatial Frequency bin (arbitrary units)')
        ax2.set_ylabel('magnitude$^2$')
        # indicate LPF bounds
        ax2.axvline(iLPF[0]-L//2, color='red', linestyle='--')
        ax2.axvline(iLPF[1]-L//2, color='red', linestyle='--')
        # indidate lanes
        ax1.axhline(ilanes[0][0], color='cyan', linestyle='--')
        ax1.axhline(ilanes[0][1], color='cyan', linestyle='--')
        ax1.axhline(ilanes[1][0], color='orange', linestyle='--')
        ax1.axhline(ilanes[1][1], color='orange', linestyle='--')

        draw()
        pause(0.01)

    return N1 + N2


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
