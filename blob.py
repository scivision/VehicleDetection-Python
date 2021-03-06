#!/usr/bin/env python
import h5py
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import skimage.feature as skif
from matplotlib.pyplot import figure, draw, pause
import cv2
from typing import Tuple
from datetime import datetime

P = {"mindist": 3, "minarea": 10, "maxarea": 1000}


def main():
    p = ArgumentParser()
    p.add_argument("infn", help="HDF5 motion file to analyze")
    p.add_argument("-cv2", help="use openCV", action="store_true")
    p.add_argument("-wvid", help="write video output filename")
    p.add_argument("-wcount", help="write blob count stem", default="counts")
    p = p.parse_args()

    writevid = p.wvid is not None
    fn = Path(p.infn).expanduser()

    with h5py.File(fn, "r") as f:
        mot = np.rot90(f["motion"][1025:, ...].astype(np.uint8), axes=(1, 2))

    bmot = mot > 15

    B = None
    ax = None
    hv = None
    if not p.cv2:
        ax = figure(1).gca()
    else:
        B = setupblob(P)
        bmot = bmot.astype(np.uint8) * 255
        if writevid:
            fourcc = cv2.VideoWriter_fourcc(*"FFV1")
            hv = cv2.VideoWriter(str(fn.parent / p.wvid), fourcc, fps=10, frameSize=(30, 41))

    countfn = fn.parent / p.wcount
    Ncount = []
    for i, m in enumerate(bmot):

        if ax is not None:
            ax.cla()
            ax.imshow(bmot[i], origin="bottom")

            sblob(m, ax)
        else:
            final, N = cv2blob(mot[i] * 2, m, B)
            cv2.imshow("result", final)
            if writevid:
                hv.write(final)

            cv2.waitKey(1)

        Ncount.append(N)

        if i and not i % 500:
            countfn = fn.parent / (p.wcount + datetime.now().isoformat()[:-7] + ".h5")
            with h5py.File(countfn, "w") as f:
                f["count"] = Ncount
                f["index"] = i
            Ncount = []

    if ax is None:
        if hv:
            hv.release()
        cv2.destroyAllWindows()


def cv2blob(img: np.ndarray, motion: np.ndarray, B) -> Tuple[np.ndarray, int]:
    keypoints = B.detect(motion)
    nkey = len(keypoints)
    # kpsize = [k.size for k in keypoints]
    final = img.copy()  # is the .copy necessary?

    final = cv2.drawKeypoints(
        img, keypoints, outImage=final, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    # %% plot count of blobs
    cv2.putText(
        final,
        text=str(nkey),
        org=(int(img.shape[1] * 0.7), int(img.shape[0] * 0.3)),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=1,
    )

    return final, nkey


def setupblob(param: dict):
    """
    setup connected components "blob detection"
    """
    B = cv2.SimpleBlobDetector_Params()
    B.filterByArea = True
    B.filterByColor = False
    B.filterByCircularity = False
    B.filterByInertia = False
    B.filterByConvexity = False

    B.minDistBetweenBlobs = param["mindist"]
    B.minArea = param["minarea"]
    B.maxArea = param["maxarea"]

    # B.minThreshold = 1  # we have already made a binary image

    return cv2.SimpleBlobDetector_create(B)


def sblob(mot, ax):

    blobs = skif.blob_doh(mot, min_sigma=5, max_sigma=10000, num_sigma=20, overlap=0.1)
    # blobs = skif.blob_dog(m)
    # blobs = skif.blob_log(m, min_sigma=10, max_sigma=300, num_sigma=10, threshold=.1, overlap=1)

    good = (
        (blobs[:, 0] > 5)
        & (blobs[:, 1] > 5)
        & (blobs[:, 0] < mot.shape[0] - 5)
        & (blobs[:, 1] < mot.shape[1] - 5)
    )
    blobs = blobs[good, ...]
    for b in blobs:
        ax.scatter(b[1], b[0], b[2] * 100)

    # if blobs.size > 0:
    #    print(f'frame {i}, {blobs.shape[0]} blobs, max sigma {blobs[:,2].max()}')

    draw()
    pause(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
