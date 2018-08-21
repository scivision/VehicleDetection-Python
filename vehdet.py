#!/usr/bin/env python
"""
OpenCV Python program to detect cars in video frame

https://docs.opencv.org/3.4.2/d1/de5/classcv_1_1CascadeClassifier.html
"""
import cv2
from pathlib import Path
from argparse import ArgumentParser
from time import time


def main():
    p = ArgumentParser()
    p.add_argument('fn', help='filename or camera device handle')
    p.add_argument('-train', help='training data', default='data/cars.xml')
    p.add_argument('-q', '--quiet', help='do not show video (for embedded systems)', action='store_true')
    p = p.parse_args()

    verbose = not p.quiet

    trainfn = Path(p.train).expanduser()
    if not trainfn.is_file():
        raise FileNotFoundError(str(trainfn))

    if not p.fn.startswith('/dev'):
        fn = Path(p.fn).expanduser()
        if not fn.is_file():
            raise FileNotFoundError(str(fn))
    # capture frames from a video
    cap = cv2.VideoCapture(str(fn))

    # Trained XML classifiers describes some features of some object we want to detect
    car_cascade = cv2.CascadeClassifier(str(trainfn))

    tic = time()
    i = 0
    while True:
        i += 1
        # reads frames from a video
        ret, frames = cap.read()
        if not ret:
            break

        # convert to gray scale of each frames
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        # Detects cars of different sizes in the input image
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        if verbose:
            for (x, y, w, h) in cars:
                cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 0, 255), 2)

            cv2.imshow('video2', frames)

        # Wait for Esc key to stop
        if cv2.waitKey(1) == 27:
            print('user aborted.')
            break

    if verbose:
        cv2.destroyAllWindows()

    print(fn, 'stream ended, {:.1f} ms / frame'.format(((time()-tic)*1000)/i))


if __name__ == '__main__':
    main()
