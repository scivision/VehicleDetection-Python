#!/usr/bin/env python
import numpy as np
import picamera
import picamera.array
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
import h5py
import os


def main():
    p = ArgumentParser()
    p.add_argument('odir', help='output directory')
    p.add_argument('-r', '--resolution', nargs=2, type=int, default=(640, 480))
    p.add_argument('-fps', type=int, default=30)
    p = p.parse_args()

    if p.odir:
        outdir = Path(p.odir).expanduser() / datetime.now().isoformat()[:-10]
        vidfn = outdir/'raw.mp4'
        motfn = outdir/'motion.h5'
        print('saving', p.resolution, 'to', outdir)  # 'motion:', motfn, 'video:', vidfn)
        outdir.mkdir(exist_ok=True)
    else:
        vidfn = os.devnull

    res = p.resolution

    with picamera.PiCamera() as camera:
        with picamera.array.PiMotionArray(camera) as stream:
            camera.resolution = res
            camera.framerate = p.fps
            camera.start_recording(str(vidfn), format='h264', motion_output=stream)
            camera.wait_recording(120)
            camera.stop_recording()

            with h5py.File(motfn, 'w') as f:
                imgs = stream.array
                f['motion'] = np.hypot(imgs['x'], imgs['y'])


if __name__ == '__main__':
    main()
