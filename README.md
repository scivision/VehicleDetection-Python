# Vehicle Detection with OpenCV

Simple demo of vehicle detection with OpenCV cascade classifier.
Tested with [OpenCV 3](https://www.scivision.co/install-opencv-python-windows/)


## Install
Typically we install the latest OpenCV in Python 3 with:
```sh
pip install opencv-python
```
which has binary wheels (no-compile install) for most platforms.


## Usage

```sh
python3 vehdet.py data/video.avi -q
```

Omitting the `-q` allows live display, which is much slower--particularly on embedded systems.

## Notes

[data source](https://github.com/shaanhk/New-GithubTest)

### X11 Virtual Frame Buffer
On systems without an X11 server, a virtual frame buffer can be used.
This method is used in general where a program wants a display but doesn't have a video server.
```sh
apt install xvfb
```

then create a file `~/xvfb.sh`:
```bash
#!/bin/bash
Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
```

and run that file when a program needs a display on a display-less system.

### Libraries

On certain Linux platforms, you may need to install shared libraries:
```sh
apt install libwebp-dev libtiff-dev libjasper-dev libilmbase-dev libopenexr-dev libgstreamer1.0-dev libavcodec-dev libavformat-dev libswscale-dev libqtgui4 libqt4-test
```
to resolve errors including:

> ImportError: libwebp.so.6: cannot open shared object file: No such file or directory

> ImportError: libtiff.so.5: cannot open shared object file: No such file or directory

> ImportError: libjasper.so.1: cannot open shared object file: No such file or directory

> ImportError: libImath-2_2.so.12: cannot open shared object file: No such file or directory

> ImportError: libIlmImf-2_2.so.22: cannot open shared object file: No such file or directory

> ImportError: libgstbase-1.0.so.0: cannot open shared object file: No such file or directory

> ImportError: libavcodec.so.57: cannot open shared object file: No such file or directory

> ImportError: libavformat.so.57: cannot open shared object file: No such file or directory

> ImportError: libswscale.so.4: cannot open shared object file: No such file or directory

> ImportError: libQtGui.so.4: cannot open shared object file: No such file or directory

> ImportError: libQtTest.so.4: cannot open shared object file: No such file or directory

