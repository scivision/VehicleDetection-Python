from typing import List, Tuple
from pathlib import Path
from time import time
import cv2
from datetime import datetime


def carcascade(
    fn: Path, outdir: Path, trainfn: Path, res: Tuple[int, int], verbose: bool = False
) -> List[int]:

    trainfn = Path(trainfn).expanduser()
    if not trainfn.is_file():
        raise FileNotFoundError(str(trainfn))

    if outdir:
        outdir = Path(outdir).expanduser() / datetime.now().isoformat()[:-10]
        print("saving highlighted video previews to", outdir)
        outdir.mkdir()

    assert res[0] > 4 and res[1] > 4

    cap = cv2.VideoCapture(str(fn))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

    res = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(fn, "resolution", res, "pixels")

    counts = cascadeloop(cap, outdir, trainfn, verbose)

    if verbose:
        cv2.destroyAllWindows()

    return counts


def cascadeloop(cap, outdir: Path, trainfn: Path, verbose: bool = False) -> List[int]:
    car_cascade = cv2.CascadeClassifier(str(trainfn))

    counts = []
    tic = time()
    i = 0

    try:
        while True:
            i += 1

            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

            counts.append(len(cars))

            if verbose or outdir:
                for (x, y, w, h) in cars:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if outdir:
                cv2.imwrite(str(outdir / "{:05d}.jpg".format(i)), img)

            if verbose:
                cv2.imshow("video2", img)

                if cv2.waitKey(1) == 27:
                    print("user aborted.")
                    break

            print("image", i, len(cars), "cars.")
    except KeyboardInterrupt:
        print("user aborted.")

    print("stream ended, {:.1f} ms / frame".format(((time() - tic) * 1000) / i))

    return counts
