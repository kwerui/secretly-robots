"""Calibrate a webcam/phone camera with a ChArUco board (needed for the real carer demo).

1. Print the board:   python calibrate_camera.py board --out board.png   (print at 100%, A4)
2. Measure one printed square edge in metres and pass it as --square.
3. Capture:           python calibrate_camera.py calib --source 0 --square 0.03 --out calib.npz
   Move the board around the view (tilt it, near/far). Press SPACE to keep a frame
   (aim for 15-25), q to finish.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)  # separate from the scene's 4x4 markers
SQUARES = (5, 7)


def board(square: float = 0.03, marker: float = 0.022) -> cv2.aruco.CharucoBoard:
    return cv2.aruco.CharucoBoard(SQUARES, square, marker, DICT)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board")
    b.add_argument("--out", default="board.png")
    c = sub.add_parser("calib")
    c.add_argument("--source", default="0")
    c.add_argument("--square", type=float, required=True, help="printed square edge in metres")
    c.add_argument("--out", default="calib.npz")
    args = ap.parse_args()

    if args.cmd == "board":
        img = board().generateImage((1400, 1980), marginSize=40)
        cv2.imwrite(args.out, img)
        print(f"wrote {args.out}; print at 100% scale and measure a square")
        return

    brd = board(args.square, args.square * 0.73)
    detector = cv2.aruco.CharucoDetector(brd)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    obj_pts, img_pts, size = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners, ids, _, _ = detector.detectBoard(gray)
        view = frame.copy()
        if ids is not None and len(ids) > 6:
            cv2.aruco.drawDetectedCornersCharuco(view, corners, ids)
        cv2.putText(view, f"kept {len(obj_pts)}  (SPACE keep, q done)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and ids is not None and len(ids) > 6:
            o, i = brd.matchImagePoints(corners, ids)
            obj_pts.append(o)
            img_pts.append(i)
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(obj_pts) < 8:
        raise SystemExit("need at least 8 good frames")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    np.savez(args.out, K=K, dist=dist, rms=rms, size=size)
    print(f"RMS reprojection error {rms:.3f} px -> {args.out}")


if __name__ == "__main__":
    main()
