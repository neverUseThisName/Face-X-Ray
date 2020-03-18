'''Face swap demo described in face X-Ray paper

@author Zhuolin Fu

key requirement: numpy, scikit-image, dlib, tqdm, color_transfer.

steps:
    1. input: source face image (I_B in paper, one image file or directory of images) and a directory of real face images as face database.
    2. search face database for the one whose landmarks are close to source face image.
    3. apply convex hull, random deform, color correction (to be added) and swap.
    4. save result in ./dump
'''

import argparse, sys, os
from os.path import basename, splitext
from PIL import Image
from functools import partial

from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import dlib
from tqdm import tqdm

from color_transfer import color_transfer
from utils import files, FACIAL_LANDMARKS_IDXS, shape_to_np

def main():
    args = get_parser()

    # source faces
    srcFaces = tqdm(files(args.srcFacePath, ['.jpg']))

    # real faces database
    #ds = image2pilBatch(files(args.faceDatabase, ['.jpg']))

    # face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shapePredictor)

    
    for i, srcFace in enumerate(srcFaces):
        # load bgr
        try:
            srcFaceBgr = cv2.imread(srcFace)
        except:
            tqdm.write(f'Fail loading: {srcFace}')
            continue

        # detect landmarks
        srcLms = get_landmarks(detector, predictor, cv2.cvtColor(srcFaceBgr, cv2.COLOR_BGR2RGB))
        if srcLms is None:
            tqdm.write(f'No face: {srcFace}')
            continue

        # find first face whose landmarks are close enough in real face database
        targetRgb = find_one_neighbor(detector, predictor, srcFace, srcLms, files(args.faceDatabase, ['.jpg']), args.threshold)
        if targetRgb is None: # if not found
            tqdm.write(f'No Match: {srcFace}')
            continue

        # if found
        targetBgr = cv2.cvtColor(targetRgb, cv2.COLOR_RGB2BGR)
        hullMask = convex_hull(srcFaceBgr.shape, srcLms) # size (h, w, c) mask of face convex hull

        # generate random deform
        anchors, deformedAnchors = random_deform(hullMask.shape[:2], 4, 4)

        # piecewise affine transform and blur
        warped = piecewise_affine_transform(hullMask, anchors, deformedAnchors) # size (h, w) warped mask
        blured = cv2.GaussianBlur(warped, (5,5), 3)

        # swap
        left, up, right, bot = min(srcLms[:,0]), min(srcLms[:,1]), max(srcLms[:,0]), max(srcLms[:,1])
        targetBgrT = color_transfer(srcFaceBgr[up:bot,left:right,:], targetBgr)
        resultantFace = forge(srcFaceBgr, targetBgrT, blured) # forged face

        # save face images
        cv2.imwrite(f'./dump/mask_{i}.jpg', hullMask)
        cv2.imwrite(f'./dump/deformed_{i}.jpg', warped*255)
        cv2.imwrite(f'./dump/blured_{i}.jpg', blured*255)
        cv2.imwrite(f'./dump/src_{i}.jpg', srcFaceBgr)
        cv2.imwrite(f'./dump/target_{i}.jpg', targetBgr)
        cv2.imwrite(f'./dump/target_T_{i}.jpg', targetBgrT)
        cv2.imwrite(f'./dump/forge_{i}.jpg', resultantFace)


def get_landmarks(detector, predictor, rgb):
    # first get bounding box (dlib.rectangle class) of face.
    boxes = detector(rgb, 1)
    for box in boxes:
        landmarks = shape_to_np(predictor(rgb, box=box))
        break
    else:
        return None
    return landmarks.astype(np.int32)


def find_one_neighbor(detector, predictor, srcPath, srcLms, faceDatabase, threshold):
    for face in faceDatabase:
        rgb = dlib.load_rgb_image(face)
        landmarks = get_landmarks(detector, predictor, rgb)
        if landmarks is None:
            continue
        dist = distance(srcLms, landmarks)
        if dist < threshold and basename(face).split('_')[0] != basename(srcPath).split('_')[0]:
            return rgb
    return None


def forge(srcRgb, targetRgb, mask):
    #mask = np.dstack([mask]*3)
    return (mask * targetRgb + (1 - mask) * srcRgb).astype(np.uint8)


def convex_hull(size, points, fillColor=(255,)*3):
    mask = np.zeros(size, dtype=np.uint8) # mask has the same depth as input image
    points = cv2.convexHull(np.array(points))
    corners = np.expand_dims(points, axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped


def distance(lms1, lms2):
    return np.linalg.norm(lms1 - lms2)


def get_parser():
    parser = argparse.ArgumentParser(description='Demo for face x-ray fake sample generation')
    parser.add_argument('--srcFacePath', '-sfp', type=str)
    parser.add_argument('--faceDatabase', '-fd', type=str)
    parser.add_argument('--threshold', '-t', type=float, default=25, help='threshold for facial landmarks distance')
    parser.add_argument('--shapePredictor', '-sp', type=str, default='./shape_predictor_68_face_landmarks.dat', help='Path to dlib facial landmark predictor model')
    return parser.parse_args()


if __name__ == '__main__':
    main()
