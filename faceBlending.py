'''Face swap demo described in face X-Ray paper

This demo is under developing.

key requirement: torch, torchvision, scikit-image, facenet_pytorch (all can be pip installed).

Note: facenet_pytorch can only detect 5 points landmarks. So I'm looking for a substitute.

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
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import cv2
from torchvision import datasets, models, transforms
import dlib
from tqdm import tqdm

from color_transfer import color_transfer
from facenet_pytorch import MTCNN
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
        # load rgb
        srcFaceRgb = dlib.load_rgb_image(srcFace)

        # detect landmarks
        srcLms = get_landmarks(detector, predictor, srcFaceRgb)
        if srcLms is None:
            tqdm.write(f'No face: {srcFace}')
            continue

        # find first face whose landmarks are close enough in real face database
        targetRgb = find_one_neighbor(detector, predictor, srcFace, srcLms, files(args.faceDatabase, ['.jpg']), args.threshold)
        if targetRgb is None: # if not found
            continue

        # if found
        hullMask = convex_hull(srcFaceRgb.shape, srcLms) # size (h, w, c) mask of face convex hull

        # generate random deform
        anchors, deformedAnchors = random_deform(hullMask.shape[:2], 4, 4)

        # piecewise affine transform
        warped = piecewise_affine_transform(hullMask, anchors, deformedAnchors) # size (h, w) warped mask
        blured = cv2.GaussianBlur(warped, (5,5), 3)

        # swap
        targetRgbT = color_transfer(srcFaceRgb, targetRgb)
        resultantFace = forge(srcFaceRgb, targetRgbT, blured) # forged face

        # save face images
        cv2.imwrite(f'./dump/mask_{i}.jpg', hullMask)
        cv2.imwrite(f'./dump/deformed_{i}.jpg', warped*255)
        cv2.imwrite(f'./dump/blured_{i}.jpg', blured*255)
        cv2.imwrite(f'./dump/src_{i}.jpg', cv2.cvtColor(srcFaceRgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./dump/target_{i}.jpg', cv2.cvtColor(targetRgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./dump/target_T_{i}.jpg', cv2.cvtColor(targetRgbT, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./dump/forge_{i}.jpg', cv2.cvtColor(resultantFace, cv2.COLOR_RGB2BGR))


def get_landmarks(detector, predictor, rgb):
    # first get bounding box (dlib.rectangle class) of face.
    boxes = detector(rgb, 1)
    # print(type(boxes))
    # for box in boxes:
    #     print(f'{type(box)}: {box}')
    # input()
    for box in boxes:
        landmarks = shape_to_np(predictor(rgb, box=box))
        break
    else:
        return None
    #jawStart, jawEnd = FACIAL_LANDMARKS_IDXS['jaw']
    #contour = landmarks[jawStart:jawEnd]
    return landmarks[:27].astype(np.int32)


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
    e.g. where nrows = 4, ncols = 5
    ________*_____________________________________
    |                                            |
    |                                            |
    |       *      *     *      *      *         |
    |                                            |
    |       *      *     *      *      *         |
    |                                            |
    |       *      *     *      *      *         |
    |                                            |
    |       *      *     *      *      *         |
    |                                            |
    ______________________________________________

    '''
    h, w = imageSize
    rows = np.linspace(0, h, nrows).astype(np.int32)
    cols = np.linspace(0, w, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors #+ np.random.normal(mean, std, size=anchors.shape)
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