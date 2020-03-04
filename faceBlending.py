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

from facenet_pytorch import MTCNN
from utils import files

def main():
    args = get_parser()

    # source faces
    srcFaces = images2pilBatch(files(args.srcFacePath, ['.jpg']), 1)

    # real faces database
    #ds = image2pilBatch(files(args.faceDatabase, ['.jpg']))

    # face detector
    detector = MTCNN(select_largest=False)

    
    for i, (srcPath, srcFace) in enumerate(srcFaces):
        # detect landmarks for source face (background face in paper)
        _, _, srcLms = get_landmarks(detector, srcFace)
        srcLms = srcLms[0].astype(np.int32)

        # find first face whose landmarks are close enough in real face database
        targetFace = find_one_neighbor(detector, srcPath, images2pilBatch(files(args.faceDatabase, ['.jpg']), 10), args.threshold)
        if targetFace is None: # if not found
            continue

        # if found
        hullMask = convex_hull(srcFace.size, srcLms) # size (h, w) mask of face convex hull

        # generate random deform
        deformedLms = random_deform(srcFace.size, srcLms)
        warped = piecewise_affine_transform(hullMask, srcLms, deformedLms) # size (h, w) warped mask
        resultantFace = forge(srcFace, targetFace, warped) # forged face

        # save face images
        srcFace.save(f'./dump/src_{i}.jpg')
        targetFace.save(f'./dump/target_{i}.jpg')
        cv2.imwrite(f'./dump/forge_{i}.jpg', cv2.cvtColor(resultantFace, cv2.COLOR_RGB2BGR))


def get_landmarks(detector, images):
    # check for detector type
    if isinstance(detector, MTCNN):
        detect = partial(detector.detect, landmarks=True)
    return detect(images)


def find_one_neighbor(detector, srcPath, dl, threshold):
    _, _, srcLms = get_landmarks(detector, Image.open(srcPath))
    srcLms = srcLms[0]
    for pathBatch, pilBatch in dl:
        _, _, landmarksBatch = get_landmarks(detector, pilBatch)
        distanceBatch = [distance(srcLms, lms) for lms in landmarksBatch]
        hitPilBatch = [pil for path, pil, dis in zip(pathBatch, pilBatch, distanceBatch) if dis < threshold and basename(path).split('_')[0]!=basename(srcPath).split('_')[0]]
        if hitPilBatch:
            return hitPilBatch[0]
    return None


def forge(srcFace, targetFace, mask):
    # get pixel values
    if isinstance(srcFace, Image.Image): # if input is pil image
        srcFacePixels = np.array(srcFace)
    if isinstance(targetFace, Image.Image):
        targetFacePixels = np.array(targetFace)
    mask = np.dstack([mask]*3)
    return mask * targetFacePixels + (1 - mask) * srcFacePixels

def images2pilBatch(images, batchSize=1):
    '''
    if batchSize==1, return (path, pil).
    else return ([path0, path1, ...], [pil0, pil1, ...])
    '''
    if batchSize <= 0:
        raise ValueError(f'Batch size must be positive, but got {batchSize}.')
    pathBatch = []
    imagePilBatch = []
    for image in images:
        try:
            imagePil = Image.open(image)
        except:
            imagePil = None
        pathBatch.append(image)
        imagePilBatch.append(imagePil)
        if len(pathBatch) == batchSize:
            if batchSize == 1:
                yield pathBatch[0], imagePilBatch[0]
                pathBatch, imagePilBatch = [], []
            else:
                yield pathBatch, imagePilBatch
                pathBatch, imagePilBatch = [], []


def convex_hull(size, points, fillColor=(255,)*3):
    mask = np.zeros(size, dtype=np.uint8) # mask has the same depth as input image
    corners = np.expand_dims(np.array(points), axis=0).astype(np.int32)
    cv2.fillPoly(mask, corners, fillColor)
    return mask


def random_deform(imageSize, lms, mean=0, std=3):
    h, w = imageSize
    deformed = lms + np.random.normal(mean, std, size=lms.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])
    return deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    warped = warp(image, trans)
    return warped.astype(np.uint8)


def distance(lms1, lms2):
    return np.linalg.norm(lms1 - lms2)


def get_parser():
    parser = argparse.ArgumentParser(description='Demo for face x-ray fake sample generation')
    parser.add_argument('--srcFacePath', '-sfp', type=str)
    parser.add_argument('--faceDatabase', '-fd', type=str)
    parser.add_argument('--threshold', '-t', type=float, default=25)
    return parser.parse_args()


if __name__ == '__main__':
    main()