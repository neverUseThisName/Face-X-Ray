# Face-X-Ray
Unofficial implementation of paper 'Face X-ray for More General Face Forgery Detection'. (This demo is under developing....)

## Key Requirements
torch, torchvision, scikit-image, dlib, color_transfer.

## How it works
    1. input: source face image (I_B in paper, one image file or directory of images) and a directory of real face images as face database.
    2. search face database for the one whose landmarks are close to source face image.
    3. apply convex hull, random deform, piecewise affine transform, color correction and swap.
    4. save result in ./dump
    
## Demo Result
![](https://github.com/neverUseThisName/Face-X-Ray/blob/master/result/forge_0.jpg)
![](https://github.com/neverUseThisName/Face-X-Ray/blob/master/result/target_0.jpg)
