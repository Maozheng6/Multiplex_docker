import numpy as np
import cv2
import SimpleITK
import SimpleITK as sitk
fixedImage = SimpleITK.ReadImage("O0135-multires_Image_643_dan.png", sitk.sitkFloat32)
fixedImage = SimpleITK.Cast(fixedImage,sitk.sitkUInt8)
sitk.WriteImage(fixedImage, 'fixedImage.png')
#movingImage = SimpleITK.ReadImage("O0135-multires_Image_643_wsi.png", sitk.sitkFloat32)
#resultImage = SimpleITK.Elastix(fixedImage, movingImage)
