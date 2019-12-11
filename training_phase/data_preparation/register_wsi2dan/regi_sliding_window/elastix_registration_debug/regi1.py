#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os


def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))

if len ( sys.argv ) < 4:
    print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
    sys.exit ( 1 )


fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)

moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
#moving2 = sitk.ReadImage(sys.argv[2], sitk.sitkUInt8)
reader = sitk.ImageFileReader()
reader.SetImageIO("PNGImageIO")
reader.SetFileName(sys.argv[2])
moving2 = reader.Execute();

sitk.WriteImage(moving2, 'moving2.png')
R = sitk.ImageRegistrationMethod()
R.SetMetricAsMeanSquares()
R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
R.SetInterpolator(sitk.sitkLinear)

R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

outTx = R.Execute(fixed, moving)

print("-------")
print(outTx)
print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
print(" Iteration: {0}".format(R.GetOptimizerIteration()))
print(" Metric value: {0}".format(R.GetMetricValue()))

sitk.WriteTransform(outTx,  sys.argv[3])

if ( not "SITK_NOSHOW" in os.environ ):

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving2)
    out = resampler.Execute(out)
    out = resampler.Execute(out)
    out = resampler.Execute(out)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = out#sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    #cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
    #sitk.Show( cimg, "ImageRegistration1 Composition" )
    sitk.WriteImage(simg1, 'fixed.png')
    #sitk.WriteImage(simg2, 'registered.png')
    writer = sitk.ImageFileWriter()
    writer.SetFileName('registered.png')
    writer.Execute(simg2)
