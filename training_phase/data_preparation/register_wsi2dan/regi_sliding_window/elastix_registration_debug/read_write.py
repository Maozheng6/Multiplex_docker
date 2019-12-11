import SimpleITK as sitk
inputImageFileName = 'O0135-multires_Image_643_wsi.png'
outputImageFileName = '111.png'
reader = sitk.ImageFileReader()
reader.SetImageIO("PNGImageIO")
reader.SetFileName(inputImageFileName)
image = reader.Execute();

writer = sitk.ImageFileWriter()
writer.SetFileName(outputImageFileName)
writer.Execute(image)
