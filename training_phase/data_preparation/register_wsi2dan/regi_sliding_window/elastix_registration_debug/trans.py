import SimpleITK as sitk
basic_transform = sitk.Euler2DTransform()
basic_transform.SetTranslation((2,3))

sitk.WriteTransform(basic_transform, 'euler2D.tfm')
read_result = sitk.ReadTransform('euler2D.tfm')

assert(str(type(read_result) != type(basic_transform)))
