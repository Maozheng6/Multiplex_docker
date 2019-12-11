import numpy as np
import cv2
from PIL import Image

def wsi_2_dan(bgr_im):
    matrix = np.array([
        [0.211008, 0.339041, -0.646346, ],
        [-7.565876, -1.898291, -3.057396, ],
        [8.308303, 1.768644, 4.262272, ],
        [0.423514, -0.304383, 1.079028, ],
        [8.498341, 2.800226, -2.535058, ],
        [-7.941607, -0.727354, -6.032954, ],
        [-0.076087, 2.375034, -0.494513, ],
        [-2.028586, -0.758861, 1.203198, ],
        [3.215897, 0.753147, 1.475563, ],
        [-0.306964, -0.675827, 0.833499, ],
        [0.501877, -0.557836, 4.965655, ],
        [-3.134376, -1.322737, 2.170693, ],
        [0.412375, -1.682529, -0.360446, ],
    ], dtype=np.float32)

    rgb_im_0to1 = bgr_im[..., ::-1].astype(np.float32) / 255.0
    s0, s1 = rgb_im_0to1.shape[0], rgb_im_0to1.shape[1]
    fea = rgb_im_0to1.reshape((s0 * s1, 3))
    fea = np.concatenate((np.ones((fea.shape[0], 1), dtype=np.float32),
                          fea**0.5,
                          fea,
                          fea*fea,
                          np.log(1.1+fea)), axis=-1)
    #fea = np.concatenate((np.ones((fea.shape[0], 1), dtype=np.float32), fea), axis=-1)
    trans_im = np.matmul(fea, matrix)
    trans_im = trans_im.reshape((s0, s1, 3))

    return np.clip(trans_im[..., ::-1] * 255.0, 0, 255)


def dan_2_wsi(bgr_im):
    matrix_list = [np.array([
    [-0.691400, -0.907071, -0.312631, ],
    [0.990122, 0.524586, 1.050779, ],
    [0.399034, 0.060406, -0.272387, ],
    [-0.458520, -0.099836, -0.586634, ],
    [-0.967061, -0.500585, 2.678765, ],
    [-6.216358, -7.330121, -9.360523, ],
    [0.646215, 0.035609, 4.642730, ],
    [0.207017, -0.013670, -0.629438, ],
    [1.763808, 2.080769, 2.373848, ],
    [-0.082624, 0.071838, -1.054938, ],
    [0.294018, -0.121715, -4.899847, ],
    [7.110159, 9.428182, 12.031165, ],
    [-0.141877, 0.217981, -3.842863, ],
    ], dtype=np.float32),
    np.array([
    [-0.348998, 0.095725, -0.042487, ],
    [0.995965, 0.296123, 1.066293, ],
    [0.206360, 0.228342, -0.302816, ],
    [-0.681683, -0.374928, -1.113111, ],
    [-2.713301, 1.112927, 1.193312, ],
    [1.341621, -1.042814, -1.686801, ],
    [-1.262450, 1.833873, 1.974162, ],
    [0.702425, -0.333071, -0.339547, ],
    [-0.365274, 0.309220, 0.417382, ],
    [0.374709, -0.432080, -0.544830, ],
    [2.895458, -1.239219, -2.688078, ],
    [-1.702090, 1.214354, 2.759007, ],
    [2.458807, -0.988419, 0.366372, ],
    ], dtype=np.float32),

    #O3936_cd16
    np.array([
    [-0.847856, -0.835641, 0.276094, ],
    [0.659347, -1.119521, -2.412230, ],
    [-0.119552, 0.271574, 0.547180, ],
    [-0.400778, 1.314334, 1.690406, ],
    [-5.607094, -7.277171, -12.218688, ],
    [-2.456439, 0.250656, -0.830469, ],
    [1.854825, 0.265907, 17.312166, ],
    [1.179525, 1.557734, 2.451412, ],
    [0.675682, -0.017055, 0.407587, ],
    [-0.366545, 0.277632, -3.954805, ],
    [7.184437, 10.545779, 18.665323, ],
    [3.456853, 0.723484, 0.492653, ],
    [-1.449584, -2.207663, -21.755878, ],
    ], dtype=np.float32),
    #N9430 Image_589
    np.array([
    [-0.372883, 0.155283, 0.430937, ],
    [1.782100, -1.027641, -1.069290, ],
    [0.077991, 0.626502, -0.026790, ],
    [-1.221496, 1.010353, 1.381791, ],
    [0.298986, -5.542871, -9.020520, ],
    [1.541923, 4.094269, 3.276868, ],
    [-4.965525, 2.979019, 10.207441, ],
    [-0.018439, 1.148344, 1.935642, ],
    [-0.264824, -0.792277, -0.634855, ],
    [1.184464, -0.500749, -2.197257, ],
    [-1.848618, 8.277455, 12.244605, ],
    [-1.640417, -5.145684, -3.299775, ],
    [7.338722, -4.823029, -13.527246, ],
    ], dtype=np.float32),

    #N22034 Image_581
    np.array([
    [-0.342473, -0.757583, 1.835823, ],
    [1.983790, 1.953061, 1.175429, ],
    [0.062471, -0.224956, -0.083082, ],
    [-1.373470, -1.068827, -0.472709, ],
    [0.518166, 10.287962, 4.639609, ],
    [-3.637795, -11.295032, -9.450535, ],
    [0.203679, -5.674267, 21.144709, ],
    [0.085974, -2.356541, -1.125023, ],
    [0.966892, 3.011987, 2.567600, ],
    [-0.190982, 1.197293, -5.049102, ],
    [-2.752493, -15.257564, -7.077002, ],
    [4.439950, 14.664626, 11.125823, ],
    [1.886983, 8.523763, -23.327636, ],
    ], dtype=np.float32),

    #L6745
    np.array([
    [-1.035352, -0.043666, -0.109958, ],
    [2.150797, 0.415399, -0.015848, ],
    [0.561727, 1.006224, 1.292150, ],
    [-1.847759, -0.339066, -1.114157, ],
    [7.598414, 2.468008, 1.627785, ],
    [-5.826417, -0.879380, -7.830207, ],
    [-11.113488, -2.576245, 6.261864, ],
    [-1.765692, -0.613363, -0.607376, ],
    [2.080572, 0.707447, 2.683082, ],
    [2.235499, 0.594956, -2.006985, ],
    [-11.350936, -3.669145, -1.822323, ],
    [5.837961, 0.482491, 6.985591, ],
    [16.462489, 3.733693, -3.916060, ],
    ], dtype=np.float32),
    #O3105
    np.array([
    [-1.814545, -1.407119, -0.645490, ],
    [-0.140223, 0.296693, -0.144425, ],
    [1.002740, 0.273782, 0.543349, ],
    [-0.646984, -0.155082, -0.524997, ],
    [-0.181083, 5.445331, 2.667935, ],
    [-4.856306, -7.111665, -7.053956, ],
    [-10.134331, -10.388444, 0.056523, ],
    [0.099131, -1.279818, -0.784075, ],
    [1.370425, 1.974001, 2.100048, ],
    [2.242514, 2.478024, -0.206646, ],
    [0.724870, -7.249971, -2.958062, ],
    [4.713680, 9.280996, 7.491798, ],
    [13.674031, 12.805571, 2.311974, ],
    ], dtype=np.float32),

   #L6745 Image_651
    np.array([
    [-0.721131, 0.595376, 0.990045, ],
    [1.445514, -0.508869, -0.830588, ],
    [0.528650, 0.516233, 1.318794, ],
    [-1.403505, 0.906759, -0.265586, ],
    [1.294699, -5.292382, -6.155943, ],
    [-3.785459, 3.003405, -3.125272, ],
    [-3.666861, 7.150447, 18.896824, ],
    [-0.370111, 1.156218, 1.195779, ],
    [1.386562, -0.534236, 1.393105, ],
    [0.593168, -1.471661, -4.883745, ],
    [-2.673795, 6.919111, 8.598681, ],
    [3.527000, -3.240761, 1.365339, ],
    [6.567167, -10.068847, -20.488370, ],
    ], dtype=np.float32),
    #########################################
    #L6745 Image_645
    np.array([
    [-1.153398, -0.742396, -0.076000, ],
    [0.913061, -0.079883, 1.209378, ],
    [0.071610, 0.511193, -0.086464, ],
    [-0.834243, 0.426459, -1.851186, ],
    [-4.453139, -1.901712, 0.093140, ],
    [-2.235855, -2.674827, -5.908273, ],
    [-2.611519, -2.397311, 7.541320, ],
    [0.953831, 0.415193, 0.039293, ],
    [0.616453, 0.769413, 1.570260, ],
    [0.627376, 0.793090, -2.314184, ],
    [5.195811, 2.402067, -1.641866, ],
    [2.494510, 3.105438, 6.891947, ],
    [4.198875, 2.070389, -4.660162, ],
    ], dtype=np.float32),

    #N9430   image_595

    np.array([
    [-1.249157, -1.352967, -1.030913, ],
    [1.811385, 0.850428, 1.142159, ],
    [-0.156405, -0.406774, -1.169444, ],
    [-1.135493, -0.116637, -0.607338, ],
    [1.260490, 1.229967, 1.142397, ],
    [-8.678963, -11.359181, -13.876838, ],
    [-3.280548, -1.283855, 5.796755, ],
    [-0.195623, -0.322029, -0.396131, ],
    [2.300450, 2.971391, 3.584382, ],
    [0.567544, 0.242361, -1.660072, ],
    [-3.559541, -2.864484, -2.942311, ],
    [11.027726, 15.191888, 18.547595, ],
    [5.659506, 1.888443, -4.766260, ],
    ], dtype=np.float32),

    #N22034    Image_584
    np.array([
    [-0.843135, -1.116633, -0.503482, ],
    [-0.176548, 0.015263, -0.269271, ],
    [1.479645, 0.600750, 1.134416, ],
    [-0.840890, -0.299453, -1.801483, ],
    [-3.761901, -0.476544, -2.724586, ],
    [2.571882, -2.710755, -0.676748, ],
    [-6.204990, -6.306740, -0.538665, ],
    [0.957717, 0.065983, 0.467112, ],
    [-0.290386, 1.062823, 0.705931, ],
    [1.301926, 1.388210, -0.196331, ],
    [5.231415, 0.547496, 3.905834, ],
    [-5.012744, 3.172922, -1.101847, ],
    [8.934919, 8.107826, 4.440312, ],
    ], dtype=np.float32),

    #O3936 Image_621
    np.array([
    [0.433120, 0.700673, 1.140691, ],
    [0.812521, 0.498188, 0.740020, ],
    [0.099811, -0.080738, -0.147698, ],
    [0.085491, 0.734671, -0.186189, ],
    [-1.422384, -1.702829, -3.458806, ],
    [0.316836, 1.470842, -0.637141, ],
    [4.567290, 5.718797, 14.850359, ],
    [0.396657, 0.590887, 0.959111, ],
    [-0.015113, -0.459259, 0.218928, ],
    [-1.033272, -1.137768, -3.786370, ],
    [1.499940, 1.375615, 3.325566, ],
    [-0.590584, -0.742496, 0.819703, ],
    [-5.582694, -8.113975, -16.236328, ],
    ], dtype=np.float32),

    #L6745  Image_648
    np.array([
    [-0.445799, -0.392390, -0.546426, ],
    [1.176924, -0.714361, 0.459571, ],
    [0.342661, 0.559513, 0.130329, ],
    [-0.770841, 0.677469, -1.107242, ],
    [0.073032, -4.718188, -1.429872, ],
    [-0.628825, 4.512059, -0.708842, ],
    [-3.450640, -3.069632, -0.716583, ],
    [0.143186, 1.030401, 0.351072, ],
    [0.290381, -0.852456, 0.307737, ],
    [0.762265, 0.894574, -0.033883, ],
    [-0.955443, 6.831888, 1.346934, ],
    [0.437461, -5.263859, 0.851235, ],
    [5.197202, 2.543345, 3.533534, ],
    ], dtype=np.float32),

    ]
    m_indx = np.random.choice(len(matrix_list))
    #print('domain_shift_matrix_indx',m_indx)
    matrix = matrix_list[m_indx]
    rgb_im_0to1 = bgr_im[..., ::-1].astype(np.float32) / 255.0
    s0, s1 = rgb_im_0to1.shape[0], rgb_im_0to1.shape[1]
    fea = rgb_im_0to1.reshape((s0 * s1, 3))
    fea = np.concatenate((np.ones((fea.shape[0], 1), dtype=np.float32),
                          fea**0.5,
                          fea,
                          fea*fea,
                          np.log(1.1+fea)), axis=-1)
    #fea = np.concatenate((np.ones((fea.shape[0], 1), dtype=np.float32),      fea), axis=-1)
    trans_im = np.matmul(fea, matrix)
    trans_im = trans_im.reshape((s0, s1, 3))

    return np.clip(trans_im[..., ::-1] * 255.0, 0, 255)
