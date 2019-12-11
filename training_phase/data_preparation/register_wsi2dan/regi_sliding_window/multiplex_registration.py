import numpy as np
import cv2 as cv
import os;
import glob;

# Number of features to extract
# The wsi patch is assumed to be larger than the reference patch and so the number of features extracted is larger
# Number of features to extract from the reference patch
MAX_FEATURES_1 = 500;
# Number of features to extract from the wsi patch
MAX_FEATURES_2 = 5000;
# percent of top matched features to use
GOOD_MATCH_PERCENT = 0.15
# matching similarity constraints
# nearest feature should be no more than MATCH_MIN_DIST
MATCH_MIN_DIST =50# 15;
# farthest feature should be no more than MATCH_MAX_DIST
MATCH_MAX_DIST =70 #25;



def register(ref_patch_filepath, wsi_patches_dir, out_dir):
    print('ref_patch_filepath = ', ref_patch_filepath)
    print('wsi_patches_dir = ', wsi_patches_dir)
    ref_ratio = 0.293/0.3468
    # ORB features
    # Initiate ORB detector
    orb1 = cv.ORB_create(MAX_FEATURES_1)
    orb2 = cv.ORB_create(MAX_FEATURES_2)

    # read ref patch
    img1 = cv.imread(ref_patch_filepath,0)
    img1 = cv.resize(img1,((int(img1.shape[1]*ref_ratio),int(img1.shape[0]*ref_ratio))))
    # find the keypoints with ORB
    kp1, descriptors1 = orb1.detectAndCompute(img1,None)
    # draw only keypoints location,not size and orientation
    img1_feat = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)

    # read wsi patches directory
    wsi_patches_files = glob.glob(os.path.join(wsi_patches_dir, '*.png'))
    print('len(wsi_patches_files)',len(wsi_patches_files))

    if(len(wsi_patches_files ) <= 0):
        print('')
        print('No .PNG files in WSI patches directory');
        return;

    found = False;
    for input_wsi_patch_filepath in wsi_patches_files:
        img2 = cv.imread(input_wsi_patch_filepath,0)
        img2 = cv.resize(img2,(int(img2.shape[1]/2),int(img2.shape[0]/2)))
        kp2, descriptors2 = orb2.detectAndCompute(img2,None)

        # Match features
        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        dist_list=[x.distance for x in matches]
        print('EEEEEEEEEEEEEEE')
        print('dist_list',dist_list)
        print('dist_list.mean()',dist_list.mean())
        print('matches[0].distance = ', matches[0].distance)
        print('matches[-1].distance = ', matches[-1].distance)

        if(matches[0].distance > MATCH_MIN_DIST or matches[-1].distance > MATCH_MAX_DIST):
            continue;

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        h, mask = cv.findHomography(points2, points1, cv.RANSAC)

        # Use homography
        img2 = cv.imread(input_wsi_patch_filepath,1)
        img2 = cv.resize(img2,(int(img2.shape[1]/2),int(img2.shape[0]/2)))
        img1 = cv.imread(ref_patch_filepath,1)
        img1 =cv.resize(img1,((int(img1.shape[1]*ref_ratio),int(img1.shape[0]*ref_ratio))))
        height, width,channels = img1.shape
        img2Reg = cv.warpPerspective(img2, h, (width, height))
        filename_ref = os.path.splitext(os.path.split(ref_patch_filepath)[1])[0];
        filename_target, ext = os.path.splitext(os.path.split(input_wsi_patch_filepath)[1]);
        cv.imwrite(os.path.join(out_dir, filename_ref + '_match_'+filename_target+ext), img2Reg)
        print('')
        print('found match: ', input_wsi_patch_filepath);
        found = True;
        break;
    if(not found):
        print('')
        print('No match found for: ', ref_patch_filepath);
    print('');

def register_single(ref_patch_filepath, wsi_patch_filepath, out_dir):
    print('ref_patch_filepath = ', ref_patch_filepath)
    print('wsi_patch_filepath = ', wsi_patch_filepath)


    # read ref patch
    img1 = cv.imread(ref_patch_filepath,0)
    img2 = cv.imread(wsi_patch_filepath,0)
    img1_area=img1.shape[1]*img1.shape[0]
    img2_area=img2.shape[1]*img2.shape[0]
    area_ratio=img2_area/img1_area


    MAX_FEATURES_2 = int(MAX_FEATURES_1 *  area_ratio)
    # ORB features
    # Initiate ORB detector
    orb1 = cv.ORB_create(MAX_FEATURES_1)
    orb2 = cv.ORB_create(MAX_FEATURES_2)

    # find the keypoints with ORB
    kp1, descriptors1 = orb1.detectAndCompute(img1,None)
    # draw only keypoints location,not size and orientation
    img1_feat = cv.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    cv.imwrite('key_points1.png',img1_feat)
    print('kp1',kp1)
    found = False;
    #img2 = cv.imread(wsi_patch_filepath,0)
    kp2, descriptors2 = orb2.detectAndCompute(img2,None)

    img2_feat = cv.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
    cv.imwrite('key_points2.png',img2_feat)

    # Match features
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    print('matches',matches)
    dist_list=[x.distance for x in matches]
    print('dist_list',dist_list)
    print('dist_list.mean()',np.mean(np.array(dist_list)))

    print('matches[0].distance = ', matches[0].distance)
    print('matches[-1].distance = ', matches[-1].distance)

    if(matches[0].distance > MATCH_MIN_DIST or matches[-1].distance > MATCH_MAX_DIST):
        print('')
        print('No match found')
        return;

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points2, points1, cv.RANSAC)

    # Use homography
    img2 = cv.imread(wsi_patch_filepath,1)
    img1 = cv.imread(ref_patch_filepath,1)
    height, width,channels = img1.shape
    img2Reg = cv.warpPerspective(img2, h, (width, height))
    filename = os.path.splitext(os.path.split(ref_patch_filepath)[1])[0];
    ext = os.path.splitext(wsi_patch_filepath)[1];
    cv.imwrite(os.path.join(out_dir, filename + '_wsi_match'+ext), img2Reg)

if __name__ == "__main__":
    '''
    # Search over all WSI patches and register
    out_dir = './output_patches'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    wsi_patches_dir = '/scratch/KurcGroup/mazhao/tiles_slide/O0135-multires/'
    ref_patches_dir = '/scratch/KurcGroup/mazhao/ICCV_NEW_DOTS_data_code_v2_regi-wsi/O3936/'

    ref_patches = glob.glob(os.path.join(ref_patches_dir, '*.tif'));
    print('len(ref_patches)',len(ref_patches))
    for ref_patch_filepath in ref_patches[2:3]:
        register(ref_patch_filepath, wsi_patches_dir, out_dir);
    '''
    # Register to a specific WSI patch
    out_dir = './s_out/'
    ref_patch_filepath = './test_imgs/O3936_dan_Image_621_level_4.png'
    wsi_patch_filepath = './cropped_imgs/O3936_wsi_Image_621_25952_89188_157016_114118_level_0.png'

    register_single(ref_patch_filepath, wsi_patch_filepath, out_dir);
