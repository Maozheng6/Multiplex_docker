import cv2
import openslide
import numpy as np
import os
import concurrent.futures
import glob
def cropping(slide_name,wsi_path,low_res_size,upper_left_in_low,dan_tile_idx,win_size_level,level,save_folder,save_img):
#whole slide size
    print('low_res_size',low_res_size)
    print('upper_left_in_low',upper_left_in_low)
    oslide = openslide.OpenSlide(wsi_path+slide_name+'.tif')
    oslide_size = np.array(oslide.level_dimensions[0])
    print('oslide_size','0:',np.array(oslide.level_dimensions[0]),str(level),':',oslide_size)
    upper_left_in_wsi =(oslide_size * (upper_left_in_low/low_res_size)).astype(int)
    print('upper_left_in_wsi',upper_left_in_wsi)
    crop_size = win_size_level
    cropped = oslide.read_region((upper_left_in_wsi[0],upper_left_in_wsi[1]), level, crop_size)
    resize_ratio=0.5*0.3468/0.293
    new_size=(np.array(crop_size)*resize_ratio).astype(int)
    print('crop_size',crop_size)
    #resize to the same mpp as ref_img
    cropped=cropped.resize(new_size)
    #save_folder = './cropped_imgs_wd2_st_1/'
    if save_img==True:
        cropped.save(save_folder+slide_name.split('-')[0]+'_wsi_'+dan_tile_idx+'_'+str(upper_left_in_low[0])+'_'+str(upper_left_in_low[1])+'_'+str(low_res_size[0])+'_'+str(low_res_size[1])+'_level_'+str(level)+'.png')

    return cropped


def register_single(img1_ref_bgr, img2_wsi_bgr, out_dir, slide_name, dan_tile_idx,MAX_FEATURES_1,GOOD_MATCH_PERCENT,save_ratio, if_regi):
    cv2.imwrite(os.path.join(out_dir, slide_name+'_'+dan_tile_idx+'_ratio'+ str(save_ratio)+'_dan.png'), img1_ref_bgr)
    '''
    #MAX_FEATURES_1 = 500
    #GOOD_MATCH_PERCENT = 0.15
    # read ref patch
    #img1 = cv.imread(ref_patch_filepath,0)
    #img2 = cv.imread(wsi_patch_filepath,0)
    print('img1_ref_bgr.shape',img1_ref_bgr.shape)
    print('img2_wsi_bgr.shape',img2_wsi_bgr.shape)
    img1 = cv2.cvtColor(img1_ref_bgr, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_wsi_bgr, cv2.COLOR_BGR2GRAY)
    img1_area=img1.shape[1]*img1.shape[0]
    img2_area=img2.shape[1]*img2.shape[0]
    area_ratio=img2_area/img1_area


    MAX_FEATURES_2 = int(MAX_FEATURES_1 *  area_ratio)
    # ORB features
    # Initiate ORB detector
    orb1 = cv2.ORB_create(MAX_FEATURES_1)
    orb2 = cv2.ORB_create(MAX_FEATURES_2)

    # find the keypoints with ORB
    kp1, descriptors1 = orb1.detectAndCompute(img1,None)
    # draw only keypoints location,not size and orientation
    img1_feat = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
    #cv2.imwrite('key_points1.png',img1_feat)
    #print('kp1',kp1)
    found = False;
    #img2 = cv.imread(wsi_patch_filepath,0)
    kp2, descriptors2 = orb2.detectAndCompute(img2,None)

    img2_feat = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
    #cv2.imwrite('key_points2.png',img2_feat)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    try:
        matches = matcher.match(descriptors1, descriptors2, None)
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        print('matches',len(matches))#,matches)
        dist_list=[x.distance for x in matches]
        #print('dist_list',dist_list)
        print('dist_list.mean()',np.mean(np.array(dist_list)))

        print('matches[0].distance = ', matches[0].distance)
        print('matches[-1].distance = ', matches[-1].distance)
        sim = np.mean(np.array(dist_list))
    except:
        sim = 1000000
    if if_regi != True :
        return sim
    else:
        #MATCH_MIN_DIST = 40
        #MATCH_MAX_DIST = 55
        #if(matches[0].distance > MATCH_MIN_DIST or matches[-1].distance >      MATCH_MAX_DIST):
        #    print('')
        #    print('No match found')
        #    return;

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography
        img2 = img2_wsi_bgr#cv.imread(wsi_patch_filepath,1)
        img1 = img1_ref_bgr#cv.imread(ref_patch_filepath,1)
        cv2.imwrite('img2_wsi_bgr.png',img2_wsi_bgr)
        cv2.imwrite('img1_ref_bgr.png',img1_ref_bgr)
        height, width,channels = img1.shape
        ratioed_size = np.array((width*save_ratio, height*save_ratio)).astype('int')
        img2Reg = cv2.warpPerspective(img2, h, (ratioed_size[0],ratioed_size[1]))
        print('##img2Reg.shape',img2Reg.shape)
        cv2.imwrite(os.path.join(out_dir, slide_name+'_'+dan_tile_idx+'_ratio'+str(save_ratio)+'_wsi.png'), img2Reg)
        cv2.imwrite(os.path.join(out_dir, slide_name+'_'+dan_tile_idx+'_ratio'+ str(save_ratio)+'_dan.png'), img1)
        print('save',os.path.join(out_dir, slide_name+'_'+dan_tile_idx+'_ratio'+str(save_ratio)+'_wsi.png'))

        return img2Reg
    '''
    return img1_ref_bgr

class SLIDING_WINDOW():
    def __init__(self,slide_name, wsi_path, slide_size, dan_tile_idx, win_size_level, level, ref_img_level, regi_out_dir, step_size):
        self.slide_name = slide_name
        self.wsi_path = wsi_path
        self.oslide_size = slide_size
        self.dan_tile_idx = dan_tile_idx
        self.win_size_level = win_size_level
        self.level = level
        self.ref_img_level = ref_img_level
        self.regi_out_dir = regi_out_dir
        self.step_size = step_size
        self.sim_matrix = np.zeros((len(list(range(0,self.oslide_size[1],step_size[1]))),len(list(range(0,self.oslide_size[0],step_size[0])))))+1000

    def one_window(self,paras):
        #This is the key function
        #register one cropped patch from wsi, save the similarity value to the matrix
        [col,row,col_counter,row_counter] = paras
        window = cropping(self.slide_name,self.wsi_path,np.array((self.oslide_size[0],self.oslide_size[1])),np.array((col,row)),self.dan_tile_idx,self.win_size_level,self.level,'',False)

        # to BGR array
        window = np.array(window)[:,:,0:3]
        window = window[:,:,::-1]
        #To Do: make the two parameters controllable
        sim_value = register_single(self.ref_img_level, window, self.regi_out_dir,self.slide_name,self.dan_tile_idx,MAX_FEATURES_ROUGH,GOOD_MATCH_PERCENT_ROUGH,1.0,if_regi = False)
        print('sim_value',(row_counter,col_counter),sim_value,self.sim_matrix.shape)

        self.sim_matrix[row_counter,col_counter] = sim_value
        #print('np.min(self.sim_matrix)',self.sim_matrix)

if __name__ == '__main__':

    raw_dots_dan_folder = '/scratch/KurcGroup/mazhao/ICCV/data_multiplex/dots_dan/Danielle_raw_dots_final/raw_unzipped/'
    wsi_path ='/gpfs/scratch/mazhao/multiplex-wsi/' #'../../multiplex-wsi/'
    # parameters for raough registration

    # level of wsi for rough registration
    level=3
    MAX_FEATURES_ROUGH = 500
    GOOD_MATCH_PERCENT_ROUGH =0.15
    # parameters for fine registration

    level_fine = 0
    MAX_FEATURES_FINE = 4000
    GOOD_MATCH_PERCENT_FINE =0.30

    #output folders
    regi_out_dir = './regi_out_final_all/'
    slide_results = './middle_results_for_slides_all/' #'./slide_results_20190603_more_multiplex_labels/'#'./middle_results_for_slides/'
    #cropped_save_folder = '/scratch/KurcGroup/mazhao/register_wsi2dan/regi_sliding_window/cropped_imgs_wd2_st_1/'#'./cropped_imgs_wd2_st_1/'
    cropped_save_folder = './cropped_imgs_wd2_st_1_all/'
    if not os.path.exists(regi_out_dir):
        os.makedirs(regi_out_dir)
    if not os.path.exists(slide_results):
        os.makedirs(slide_results)
    if not os.path.exists(cropped_save_folder):
        os.makedirs(cropped_save_folder)
    ##############################

    #wsi2dan_slide ={'M28417-multires': 'M28417_2', 'M4213-multires': 'M4213_2', 'N24852-multires': 'N24852_5', 'N27243-multires': 'N27243_5', 'N29055-multires': 'N29055_30', 'N5039-multires': 'N5039_5', 'O21747-multires': 'O21747_95', 'O8372-multires': 'O8372_10', 'P0992-multires': 'P0992_80', 'P24146-multires': 'P24146_90', 'P28191-multires': 'P28191_80', 'P304-multires': 'P304_80', 'P4211-multires': 'P4211_0', 'P670-multires': 'P670_5', 'M3669-multires': 'M3669_50','N24178-multires': 'N24178_80', 'N27093-multires': 'N27093_95', 'N27702-multires': 'N27702_20', '3908-multires': 'N3908_70', 'N8945-multires': 'N8945_50', 'O6218_MULTI_3-multires': 'O6218_30', 'P0533-multires': 'P0533_50', 'N22034-multires': 'P22034_2', 'P24230-multires': 'P24230_20', 'P29193-multires': 'P29193_65', 'P31681-multires': 'P31681_40', 'P528-multires': 'P528_30'}
    #['M28417_2',  'M4213_2'  ,  'N24852_5' ,  'N27243_5' ,  'N29055_30',  'N5039_5' ,  'O21747_95',  'O8372_10',  'P0992_80',  'P24146_90',  'P28191_80' , 'P304_80',    'P4211_0',  'P670_5','M3669_50',  'N24178_80',  'N27093_95',  'N27702_20',  'N3908_70',   'N8945_50',  'O6218_30',   'P0533_50',  'P22034_2',  'P24230_20',  'P29193_65',  'P31681_40',  'P528_30']
    #wsi2dan_slide = {'N22800-multires':'L22800_20', 'L28352-multires':'L28352_2', 'L29978-multires':'L29978_2', 'M036-multires':'M036_15','P24146-multires':'P24146_90','N8032-multires':'N8033_95'}
    # {'O3936-multires':'O3936','L6745-multires':'L6745','N22034-multires':'N22034_90_Scale_bar_is_set_wrong', 'O0135-multires':'O0135','O3105-multires':'O3105_10','N9430-multires':'N9430'}#'O3936-multires':'O3936',
    ############################
    # get wsi dan map dict

    import pandas as pd

    input_df = pd.read_csv('wsi_dan_map.csv')
    pair_list = input_df[["wsi", "dan"]].values

    print(pair_list)

    wsi_dan_dict = {}
    for x in pair_list:
        wsi_dan_dict[x[0]] = x[1]

    print(wsi_dan_dict)
    wsi2dan_slide = wsi_dan_dict

    #wsi2dan_slide = {'N27243-multires':'N27243_5'}#{'N22800-multires':'N22800_20'}

    ######################################
    count_p =0
    for slide_name in wsi2dan_slide.keys():
        ref_folder = raw_dots_dan_folder+wsi2dan_slide[slide_name]+'/'
        files = glob.glob(ref_folder+'*tif')
        #print('dan folder', ref_folder)

        #print('dan files',files)
        #print('wsi',wsi_path+slide_name+'.tif')
        '''
        try:
            oslide = openslide.OpenSlide(wsi_path+slide_name+'.tif')
            oslide_size = oslide.level_dimensions[0]
        except:
            print( 'wsi tif not exists')
            continue
        '''
        if os.path.exists(wsi_path+slide_name+'.tif'):
            pass
        else:
            print( wsi_path+slide_name+'.tif', 'not exists')
            continue
        '''
        for file_i in files:
            count_p +=1
            print('count_p',count_p)
            dan_tile_idx = os.path.basename(file_i)[0:-4]
            print('dan_tile_idx',dan_tile_idx)
            ref_path = file_i
            ref_folder = os.path.basename(os.path.dirname(ref_path))
            ref_img = cv2.imread(ref_path)
            size_dan = (ref_img.shape[1],ref_img.shape[0])

            #width, height
            #the mpp of ref_img is 0.293
            #the mpp of WSI image is 0.3468*0.5
            dan_on_wsi_size = (np.array(size_dan)*0.293/(0.3468*0.5)).astype(int)
            # set window size and step size
            win_size = 2*dan_on_wsi_size
            step_size = 1*dan_on_wsi_size

            print('dan_on_wsi_size',dan_on_wsi_size)

            # sizes on the respect wsi level
            dan_on_wsi_size_level = (dan_on_wsi_size/(2**level)).astype(int)
            win_size_level = (win_size/(2**level)).astype(int)
            step_size_level = (step_size/(2**level)).astype(int)
            print('dan_on_wsi_size_level',dan_on_wsi_size_level)

            #resize ref image to the same scale of wsi at the level
            ref_img_level = cv2.resize(ref_img,(int((ref_img.shape[1]*0.293/(0.3468*0.5))/(2**level)),int((ref_img.shape[0]*0.293/(0.3468*0.5))/(2**level))))
            print('ref_img_level.shape',ref_img_level.shape)


            # get sliding_list (cooordinates of cropping patches in wsi)
            counter = 0
            col_counter = 0
            row_counter = 0
            sliding_list=[]
            for col in range(0,oslide_size[0],step_size[0]):
                if col != 0:
                    col_counter += 1
                row_counter = 0
                for row in range(0,oslide_size[1],step_size[1]):
                    counter += 1
                    sliding_list.append([col,row,col_counter,row_counter])
                    row_counter += 1

            #build the class
            sliding_window = SLIDING_WINDOW(slide_name, wsi_path, oslide_size, dan_tile_idx, win_size_level, level,ref_img_level, regi_out_dir, step_size)
            #matrix record the similarity value between the patch at each location and the ref_img
            #sim_matrix = np.zeros((len(list(range(0,oslide_size[1],step_size[1]))),len(list(range(0,   oslide_size[0],step_size[0])))))+1000


            # run in a string
            #for paras in sliding_list:
            #    sliding_window.one_window(paras)

            #run in parallel
            regi_output_folder = os.path.join(regi_out_dir,ref_folder)
            # skip existing files
            if os.path.exists(os.path.join(regi_output_folder, slide_name+'_'+dan_tile_idx+'_ratio'+str(1.0)+'_wsi.png')):
                print('exists skip:',regi_output_folder, slide_name+'_'+dan_tile_idx+'_ratio'+str(1.0)+'_wsi.png')
                continue

            # search one Dan's patch in the wsi
            with concurrent.futures.ThreadPoolExecutor( max_workers= 70) as executor:
                for number, prime in zip(sliding_list, executor.map(sliding_window.one_window, sliding_list, chunksize=4)):
                    print('%s is prime: %s' % (number, prime))
            np.save(slide_results+slide_name+'_'+dan_tile_idx+'_'+'sim_matrix.npy',sliding_window.sim_matrix)

            # visualize min_point in sim_matrix
            min_point_png = (sliding_window.sim_matrix == np.min(sliding_window.sim_matrix)).astype('uint8')*255
            cv2.imwrite(slide_results+slide_name+'_'+dan_tile_idx+'_'+'min_point.png',min_point_png)

            print('final np.min(self.sim_matrix)',sliding_window.sim_matrix)

        '''
        #refine ? do it agrain?
        '''
        log = open('./wsi_dan_map_log.txt','w')
        for file_i in files:

            dan_tile_idx = os.path.basename(file_i)[0:-4]
            print('dan_tile_idx',dan_tile_idx)
            ref_path = file_i
            ref_folder = os.path.basename(os.path.dirname(ref_path))
            ref_img = cv2.imread(ref_path)
            size_dan = (ref_img.shape[1],ref_img.shape[0])
            regi_output_folder = os.path.join(regi_out_dir,ref_folder)
            save_ratio = 1.0
            cv2.imwrite(os.path.join(regi_output_folder, slide_name+'_'+dan_tile_idx+'_ratio'+     str(save_ratio)+'_dan.png'), ref_img)
        '''
        '''
            #width, height
            #the mpp of ref_img is 0.293
            #the mpp of WSI image is 0.3468*0.5
            dan_on_wsi_size = (np.array(size_dan)*0.293/(0.3468*0.5)).astype(int)
            win_size = 2*dan_on_wsi_size
            step_size = 1*dan_on_wsi_size

            print('dan_on_wsi_size',dan_on_wsi_size)

            dan_on_wsi_size_level = (dan_on_wsi_size/(2**level)).astype(int)
            win_size_level = (win_size/(2**level)).astype(int)
            step_size_level = (step_size/(2**level)).astype(int)
            print('dan_on_wsi_size_level',dan_on_wsi_size_level)

            ref_img_level = cv2.resize(ref_img,(int((ref_img.shape[1]*0.293/(0.3468*0.5))/(2**level)),int((ref_img.shape[0]*0.293/(0.3468*0.5))/(2**level))))

            print('ref_img_level.shape',ref_img_level.shape)


            # get sliding_list for low res
            counter = 0
            col_counter = 0
            row_counter = 0
            sliding_list=[]
            for col in range(0,oslide_size[0],step_size[0]):
                if col != 0:
                    col_counter += 1
                row_counter = 0
                for row in range(0,oslide_size[1],step_size[1]):
                    counter += 1
                    sliding_list.append([col,row,col_counter,row_counter])
                    row_counter += 1

            #build the class
            sliding_window=SLIDING_WINDOW(slide_name, wsi_path, oslide_size, dan_tile_idx, win_size_level, level,ref_img_level, regi_out_dir,       step_size)

            print('dan_on_wsi_size_level',dan_on_wsi_size_level)
            regi_output_folder = os.path.join(regi_out_dir,ref_folder)

            if os.path.exists(os.path.join(regi_output_folder, slide_name+'_'+dan_tile_idx+'_ratio'+str(1.0)+'_wsi.png')):
                print('exists skip:',regi_output_folder, slide_name+'_'+dan_tile_idx+'_ratio'+str(1.0)+'_wsi.png')
                #continue

            ######################
            #^can be removed, do not know the meaning of the paragraph

            ###################
            #high res
            sliding_window.sim_matrix = np.load(slide_results+slide_name+'_'+dan_tile_idx+'_'+'sim_matrix.npy')

            # get rough  high res patch
            min_point = np.where(sliding_window.sim_matrix == np.min(sliding_window.sim_matrix))

            top_left_col = min_point[1][0]*step_size[0]-int(step_size[0]*0.5)
            top_left_row = min_point[0][0]*step_size[1]-int(step_size[1]*0.5)
            print('(top_left_col,top_left_row)',(top_left_col,top_left_row))

            level_high = level_fine
            win_size_level_new = (win_size+step_size/(2**level_high)).astype(int)
            print('win_size',win_size,win_size_level_new)

            #crop the most similar patch from the wsi
            window_level_new = cropping(slide_name,wsi_path,np.array((oslide_size[0],oslide_size[1])),np.array((top_left_col,top_left_row)),dan_tile_idx,win_size_level_new,level_high,cropped_save_folder,True)

            # register Dan's to the most similar patch from wsi.
            try:
                #if 1:
                # get RGB chennel, removing the 4th channel
                window_level_new = np.array(window_level_new)[:,:,0:3]
                # to BGR array
                window_level_new = window_level_new[:,:,::-1]
                cv2.imwrite('window_level_new.png',window_level_new)
                print('window_level_new.shape',window_level_new.shape)
                if not os.path.exists(regi_output_folder):
                    os.makedirs(regi_output_folder)
                registered = register_single(ref_img, window_level_new,regi_output_folder,slide_name,dan_tile_idx,MAX_FEATURES_FINE,GOOD_MATCH_PERCENT_FINE,1.0, if_regi = True)
                print('registered.shape',registered.shape)
                print(file_i+'regi success')
                log.write(file_i+', regi success')
            except:
                print(file_i+'regi fail')
                log.write(file_i+', regi fail')

        log.close()
        '''
