import os
import glob
import shutil
import cv2
import copy
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import collections
import pickle
import sys
import json
################################################
#parameters to change

#The folder of test set
test_set_folder='../19test_set/'
#prediction results folder
prediction_results_folder='./19test_10x20x_v3/'
#suffix of the files in prediction_results_folder, it's for getting the path of files in the 'subfolder', it can be 'png' or 'npy'
file_suffix='png'
#output folder to save the visual results generated by this script
save_visual_results_folder='./results_visual_19test_10x20xcomb'

######################################################
def edge(img,dim):
    img2=np.zeros((img.shape[0],img.shape[1],3)).astype('float')
    img1=cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    img1=cv2.dilate(img1,kernel,iterations = 1)

    for i in range(img2.shape[2]):
        if dim==12:
            img2[:,:,1]=img1.astype('float')
            img2[:,:,2]=img1.astype('float')
        else:
            if i==dim:
                img2[:,:,i]=img1.astype('float')
            else:
                img2[:,:,i]=-img1.astype('float')

    '''
    for i in img1.shape[2]:
        if i!=dim:
            img1[:,:,i]=0
    '''
    return img2

def parse_txt_to_color_and_corrdi(file_i):
    txt_file=open(file_i,'r')
    lines = list(txt_file)
    color=lines[0].rstrip('\n').split('\t')[-1]
    #print('lines',lines)
    dots_num=int(lines[2].rstrip('\n').split('\t')[-1])
    cordinates_line=lines[3:]
    cordinates_list=[]

    is_800=False
    for cord in cordinates_line:
        cord_list=cord.rstrip('\n').split('\t')
        cord_list=[round(float(x)-1) for x in cord_list ]
        cordinates_list.append(cord_list)
        #if cord_list[0]>400 or cord_list[1]>400:
        #    is_800 = True
        #    break
    if is_800 == True:
        cordinates_list=[]
        for cord in cordinates_line:
            cord_list=cord.rstrip('\n').split('\t')
            cord_list=[round((float(x)-1)/2) for x in cord_list ]
            cordinates_list.append(cord_list)
    #print(color,cordinates_list)
    if color.endswith('Color'):
        color=color[0:-5]

    return (color,dots_num,cordinates_list)




def process_one_patch_txt(folder_i,save_folder_processed_dots,imname):

    files_in_i=glob.glob(folder_i+'/*.txt')
    #parse info from txt
    all_parsed_info=[]


    for file_i in files_in_i:
        #print('%',file_i)
        [color,dots_num,cordinates_list]=parse_txt_to_color_and_corrdi(file_i)
        all_parsed_info.append([color,dots_num,cordinates_list])

    #merge Green and Yellow from Areeha, remove Brown, change Blue to Cyan

    Green_info=['Green',0,[]]
    Yellow_info=['Yellow',0,[]]
    all_parsed_info1=copy.deepcopy(all_parsed_info)
    for item in all_parsed_info1:
        if item[0]=='Green':
            Green_info=item
            all_parsed_info.remove(item)
        if item[0]=='Yellow':
            Yellow_info=item
        if item[0]=='Brown':
            all_parsed_info.remove(item)
        if item[0]== 'K17-' or  item[0]== 'K17-neg tumor' or item[0]== 'K17-NEGATIVE TUMOR' or  item[0]=='K17 NEGATIVE TUMOR' or  item[0].lower()=='k17 neg' or  item[0].lower()=='K17-negative tumor' or  item[0].lower()=='k17 - neg' or item[0].lower()=='k17-neg' or  item[0].lower()=='k17-negative' or item[0].lower()=='k17 negative tumor' or item[0].lower()=='k17-negative tumor' or  item[0].lower()== 'k17-neg tumor' or item[0].lower()=='k17-neh':
            all_parsed_info.remove(item)
            #need to recover the following 2 line
            item[0]='K17 neg'
            all_parsed_info.append(item)
        if  item[0].lower()=='k17' or item[0].lower()=='k17+ tumor':
            all_parsed_info.remove(item)

            #need to recover
            item[0]='K17'
            all_parsed_info.append(item)
    if Green_info!=['Green',0,[]]:
        if Yellow_info!=['Yellow',0,[]]:
            all_parsed_info.remove(Yellow_info)
        Yellow_info[1]+=Green_info[1]
        Yellow_info[2]+=Green_info[2]
        all_parsed_info.append(Yellow_info)

    all_parsed_info_dict={x[0]:[x[1],x[2]] for x in all_parsed_info}
    all_parsed_info_dict_sorted = collections.OrderedDict(sorted(all_parsed_info_dict.items()))
    if not os.path.exists(save_folder_processed_dots):
        os.makedirs(save_folder_processed_dots)
    image_name=imname.split('/')[-1]
    with open(save_folder_processed_dots+'/'+image_name+'-points'+'.json', 'w') as f:
        ss = json.dumps(all_parsed_info_dict_sorted)
        f.write(ss)

def visualize_one_patch(folder_i,save_folder_processed_dots,imname,save_folder,colors_by_one_annotator,precision_dict, recall_dict):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    color_idx={'CD3':2, 'CD8':4,'CD16':0, 'CD4':3, 'CD20':1, 'K17':5, 'K17 neg':6}
    #load dots
    print(imname)
    image_name=imname.split('/')[-1]
    with open(save_folder_processed_dots+'/'+image_name+'-points'+'.json', 'r') as f:
       all_parsed_info_dict_sorted = collections.OrderedDict(json.load(f))
    #restore all_parsed_info as a list
    #this is how the dict is built# all_parsed_info_dict={x[0]:[x[1],x[2]] for x in all_parsed_info}
    all_parsed_info=[[key,all_parsed_info_dict_sorted[key][0],all_parsed_info_dict_sorted[key][1]] for key in all_parsed_info_dict_sorted.keys()]
    #visualize and compute evaluation
    idx=0
    f, axarr = plt.subplots(1, 7,dpi=1000)#plt.subplots
    cell_type={'CD3':'CD3 Double Negative T cell','CD16':'CD16 Myeloid Cell','CD8':'CD8 Cytotoxic cell','CD20':'CD20 B cell','CD4':'CD4 helper T cell','K17':'K17 +','K17 neg':'K17 -'}

    if file_suffix == 'png':
        print('for glob',os.path.join(prediction_results_folder,'*'+os.path.basename(imname)[0:- 4]+ '*argmax.png'))
        heat_pred_argmax_name=glob.glob(os.path.join(prediction_results_folder,'*'+ os.path.basename(imname)[0:- 4]+ '*argmax.png'))
        print('heat_pred_argmax_name',heat_pred_argmax_name)
        heat_pred_argmax_name = heat_pred_argmax_name[0]
        heat_pred_argmax = cv2.imread( heat_pred_argmax_name,0)

    elif file_suffix == 'npy':

        print('for glob',os.path.join(prediction_results_folder,'*'+os.path.basename(imname)[0:- 4]+ '*.npy'))
        heat_pred_argmax_name=glob.glob(os.path.join(prediction_results_folder,'*'+os.path.basename(imname)[0:- 4]+ '*.npy'))
        print('heat_pred_argmax_name',heat_pred_argmax_name)
        heat_pred_argmax_name = heat_pred_argmax_name[0]
        heat_pred_argmax = np.load(heat_pred_argmax_name)

    print('Mheat_pred_argmax.shape',heat_pred_argmax.shape,np.unique(heat_pred_argmax))
    corlor_idx_iverse = dict((v,k) for k, v in color_idx.items())
    print('corlor_idx_iverse',corlor_idx_iverse)
    #######################################################

    for color,item in all_parsed_info_dict_sorted.items():
        [dots_num,cordinates_list]=item
        if color not in colors_by_one_annotator:
            colors_by_one_annotator.append(color)
        print('color',color)
        print('color_idx',color_idx)
        print('color_idx[color]',color_idx[color])
        print('str(color_idx[color])',str(color_idx[color]))
        print('22222222222222222222222222222')
        ####################################
        #argmax process
        heat_binary=np.zeros((heat_pred_argmax.shape[0],heat_pred_argmax.shape[1],3)).astype('uint8')

        heat_binary[:,:,0] = (heat_pred_argmax==color_idx[color]+1).astype('uint8')*255
        heat_binary[:,:,1] = (heat_pred_argmax==color_idx[color]+1).astype('uint8')*255
        heat_binary[:,:,2] = (heat_pred_argmax==color_idx[color]+1).astype('uint8')*255
        heat_binary = cv2.resize(heat_binary,(1920,1200),cv2.INTER_NEAREST)

        ###############################
               ########################
        kernel=np.ones((5,5))
        heat_binary1=copy.deepcopy(heat_binary)
        heat_binary_decide_color=copy.deepcopy(heat_binary)
        #heat_binary_decide_color=cv2.dilate(heat_binary_decide_color.astype('uint8'),kernel,iterations=2)

        binary_edge=edge(heat_binary1,12)
        ori_image_name=os.path.basename(imname)[0:-4]+'.png'
        print('basefolder+ori_image_name',os.path.join(test_set_folder,ori_image_name))
        ori_image=cv2.imread(os.path.join(test_set_folder,ori_image_name))

        print('ori_image.shape1',ori_image.shape)
        ori_image = cv2.resize(ori_image,(binary_edge.shape[1],binary_edge.shape[0]))
        print('ori_image.shape2',ori_image.shape)
        #if ori_image.shape[0]>400:
        #    ori_image=cv2.resize(ori_image,(400,400))
        #print('imname:',imname)
        #img_seg=np.clip(ori_image+binary_edge,0,255)
        ori_image_copy=copy.deepcopy(ori_image)
        img_seg=ori_image#np.clip(ori_image+binary_edge,0,255)
        img_seg=np.clip(ori_image+binary_edge,0,255)
        img_seg_copy=copy.deepcopy(img_seg)
        dots_mask=np.zeros_like(ori_image)
        positive_centers=[]
        if len(cordinates_list)>0:
            for center in cordinates_list:
                dots_mask[center[1],center[0],:]=255
                if heat_binary_decide_color[center[1],center[0],0]>0:
                    positive_centers.append((center[0],center[1]))
                    #img_dot=cv2.circle(img_seg, (center[0],center[1]), 0, color=[0,0,255], thickness=5)
                    img_dot=cv2.drawMarker(img_seg, (center[0],center[1]),  color=[0,0,     255],markerType= cv2.MARKER_CROSS,markerSize=10,            thickness=2)

                else:
                    #img_dot=cv2.circle(img_seg, (center[0],center[1]), 0, color=[0,255,0], thickness=5)
                    img_dot=cv2.drawMarker(img_seg, (center[0],center[1]),  color=[0,      255,     0], markerType=cv2.MARKER_CROSS,markerSize=10,thickness=2)
            recall=len(positive_centers)/len(cordinates_list)

        else:
            img_dot=img_seg
            recall=1.0
        print('img_dot.shape',img_dot.shape)
        if not os.path.exists('./dots_img/'):
            os.mkdir('./dots_img/')
        cv2.imwrite('./dots_img/'+os.path.basename(imname)[0:-4]+'.png',img_dot)
        #######################################
        #precision area and dots
        ########################################
        heat_binary2=copy.deepcopy(heat_binary)
        heat_binary2=heat_binary2[:,:,0].astype('uint8')
        _,connected_labels=cv2.connectedComponents(heat_binary2,connectivity =8)


        comp_with_dots=0
        area_with_dots=0
        area_without_dots=0
        effective_areas_num = 0
        p_mode = 'count'
        for component in range(1,len(np.unique(connected_labels))):
            one_compo=connected_labels==component
            one_compo1=one_compo#cv2.dilate(one_compo.astype('uint8')*255,kernel,iterations=2)
            effective_areas_num+=1
            if np.sum(dots_mask[:,:,0]*one_compo1)>0:
                #count
                if p_mode=='count':
                    area_with_dots=area_with_dots+np.sum((dots_mask[:,:,0]*one_compo1).astype('int'))
                else:
                    #area weighted
                    area_with_dots=area_with_dots+np.sum(one_compo)

                #print('positive dots num',np.max(dots_mask[:,:,0]),np.max(one_compo1),np.sum((dots_mask[:,:,0]*one_compo1).astype('uint8')))
            else:
                if p_mode == 'count':
                    #count
                    area_without_dots=area_without_dots+1
                else:
                    #area weighted
                    area_without_dots=area_without_dots+np.sum(one_compo)

        if p_mode!='count' and (len(np.unique(connected_labels))-1)>0 and effective_areas_num>0:
            #precision=comp_with_dots/(len(np.unique(connected_labels))-1)
            precision=area_with_dots/(area_without_dots+area_with_dots)
        elif p_mode=='count' and (len(np.unique(connected_labels))-1)>0:
            precision=area_with_dots/(area_without_dots+area_with_dots)
        else:
            precision=1

        ######################################
        #print('precision,recall',precision,recall)
        #precision_dict[color].append([comp_with_dots,(len(np.unique(connected_labels))-1)])
        #
        precision_dict[color].append([area_with_dots,area_without_dots+area_with_dots])
        recall_dict[color].append([len(positive_centers),len(cordinates_list)])
        #recall_dict[color].append([len(founded_GT),len(cordinates_list)])
        #print('precision_dict',precision_dict)
        #print('recall_dict',recall_dict)
        ########################
        #color_order_draw={'Black':1,'Red':2,'Yellow':3,'Cyan':4,'Purple':5}
        color_order_draw={'CD16':1,'CD20':2,'CD3':3,'CD4':4,'CD8':5,'K17':6,'K17 neg':7}
        draw_idx=color_order_draw[color]
        axarr[draw_idx-1].imshow(img_dot[:,:,::-1]/255)
        axarr[draw_idx-1].axis('off')
        idx+=1
        #####################
    #print('save name',os.path.join(save_folder,os.path.basename(imname)))
    f.savefig(os.path.join(save_folder,os.path.basename(imname)), bbox_inches='tight', pad_inches=0)
    plt.close(f)

    return colors_by_one_annotator,precision_dict,recall_dict


def process_one_patch(imname,precision_dict,recall_dict):

    colors_by_one_annotator = []
    annotator = imname.split('/')[-2]
    dots_folder = os.path.join(test_set_folder,os.path.basename(imname)+'-points')
    save_folder_processed_dots = os.path.join(test_set_folder,'processed_dots')
    #print(dots_folder,os.path.exists(dots_folder))
    save_dir = os.path.join(save_visual_results_folder,save_folder,annotator)
    #print('save_dir',save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        #print('##############################################')
    #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ###################################
    #process all the dot txt
    ####################################
    process_one_patch_txt(dots_folder,save_folder_processed_dots,imname)

    precision_dict1={}
    recall_dict1={}
    colors_by_one_annotator,precision_dict1,recall_dict1=visualize_one_patch(dots_folder,save_folder_processed_dots,imname,save_dir,colors_by_one_annotator,precision_dict,recall_dict)
    return precision_dict1,recall_dict1
def process_one_patch_parall(imname):
    #precision_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    #recall_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    precision_dict={'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[],'K17':[],   'K17 neg':[]}
    recall_dict={'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[],'K17':[],   'K17 neg':[]}
    colors_by_one_annotator = []
    annotator = imname.split('/')[-2]
    dots_folder = os.path.join(test_set_folder,os.path.basename(imname)+'-points')
    save_folder_processed_dots = os.path.join(test_set_folder,'processed_dots')
    #print(dots_folder,os.path.exists(dots_folder))
    save_dir = save_visual_results_folder
    #print('save_dir',save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        #print('##############################################')
    #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    ###################################
    #process all the dot txt
    ####################################
    process_one_patch_txt(dots_folder,save_folder_processed_dots,imname)

    precision_dict1={}
    recall_dict1={}
    colors_by_one_annotator,precision_dict1,recall_dict1=visualize_one_patch(dots_folder,save_folder_processed_dots,imname,save_dir,colors_by_one_annotator,precision_dict,recall_dict)
    return precision_dict1,recall_dict1
def main():
    global precision_dict
    global recall_dict
    with concurrent.futures.ProcessPoolExecutor( max_workers=20) as executor:
        for number, pr in zip(im_names, executor.map(process_one_patch_parall, im_names, chunksize=1)):
            precision_dict_t,recall_dict_t=pr
            print('pr',pr)
            for key in precision_dict.keys():
                precision_dict[key]=precision_dict[key]+precision_dict_t[key]
                recall_dict[key]=recall_dict[key]+recall_dict_t[key]
            print('%s is prime: %s' % (number, pr))

    return precision_dict,recall_dict


def main1():
    global precision_dict
    global recall_dict
    for imname in im_names:
        precision_dict,recall_dict=process_one_patch(imname,precision_dict,recall_dict)
        print(precision_dict)

    return precision_dict,recall_dict
if __name__ == '__main__':

    im_names=[]
    #precision_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    #recall_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    precision_dict={'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[],'K17':[],   'K17 neg':[]}
    recall_dict={'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[],'K17':[],   'K17 neg':[]}
    precision_dict_5 = {'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[]}
    recall_dict_5 = {'CD16':[],'CD20':[],'CD3':[],'CD4':[],'CD8':[]}

    stain_num=len(precision_dict.keys())

    im_names=glob.glob(test_set_folder+'/*points')
    im_names=[x[0:-len('-points')] for x in im_names]


    print('im_names',len(im_names),im_names)
    precision_dict,recall_dict=main()
    print('recall_dict',recall_dict)
    pickle_out = open("recall_dict.pickle","wb")
    pickle.dump(recall_dict, pickle_out)
    pickle_out.close()
    recall_final={}
    precision_final={}
    for key in recall_dict.keys():
        recall_array=np.array(recall_dict[key])
        print(recall_array)
        if len(recall_array)==0:
            recall_color=0.0
        else:
            recall_color=np.sum(recall_array[:,0])/(np.sum(recall_array[:,1])+0.0001)
        recall_final[key]=recall_color

    mean_recall=0
    for key in recall_final.keys():
        mean_recall+=recall_final[key]
    mean_recall=mean_recall/(stain_num)
    print('mean_recall',mean_recall,'recall_final',recall_final)

    ####################################
    # compute mean_recall for only 5 stains
    mean_recall_5=0
    for key in recall_final.keys():
        mean_recall_5+=recall_final[key]
    mean_recall_5=mean_recall_5/(len(recall_final.keys()))
    #print('mean_recall_5',mean_recall_5)
    #############################################
    pickle_out = open("precision_dict.pickle","wb")
    pickle.dump(precision_dict, pickle_out)
    pickle_out.close()
    for key in precision_dict.keys():
        precision_array=np.array(precision_dict[key])
        if len(recall_array)==0:
            precision_color=0.0
        else:
            precision_color=np.sum(precision_array[:,0])/(np.sum(precision_array[:,1])+0.0000001)
        precision_final[key]=precision_color

    mean_precision=0
    for key in precision_final.keys():
        mean_precision+=precision_final[key]
    mean_precision=mean_precision/(stain_num)
    print('mean_precision',mean_precision,'precision_final',precision_final)

    #########################################
    #compute mean_precision for only 5 stains
    mean_precision_5=0
    for key in precision_dict_5.keys():
        mean_precision_5+=precision_final[key]
    mean_precision_5 = mean_precision_5/(len(precision_dict_5.keys()))
    #print('mean_precision_5',mean_precision_5)



    ######################################

    F1={}
    for key in precision_dict.keys():
        #print(precision_final[key],recall_final[key])
        F1[key]=2*precision_final[key]*recall_final[key]/(precision_final[key]+recall_final[key]+0.00000001)

    F1_final=F1
    mean_F1=0
    for key in F1_final.keys():
        mean_F1+=F1_final[key]
    #!!!remove red for average
    mean_F1 = mean_F1/(stain_num)
    print('mean_F1',mean_F1,'F1_final',F1_final)

    mean_F1_5=0
    for key in precision_dict_5.keys():
        mean_F1_5+=F1_final[key]
    mean_F1_5 = mean_F1_5/(len(precision_dict_5.keys()))
    #print('mean_F1_5',mean_F1_5)
