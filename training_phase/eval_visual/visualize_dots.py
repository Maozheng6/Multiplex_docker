import os
#results='/mnt/blobfuse/train-output/ByMZ/2.18/pred_out/ByMZ-Multi_6layer_99_1_withBN_yb99.5_yerode1_cyan.75_mustdv_2-2-0.1-1_1_1_1_1_1_1_1_1_1-0.0005-0.9995-stain5-mu1.0-sigma1.0-start_stain0-GPU0/'
input_folder='/mnt/blobfuse/train-output/ByMZ/data_dots_labels_for_multiplex/zipped/Inga/Inga_Completed/Inga_Completed_Annot/'
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
results=sys.argv[1]
print('results',results)
files=glob.glob(input_folder+'/*.zip')


base_folder='/mnt/blobfuse/train-output/ByMZ/data_dots_labels_for_multiplex'
unzipped_folder='unzipped'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png-points/'
image_folder='images'#Areeha/2938_cd20h_cd3h_cd4h_cd8h.png'
save_folder='dots_visualization_with_seg_16'#Areeha/'
annotator='Inga'
'''
#unzip
for file_i in files:
    if file_i.endswith('.zip'):
        print('unzip '+file_i+' '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        if not os.path.exists(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4]):
            os.makedirs(os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        shutil.copyfile(file_i,os.path.join(os.path.join(base_folder,unzipped_folder,annotator),os.path.basename(file_i)))
        os.system('unzip '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i))
        os.system('cp *.txt '+os.path.join(base_folder,unzipped_folder,annotator)+'/'+os.path.basename(file_i)[0:-4])
        os.system('rm *.txt')
'''
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

def parse_txt_to_color_and_corrdi(file_i,Larger_than_400_):
    txt_file=open(file_i,'r')
    lines = list(txt_file)
    color=lines[0].rstrip('\n').split('\t')[-1]
    dots_num=int(lines[2].rstrip('\n').split('\t')[-1])
    cordinates_line=lines[3:]
    cordinates_list=[]
    cordinates_list_for_check_value=[]
    for cord in cordinates_line:
        cord_list=cord.rstrip('\n').split('\t')
        if Larger_than_400_:
            cord_list=[round(float(x)/2) for x in cord_list ]
        else:
            cord_list=[round(float(x)) for x in cord_list ]
        cordinates_list.append(cord_list)
        cordinates_list_for_check_value.append(cord_list[0])
        cordinates_list_for_check_value.append(cord_list[1])
    #print(color,cordinates_list)
    if color.endswith('Color'):
        color=color[0:-5]

    return (color,dots_num,cordinates_list,cordinates_list_for_check_value)



if not os.path.exists(save_folder):
    os.makedirs(save_folder)
def visualize_one_patch(folder_i,imname,save_folder,colors_by_one_annotator,precision_dict, recall_dict):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    files_in_i=glob.glob(folder_i+'/*.txt')
    color_idx={'Yellow':2, 'Purple':4,'Black':0, 'Cyan':3, 'Red':1}#'Blue':3,brown,green
    #parse info from txt
    all_parsed_info=[]

    check_400=[]

    for file_i in files_in_i:
        #print('%',file_i)
        [color,dots_num,cordinates_list,cordinates_list_for_check_value]=parse_txt_to_color_and_corrdi(file_i,0)

        check_400+=cordinates_list_for_check_value


    Larger_than_400=0
    if len(check_400)>0 and np.max(np.array(check_400))>=400:
        Larger_than_400=1

    for file_i in files_in_i:
        #print('%',file_i)
        [color,dots_num,cordinates_list,cordinates_list_for_check_value]=parse_txt_to_color_and_corrdi(file_i,Larger_than_400)
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
        if item[0]=='Blue':
            all_parsed_info.remove(item)
            item[0]='Cyan'
            all_parsed_info.append(item)
    if Green_info!=['Green',0,[]]:
        if Yellow_info!=['Yellow',0,[]]:
            all_parsed_info.remove(Yellow_info)
        Yellow_info[1]+=Green_info[1]
        Yellow_info[2]+=Green_info[2]
        all_parsed_info.append(Yellow_info)

    all_parsed_info_dict={x[0]:[x[1],x[2]] for x in all_parsed_info}
    all_parsed_info_dict_sorted = collections.OrderedDict(sorted(all_parsed_info_dict.items()))


    #visualize and compute evaluation
    idx=0
    f, axarr = plt.subplots(1, len(all_parsed_info)+1,dpi=1000)#plt.subplots
    cell_type={'Yellow':'CD3 Double Negative T cell','Black':'CD16 Myeloid Cell','Purple':'CD8 Cytotoxic cell','Red':'CD20 B cell','Cyan':'CD4 helper T cell'}
    for color,item in all_parsed_info_dict_sorted.items():
        [dots_num,cordinates_list]=item
        if color not in colors_by_one_annotator:
            colors_by_one_annotator.append(color)
        #print(os.path.join(results,os.path.basename(imname)[0:-4]+'_'+str(color_idx[color])+'_overlay.png'))
        print(color)
        print(str(color_idx[color]))
        heat_pred=cv2.imread(os.path.join(results,os.path.basename(imname)[0:-4]+'_'+str(color_idx[color])+'.png'))
        if heat_pred.shape[0]>400:
            heat_pred=cv2.resize(heat_pred,(400,400))
        heatmap1,heat_binary= cv2.threshold(heat_pred,70,255,cv2.THRESH_BINARY)
        heat_binary1=copy.deepcopy(heat_binary)
        kernel=np.ones((9,9))/25
        closing = cv2.morphologyEx(heat_binary, cv2.MORPH_CLOSE, kernel)
        if color=='Cyan':
            kernel=np.ones((0,0),np.uint8)
        if color=='Red':
            kernel=np.ones((3,3),np.uint8)
        elif color=='Yellow':
            kernel=np.ones((3,3),np.uint8)
        elif color=='Purple':
            kernel=np.ones((5,5),np.uint8)
        elif color=='Black':
            kernel=np.ones((9,9),np.uint8)
        heat_binary = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        if color=='cyan':
            heat_binary=heat_binary1
        ######################
                ########################
        kernel=np.ones((5,5))
        heat_binary1=copy.deepcopy(heat_binary)
        heat_binary_decide_color=copy.deepcopy(heat_binary)
        heat_binary_decide_color=cv2.dilate(heat_binary_decide_color.astype('uint8'),kernel,iterations=2)

        binary_edge=edge(heat_binary1,12)
        ori_image=cv2.imread(os.path.join(results,os.path.basename(imname)[0:-4]+'_ori.png'))
        if ori_image.shape[0]>400:
            ori_image=cv2.resize(ori_image,(400,400))
        #print('imname:',imname)
        #img_seg=np.clip(ori_image+binary_edge,0,255)
        ori_image_copy=copy.deepcopy(ori_image)
        img_segI=ori_image#np.clip(ori_image+binary_edge,0,255)
        img_seg=np.clip(ori_image+binary_edge,0,255)
        img_seg_copy=copy.deepcopy(img_seg)
        dots_mask=np.zeros_like(ori_image)
        positive_centers=[]
        if len(cordinates_list)>0:
            for center in cordinates_list:
                dots_mask[center[1],center[0],:]=255
                if heat_binary_decide_color[center[1],center[0],0]>0:
                    positive_centers.append((center[0],center[1]))
                    img_dot=cv2.circle(img_seg, (center[0],center[1]), 0, color=[0,255,255], thickness=5)
                else:
                    img_dot=cv2.circle(img_seg, (center[0],center[1]), 0, color=[0,255,0], thickness=5)
            recall=len(positive_centers)/len(cordinates_list)

        else:
            img_dot=img_seg
            recall=1.0
        ######################precision
        ##dots weighted with areas
        '''
        heat_binary2=copy.deepcopy(heat_binary)
        heat_binary2=heat_binary2[:,:,0].astype('uint8')
        _,connected_labels=cv2.connectedComponents(heat_binary2,connectivity =8)


        comp_with_dots=0
        area_with_dots=0
        area_without_dots=0
        for component in range(1,len(np.unique(connected_labels))):
            one_compo=connected_labels==component
            one_compo1=cv2.dilate(one_compo.astype('uint8')*255,kernel,iterations=2)
            if np.sum(dots_mask[:,:,0]*one_compo1)>0:
                comp_with_dots+=1
                area_with_dots=area_with_dots+np.sum(one_compo)
            else:
                area_without_dots=area_without_dots+np.sum(one_compo)

        if (len(np.unique(connected_labels))-1)>0:
            #precision=comp_with_dots/(len(np.unique(connected_labels))-1)
            precision=area_with_dots/(area_without_dots+area_with_dots)
        else:
            precision=1
        #print('precision,recall',precision,recall)
        #precision_dict[color].append([comp_with_dots,(len(np.unique(connected_labels))-1)])
        precision_dict[color].append([area_with_dots,area_without_dots+area_with_dots])
        recall_dict[color].append([len(positive_centers),len(cordinates_list)])
        #print('precision_dict',precision_dict)
        #print('recall_dict',recall_dict)
        '''
        heat_binary2=copy.deepcopy(heat_binary)
        heat_binary2=heat_binary2[:,:,0].astype('uint8')

        connectivity=8
        # Perform the operation
        output = cv2.connectedComponentsWithStats(heat_binary2, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        connected_labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]

        comp_with_dots=0
        area_with_dots=0
        area_without_dots=0
        print('cordinates_list',cordinates_list)
        print('centroids',centroids)
        print(len(np.unique(connected_labels)),len(centroids))
        founded_GT=[]

        true_positive_detection=[]
        false_positive_detection=[]
        cordinates_list_copy=copy.deepcopy(cordinates_list)
        img_dot_pr=img_seg_copy
        for centroids_id in range(1,len(np.unique(connected_labels))):
            gt_distance={}

            for gt_dot in cordinates_list_copy:

                distance=np.sqrt((centroids[centroids_id][0]-gt_dot[0])**2+(centroids[centroids_id][1]-gt_dot[1])**2)
                gt_distance[distance]=gt_dot

            if len(sorted(gt_distance.keys()))!=0:
                gt_nearest_dis=sorted(gt_distance.keys())[0]
                print('nearest',centroids[centroids_id],gt_distance[gt_nearest_dis],gt_nearest_dis)
                if gt_nearest_dis<16:
                    true_positive_detection.append(centroids[centroids_id])
                    founded_GT.append(gt_distance[gt_nearest_dis])
                    cordinates_list_copy.remove(gt_distance[gt_nearest_dis])
                    img_dot_pr=cv2.circle(img_seg_copy, (gt_distance[gt_nearest_dis][0],gt_distance[gt_nearest_dis][1]), 0, color=[0,0,255], thickness=7)
                    img_dot_pr=cv2.circle(img_seg_copy, (int(centroids[centroids_id][0]),int(centroids[centroids_id][1])), 16, color=[0,0,255], thickness=2)

                else:
                    false_positive_detection.append(centroids[centroids_id])
                    img_dot_pr=cv2.circle(img_seg_copy, (int(centroids[centroids_id][0]),int(centroids[centroids_id][1])), 16, color=[0,255,0], thickness=2)

            else:
                false_positive_detection.append(centroids[centroids_id])
                img_dot_pr=cv2.circle(img_seg_copy, (int(centroids[centroids_id][0]),int(centroids[centroids_id][1])), 16, color=[0,255,0], thickness=2)

        for cord in cordinates_list_copy:
            #if cord not in founded_GT:
            img_dot_pr=cv2.circle(img_seg_copy, (cord[0],cord[1]), 0, color=[0,255,0], thickness=7)

        if (len(np.unique(connected_labels))-1)>0:
            #precision=comp_with_dots/(len(np.unique(connected_labels))-1)
            precision=len(true_positive_detection)/(len(centroids)-1+0.000001)
        else:
            precision=1
        #print('precision,recall',precision,recall)
        #precision_dict[color].append([comp_with_dots,(len(np.unique(connected_labels))-1)])
        precision_dict[color].append([len(true_positive_detection),len(centroids)-1])
        #recall_dict[color].append([len(positive_centers),len(cordinates_list)])
        recall_dict[color].append([len(founded_GT),len(cordinates_list)])
        #print('precision_dict',precision_dict)
        #print('recall_dict',recall_dict)
        ########################
        if idx==0:
            axarr[idx].imshow(ori_image_copy[:,:,::-1]/255)
            axarr[idx].set_title('RGB',fontsize=6)
            axarr[idx].axis('off')
        '''
        axarr[idx+1].imshow(img_dot[:,:,::-1]/255)
        axarr[idx+1].set_title(color+'\n'+cell_type[color]+'\n'+'found dots/total dots : '+str(len(positive_centers))+'/'+str(dots_num)+'\n'+'area w/ dots/ total areas : %.2f'%(area_with_dots/(area_without_dots+area_with_dots+0.01)),fontsize=4)
        axarr[idx+1].axis('off')
        '''
        axarr[idx+1].imshow(img_dot_pr[:,:,::-1]/255)
        axarr[idx+1].set_title(color+'\n'+cell_type[color]+'\n'+'detected dots/total dots : '+str(len(founded_GT))+'/'+str(len(cordinates_list))+' = %.0f'%((len(founded_GT)/(len(cordinates_list)+0.0001))*100)+'%'+'\n'+'possitive detection/ total detection : %.0f'%((len(true_positive_detection)/(len(centroids)-1+0.00001))*100)+'%',fontsize=3)
        axarr[idx+1].axis('off')
        idx+=1
        #####################
    #print('save name',os.path.join(save_folder,os.path.basename(imname)))
    f.savefig(os.path.join(save_folder,os.path.basename(imname)), bbox_inches='tight', pad_inches=0)
    plt.close(f)

    return colors_by_one_annotator,precision_dict,recall_dict


annotators=['Areeha','Christian','Emily','Inga']
colors_by_annotators={}
#for annotator in annotators:
def process_one_annotator(annotator):
    im_names=glob.glob(os.path.join(base_folder,image_folder,annotator)+'/*.png')
    colors_by_one_annotator=[]
    precision_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    recall_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    for imname in im_names:
        #print(imname)
        dots_folder=os.path.join(base_folder,unzipped_folder,annotator,os.path.basename(imname)+'-points')
        #print(dots_folder,os.path.exists(dots_folder))
        save_dir=os.path.join(base_folder,save_folder,annotator)
        #print('save_dir',save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            #print('##############################################')
        #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        colors_by_one_annotator,precision_dict,recall_dict=visualize_one_patch(dots_folder,imname,save_dir,colors_by_one_annotator,precision_dict,recall_dict)
    colors_by_annotators[annotator]=colors_by_one_annotator

def process_one_patch(imname,precision_dict,recall_dict):

    colors_by_one_annotator=[]
    annotator=imname.split('/')[-2]
    dots_folder=os.path.join(base_folder,unzipped_folder,annotator,os.path.basename(imname)+'-points')
    #print(dots_folder,os.path.exists(dots_folder))
    save_dir=os.path.join(results,save_folder,annotator)
    #print('save_dir',save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        #print('##############################################')
    #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    colors_by_one_annotator,precision_dict1,recall_dict1=visualize_one_patch(dots_folder,imname,save_dir,colors_by_one_annotator,precision_dict,recall_dict)
    return precision_dict1,recall_dict1

def main():
    global precision_dict
    global recall_dict
    with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
        for number, pr in zip(im_names, executor.map(process_one_patch, im_names, chunksize=1)):
            precision_dict,recall_dict=pr
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
    precision_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    recall_dict={'Black':[],'Red':[],'Yellow':[],'Cyan':[],'Purple':[]}
    for annotator in ['Areeha','Christian','Emily','Inga']:#,'Christian','Emily'Areeha]:
        im_names+=glob.glob(os.path.join(base_folder,image_folder,annotator)+'/*.png')
    precision_dict,recall_dict=main1()
    #process_one_annotator('Areeha')

    pickle_out = open("recall_dict.pickle","wb")
    pickle.dump(recall_dict, pickle_out)
    pickle_out.close()
    recall_final={}
    precision_final={}
    for key in recall_dict.keys():
        recall_array=np.array(recall_dict[key])
        recall_color=np.sum(recall_array[:,0])/(np.sum(recall_array[:,1])+0.0001)
        recall_final[key]=recall_color
    print('recall_final',recall_final)
    pickle_out = open("precision_dict.pickle","wb")
    pickle.dump(precision_dict, pickle_out)
    pickle_out.close()
    for key in precision_dict.keys():
        precision_array=np.array(precision_dict[key])
        precision_color=np.sum(precision_array[:,0])/(np.sum(precision_array[:,1])+1)
        precision_final[key]=precision_color
    print('precision_final',precision_final)
    F1={}
    for key in precision_dict.keys():
        #print(precision_final[key],recall_final[key])
        F1[key]=2*precision_final[key]*recall_final[key]/(precision_final[key]+recall_final[key])
    print('F1',F1)
