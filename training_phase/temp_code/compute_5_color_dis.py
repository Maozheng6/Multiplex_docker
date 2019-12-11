import numpy as np
#stain_OD=np.array([[82.60197,90.8649,98.27977],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[116.36674,16.986359, 12.227117],[35.247486,82.52399,34.31515]])
stain_OD=np.array([[26.784452,20.88244,14.707071],[14.87054,138.48598,65.88664],[ 17.725187,24.566036,76.55223],[50.2502, 11.4816885, 8.962005],[35.247486,82.52399,34.31515]])

stain_RGB=np.exp(-stain_OD*np.log(255)/255)*255-1
dis_matrix=np.zeros((5,5))
for i in range(5):
    for j in range(5):
        s1=stain_RGB[i]
        s2=stain_RGB[j]
        dis=np.sqrt(np.sum((s1-s2)*(s1-s2)))
        if dis==0:
            dis_matrix[i,j]=255
        else:
            dis_matrix[i,j]=dis
        
print(np.amin(dis_matrix,axis=0)/2)

