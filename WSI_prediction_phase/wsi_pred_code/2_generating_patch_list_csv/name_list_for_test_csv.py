import os
import sys
import glob

slide_path =sys.argv[1]
slide_name=os.path.basename(slide_path)
lists=glob.glob(slide_path+'/*png')
print(len(lists))
f=open('./patch_lists_csv/'+slide_name+'.csv','w')
f.write('patch_path,label'+'\n')
for i in lists:
    f.write(i+',1'+'\n')
f.close()
