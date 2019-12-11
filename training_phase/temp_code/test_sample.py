import numpy as np
ins_id=0
class_iter_idx=0
while ins_id < 13:
            if ins_id>2 and ( ins_id%6 == 0 or (ins_id-1)%6 == 0):
                class_iter_idx=10
                dots_flag=1
            elif ins_id%2 == 0 :
                class_iter_idx=np.random.choice(5)
                print('class_iter_idx1',class_iter_idx)
                class_iter_idx=2*class_iter_idx
                print('class_iter_idx2',class_iter_idx)
                dots_flag=0
            else:
                class_iter_idx+=1
                dots_flag=0   
            print(class_iter_idx,dots_flag)
            ins_id+=1
