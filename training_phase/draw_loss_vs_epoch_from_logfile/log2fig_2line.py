import os
import sys
input_folder=sys.argv[1]
start_epo=0
end_epo=10000000000000000
if len(sys.argv)>2:
    start_epo=int(sys.argv[2])
if len(sys.argv)>3:
    end_epo=int(sys.argv[3])
if len(sys.argv)>4:
    input_folder2=sys.argv[4]
filename=os.path.join(input_folder,'log.txt')
file=open(filename,'r')
newlist=[]#i if i.startswith(' Minibatch') for i in open(filename,'r')]
epoch_num=[]
loss=[]

for i in open(filename,'r'):
    if i.startswith('Finished Epoch'):
        newlist.append(i)
        #print(i)
        temp=i.split('metric =')[-1]
        temp=temp.split('%')[0]

        temp1=i.split(' of')[0]
        temp1=temp1.split(' ')[-1]
        temp1=temp1.split('[')[-1]
        if start_epo<=int(temp1)<=end_epo:
            loss.append(float(temp)/100)
            epoch_num.append(int(temp1))

        #print(temp,temp1)

print(loss)
print(epoch_num)
import matplotlib.pyplot as plt
if  len(sys.argv)>4:
    filename=os.path.join(input_folder2,'log.txt')
    file=open(filename,'r')
    newlist=[]#i if i.startswith(' Minibatch') for i in open(filename,'r')]
    epoch_num2=[]
    loss2=[]

    for i in open(filename,'r'):
        if i.startswith('Finished Epoch'):
            newlist.append(i)
            #print(i)
            temp=i.split('metric =')[-1]
            temp=temp.split('%')[0]

            temp1=i.split(' of')[0]
            temp1=temp1.split(' ')[-1]
            temp1=temp1.split('[')[-1]
            if start_epo<=int(temp1)<=end_epo:
                loss2.append(float(temp)/100)
                epoch_num2.append(int(temp1))

        #print(temp,temp1)

print(loss)
plt.figure(1)

plt.plot(epoch_num,loss, 'b--')
if  len(sys.argv)>4:
    line_up, = plt.plot(epoch_num,loss, 'r--',label='loss on a batch of images')
    line_down, = plt.plot(epoch_num,loss2, 'b--',label='loss on single image')
    plt.legend(handles=[line_up, line_down])
    #plt.plot(epoch_num,loss, 'b--',label='loss on a batch',epoch_num,loss2, 'r--','loss on single image')
plt.xlabel('epoch number')
plt.ylabel('Loss')
plt.title('epochs vs. Training loss')

plt.savefig(os.path.join(input_folder,'loss.png'))
plt.close()
