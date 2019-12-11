import os
import sys
input_folder=sys.argv[1]
start_epoch=0
if len(sys.argv)>2:
    start_epoch=int(sys.argv[2])
filename=os.path.join(input_folder,'log.txt')
file=open(filename,'r')
newlist=[]#i if i.startswith(' Minibatch') for i in open(filename,'r')]
epoch_num=[]
loss=[]
for i in open(filename,'r'):
    if i.startswith('Finished Epoch') :
        newlist.append(i)
        #print(i)
        temp=i.split('loss = ')[-1]
        temp=temp.split('*')[0]
        temp1=i.split(' of')[0]
        temp1=temp1.split(' ')[-1]
        temp1=temp1.split('[')[-1]
        if int(temp1)>=start_epoch:
            loss.append(float(temp))
            epoch_num.append(int(temp1))

        #print(temp,temp1)

print(loss)
print(epoch_num)
import matplotlib.pyplot as plt

plt.figure(1)

plt.plot(epoch_num,loss, 'b--')
plt.xlabel('epoch number')
plt.ylabel('Loss')
plt.title('epochs vs. Training loss')

plt.savefig(os.path.join(input_folder,'loss.png'))
plt.close()
