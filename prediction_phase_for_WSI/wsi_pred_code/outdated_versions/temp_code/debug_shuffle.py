import random

a=list(range(10))
a=[str(x) for x in a ]
b=[]
for x in a:
    b.append([x,x,1])
random.shuffle(b)
print('b',b)
#c=[x[0] for x in b]
#print(c)
#d=list(set(c))
#print(len(d))
