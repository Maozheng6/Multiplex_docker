#import concurrent.futures
import numpy as np

names=[x.rstrip('\n') for x in open('./maozheng_tumor_patch_list.txt')]


def check(i):
    print(i)
    a=np.load(i)
    #if a.shape!=(1,4,240,240):
        #print(i)


def main():
    with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
        for number, prime in zip(names, executor.map(check, names, chunksize=10)):
            print('%s is prime: %s' % (number, prime))


for ii in names:
    check(ii)

