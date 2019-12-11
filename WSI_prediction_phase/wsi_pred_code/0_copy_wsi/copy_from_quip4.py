import subprocess
import os
import paramiko
import concurrent.futures
#name_list= ['L6745_','O0135_','O3936_']
#['L22800_20','L28352_2','L29978_2','M036_15','P24146_90','N8033_95','M28417_2','N24852_5','N29055_30','O21747_95','P0992_80','P28191_80','P4211_0','M3669_50','N27093_95','N3908_70','O6218_30','P22034_2','P29193_65', 'P528_30','M4213_2','N27243_5','N5039_5','O8372_10','P24146_90','P304_80','P670_5','N24178_80','N27702_20','N8945_50','P0533_50','P24230_20','P31681_40']
'''
#make the name complete
name_list = [x.split('_')[0]+'-multires.tif' for x in name_list]
print(len(name_list))

#remove existing slides
for i in name_list:
    if os.path.exists('./'+i):
        print('exists',i)
        name_list.remove(i)

print(name_list)
'''
name_list=  ['L6745-multires.tif','O3936-multires.tif','O0135-multires.tif']#['N22800-multires.tif', 'L28352-multires.tif', 'M036-multires.tif', 'P24146-multires.tif', 'N8033-multires.tif', 'N24852-multires.tif', 'N29055-multires.tif', 'O21747-multires.tif', 'P0992-multires.tif', 'P28191-multires.tif', 'M3669-multires.tif', 'N27093-multires.tif', 'N3908-multires.tif', 'O6218-multires.tif', 'P22034-multires.tif', 'P29193-multires.tif', 'P528-multires.tif', 'M4213-multires.tif', 'N27243-multires.tif', 'O8372-multires.tif','P24146-multires.tif', 'P304-multires.tif', 'P670-multires.tif', 'N24178-multires.tif', 'N27702-multires.tif', 'N8945-multires.tif', 'P0533-multires.tif', 'P24230-multires.tif', 'P31681-multires.tif']
print(len(name_list))
print('scp mazhao6@quip4.uhmc.sunysb.edu:/data/ug3_seer/img/k17/new_images/roche_40/'+name_list[0])
#subprocess.call('scp mazhao6@quip4.uhmc.sunysb.edu:/data/ug3_seer/img/k17/new_images/roche_40/'+name_list[0]+' .',shell=True)
#subprocess.call('BMI.2013!')
def ssh_copy(slide_name):
    COMP = "quip4.uhmc.sunysb.edu"
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(COMP, username="mazhao6", password="BMI.2013!", allow_agent = False)
    src = "/data/ug3_seer/img/k17/new_images/roche_40/"+slide_name
    dst = "/gpfs/scratch/mazhao/multiplex-wsi/"+slide_name
    print(src)
    ftp = ssh.open_sftp()
    try:
        ftp.get(src , dst)
    except:
        return 'fail '+src
    ftp.close()
    return 'success '+src

def main():
    copy_log=open('copy_log.txt','w')
    with concurrent.futures.ProcessPoolExecutor( max_workers=10) as executor:
        for number, prime in zip(name_list, executor.map(ssh_copy, name_list, chunksize=1)):
            print('%s is prime: %s' % (number, prime))
            copy_log.write(prime)
    copy_log.close()


if __name__ == '__main__':
    a=1
    main()
