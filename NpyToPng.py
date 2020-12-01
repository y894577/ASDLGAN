import numpy as np
from PIL import Image
import os

dir="F:\\python\\DCGAN-for-Steganography-master\\pros\\"
dest_dir="F:\\python\\DCGAN-for-Steganography-master\\pros\\png"
def npy2jpg(dir,dest_dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    file=dir+'1.npy'
    con_arr=np.load(file)
    count=0
    for con in con_arr:
        arr=con[0]
        label=con[1]
        print(np.argmax(label))
        arr=arr*255
        #arr=np.transpose(arr,(2,1,0))
        arr=np.reshape(arr,(3,112,112))
        r=Image.fromarray(arr[0]).convert("L")
        g=Image.fromarray(arr[1]).convert("L")
        b=Image.fromarray(arr[2]).convert("L")

        img=Image.merge("RGB",(r,g,b))

        label_index=np.argmax(label)
        img.save(dest_dir+str(label_index)+"_"+str(count)+".jpg")
        count=count+1

if __name__=="__main__":
    npy2jpg(dir,dest_dir)