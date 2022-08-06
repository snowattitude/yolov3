import os
import numpy as np



#--------------------------------------------------
#设置文件路径
#--------------------------------------------------

total_path1 = 'D:\coco\images'
total_path2 = 'D:\PyTorch-YOLOv3\PyTorch-YOLOv3\data\coco'
train_picture_path = total_path1+'\\train2014'
val_picture_path = total_path1+'\\val2014'
train_path_file = total_path2+'\\trainvalno5k.txt'
val_path_file = total_path2+'\\5k.txt'

#--------------------------------------------------
#读取照片名称并加上相对路径
#--------------------------------------------------

pathdir_train = os.listdir(train_picture_path)
pathdir_train = [os.path.join('\images\\train2014\\', filename) for filename in pathdir_train]
pathdir_val = os.listdir(val_picture_path)
pathdir_val = [os.path.join('\images\\val2014\\', filename) for filename in pathdir_val]

#-----------------------
#第一种
#-----------------------

# with open(train_path_file,'w') as train_path:
#     for i in pathdir:
#         train_path.write(str(i)+'\n')

#-----------------------
#第二种
#-----------------------

#先转为numpy的（n,1）再用np.savetxt写入txt文件
pathdir = np.array(pathdir_train).reshape(82783,1)
np.savetxt(train_path_file, np.c_[pathdir], fmt='%s')
pathdir = np.array(pathdir_val).reshape(40504,1)
np.savetxt(val_path_file, np.c_[pathdir], fmt='%s')



