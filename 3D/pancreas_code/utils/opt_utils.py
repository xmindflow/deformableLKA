# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np
import torch


label2order = {
    0:[0,1,2,3],1:[0,1,3,2],2:[0,2,1,3],3:[0,2,3,1],  4:[0,3,1,2],5:[0,3,2,1],
    6:[1,0,2,3],7:[1,0,3,2],8:[1,2,0,3],9:[1,2,3,0],10:[1,3,0,2],11:[1,3,2,0],
    12:[2,0,1,3],13:[2,0,3,1],14:[2,1,0,3],15:[2,1,3,0],16:[2,3,0,1],17:[2,3,1,0],
    18:[3,0,1,2],19:[3,0,2,1],20:[3,1,0,2],21:[3,1,2,0],22:[3,2,0,1],23:[3,2,1,0]
}

label_index_vec={
    0:np.array([[1,0],[0,0]],dtype=np.float32),
    1:np.array([[0,1],[0,0]],dtype=np.float32),
    2:np.array([[0,0],[1,0]],dtype=np.float32),
    3:np.array([[0,0],[0,1]],dtype=np.float32)
}

def partition(image,div_num):
    result = []
    w, h, d = image.shape
    x_len = w // 2
    y_len = h // 2
    # z_len = d // 2

    divide_y = []
    divide_z = []
    divide_x = torch.split(image, x_len, dim=0)
    for i in divide_x:
        divide_y += torch.split(i, y_len, dim=1)
    # Z轴暂时不划分
    # for ii in divide_y:
    #     divide_z += (torch.split(ii, z_len, dim=2))

    # 重新组合
    decomposition = divide_y

    # 打乱组合
    label = [0]
    label_ = np.arange(1,24)
    np.random.shuffle(label_)
    label = label+ list(label_)

    for label_temp in label:
        index_temp = label2order[label_temp]
        organization_tmp = []
        for i in range(0, div_num * div_num):
            organization_tmp.append(decomposition[index_temp[i]])

        result_temp1 = torch.cat((organization_tmp[0], organization_tmp[1]), dim=1)
        result_temp2 = torch.cat((organization_tmp[2], organization_tmp[3]), dim=1)
        result_temp = torch.cat((result_temp1, result_temp2), dim=0)
        result_temp = torch.unsqueeze(result_temp,dim=0)
        result_temp = torch.unsqueeze(result_temp,dim=0)
        result.append(result_temp)

    # make index VEC
    index_VEC = []
    for label_temp in label:
        index_temp = label2order[label_temp]
        index_vec_temp_1 = np.hstack((label_index_vec[index_temp[0]],label_index_vec[index_temp[1]]))
        index_vec_temp_2 = np.hstack((label_index_vec[index_temp[2]], label_index_vec[index_temp[3]]))
        index_vec_temp = np.vstack((index_vec_temp_1,index_vec_temp_2))
        index_vec_temp = torch.from_numpy(index_vec_temp)
        index_vec_temp = torch.unsqueeze(index_vec_temp,0)
        index_vec_temp = torch.unsqueeze(index_vec_temp,1)
        index_vec_temp = torch.unsqueeze(index_vec_temp,-1)
        index_VEC.append(index_vec_temp)


    return {'image': result, 'label': label, 'index_vec':index_VEC}

def cut_partition(image,div_num):
    #todo 扣掉右下角,Z轴中间部分 8/1 测试
    b,c,h,w,d = image.shape
    cut_size = (h//2, w//2, d//2)
    h_start = h-cut_size[0]
    w_start = 0
    d_start = cut_size[2]//2

    zero_padd = torch.ones(cut_size)
    ori_cut_partition = image[:,:,h_start:h_start+cut_size[0],w_start:w_start+cut_size[1],d_start:d_start+cut_size[2]]
    image[:,:,h_start:h_start+cut_size[0],w_start:w_start+cut_size[1],d_start:d_start+cut_size[2]] = zero_padd
    cut_bbox = [h_start,w_start,d_start,cut_size[0],cut_size[1],cut_size[2]]

    return image,ori_cut_partition,cut_bbox





def Decomposition_and_reorganization_MRI(image,div_num):
    # To apply random contrast and brightness (random intensity transformations) on the input image (Pre-training stages)

    # # placeholders for the network
    # x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
    # # brightness + contrast changes final image
    # rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.3,seed=1)
    # rd_cont = tf.image.random_contrast(rd_brit,lower=0.7,upper=1.3,seed=1)
    # rd_fin=tf.clip_by_value(rd_cont,0,1.5)
    b = image.shape[0]
    samples = []
    for i in range(b):
        samples.append(partition(image[i, 0, :, :, :], div_num))

    return samples


def Cutout_MRI(image,div_num):
    # To apply random contrast and brightness (random intensity transformations) on the input image (Pre-training stages)

    # # placeholders for the network
    # x_tmp = tf.placeholder(tf.float32, shape=[batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
    # # brightness + contrast changes final image
    # rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.3,seed=1)
    # rd_cont = tf.image.random_contrast(rd_brit,lower=0.7,upper=1.3,seed=1)
    # rd_fin=tf.clip_by_value(rd_cont,0,1.5)

    samples, ori_cut_partition,cut_bbox = cut_partition(image, div_num)

    return samples,ori_cut_partition,cut_bbox




def ToTensor(sample):
    """Convert ndarrays in sample to Tensors."""


    image = sample['image']
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
    if 'onehot_label' in sample:
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
    else:
        image = (image / 255.).astype(np.float32)  # 归一化
        image = (image - 0.109565) / 0.103618  # 标准化
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}