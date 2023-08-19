import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os

output_size =[112, 112, 80]

def covert_h5():
    listt = glob('G:\Data\\2018_Atrial_Segmentation_Challenge_COMPLETE_DATASET\Training Set\\*\lgemri.nrrd')
    for item in tqdm(listt):
        image, img_header = nrrd.read(item)
        label, gt_header = nrrd.read(item.replace('lgemri.nrrd', 'laendo.nrrd'))
        label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        #image = (image-np.min(image))/(np.max(image)-np.min(image))
        #image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.uint8)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        save_dir = item.replace('lgemri.nrrd', 'wyc_mri_norm0_255.h5')
        save_dir = save_dir.replace('Training Set','wyc_h5_data_unint8')
        pre_dir = os.path.dirname(save_dir)
        if not os.path.exists(pre_dir):
          os.makedirs(pre_dir)
        f = h5py.File(save_dir, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()