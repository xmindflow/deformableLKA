import numpy as np
from PIL import Image
data_x = np.load('/work/scratch/niggemeier/projects/transnorm/PH2/X_tr_256x256.npy')
label = np.load('/work/scratch/niggemeier/projects/transnorm/PH2/Y_tr_256x256.npy')

print(f"length of data: {np.shape(data_x)}")

# pre allocate memory
res_data = np.zeros([200, 256, 256, 3])
res_label = np.zeros([200, 256, 256])

# resize
for idx in range(200):
    print(idx+1)
    
    data_slice = np.transpose(data_x[idx,:,:,:]* 255, (1,2,0)) # 3, 256 ,256 --> 256,256,3
    print(data_slice.shape)
    cur_data = Image.fromarray(data_slice.astype('uint8'))
    resized_img = cur_data.resize((224,224), Image.BILINEAR)
    resized_data = np.double(resized_img)
    data_slice = np.double(data_slice)
    
    label_slice = np.squeeze(label[idx]*255)
    cur_label = Image.fromarray(label_slice.astype('uint8'))
    resized_label_img = cur_label.resize((224,224), Image.BILINEAR)
    resized_label = np.double(resized_label_img)
    label_slice = np.double(label_slice)
    #resized_img = resized_img.save('./zpil_resized_img.png')
    #resized_label_img = resized_label_img.save('./zpilresized_label_img.png')
    
    res_data[idx,:,:,:] = data_slice#resized_data
    res_label[idx,:,:]  = label_slice#resized_label
    
print('Reading PH2 finished')

Train_img      = res_data[0:80,:,:,:]
Validation_img = res_data[80:100,:,:,:]
Test_img       = res_data[100:200,:,:,:]

Train_mask      = res_label[0:80,:,:]
Validation_mask = res_label[80:100,:,:]
Test_mask       = res_label[100:200,:,:]


np.save('./PH2/test256/data_train', Train_img)
np.save('./PH2/test256/data_test' , Test_img)
np.save('./PH2/test256/data_val'  , Validation_img)

np.save('./PH2/test256/mask_train', Train_mask)
np.save('./PH2/test256/mask_test' , Test_mask)
np.save('./PH2/test256/mask_val'  , Validation_mask)