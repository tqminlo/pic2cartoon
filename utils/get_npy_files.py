import os
import numpy as np
from tensorflow.python.keras.preprocessing import image
import tqdm

folder_imgs = '../dataset'
folder_npy = '../data_npy'
os.makedirs(folder_npy, exist_ok=True)

for member in os.listdir(folder_imgs):
    folder_imgs_member = os.path.join(folder_imgs, member)
    path_npy_member = os.path.join(folder_npy, member + '.npy')
    if os.path.exists(path_npy_member)==True:
        continue

    list_tensor = []
    for img in tqdm.tqdm(os.listdir(folder_imgs_member)):
        path_img = os.path.join(folder_imgs_member, img)
        x = image.load_img(path_img, target_size=(256,256))
        x = image.img_to_array(x)
        list_tensor.append(x)

    np.save(path_npy_member, list_tensor)



