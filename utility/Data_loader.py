import os
import numpy as np

from PIL import Image
from tqdm import tqdm

class File_loader(object):
    
    """
    Reading image data files with interp or no.
    Interpolated images showed better result than not.
    """
    
    def read_files(self, read_path, interp = True):
        
        file_list = os.listdir(read_path)  # reading file list
        data_l = []
        label_l = []
        
        weight_list = [1, 1, 1]
        
        for file_name in tqdm(file_list):
            
            PATH = read_path + file_name

            # stacking
            if '.npy' in file_name:        
                image_a = np.load(PATH)      # 어레이 불러오고
                
                if interp == True:
                    
                    image_1 = np.array(Image.fromarray(image_a[0].astype('uint8')).resize((32,32), resample=Image.BILINEAR)).reshape(1, 32, 32)
                    image_2 = np.array(Image.fromarray(image_a[1].astype('uint8')).resize((32,32), resample=Image.BILINEAR)).reshape(1, 32, 32)
                    image_3 = np.array(Image.fromarray(image_a[2].astype('uint8')).resize((32,32), resample=Image.BILINEAR)).reshape(1, 32, 32)
                    
                    image = np.concatenate((image_1, image_2, image_3), axis=0)
                    
                else:
                    image_1 = image_a[0].reshape(1, 32, 24)
                    image_2 = image_a[1].reshape(1, 32, 24)
                    image_3 = image_a[2].reshape(1, 32, 24)
                    
                    image = np.concatenate((image_1, image_2, image_3), axis=0)
                    
                    
            # No stacking
            else:
                image_p = Image.open(PATH)   # PIL 이미지 불러오고
                
                if interp == True:
                    image_p = image_p.resize((32, 32), resample = Image.BILINEAR)
                    image = np.array(image_p).reshape(1, 32, 32)
                    
                else:
                    image = np.array(image_p).reshape(1, 32, 24)
            
            label = int(file_name[-5]) - 1
            data_l.append(image)
            label_l.append(label)
            
        data_a = np.asarray(data_l)
        label_a = np.asarray(label_l)

        return data_a, label_a