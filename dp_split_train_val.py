import shutil
import os
import numpy as np

class data_prep:
    def __init__(self):
        # Set the train_ratio
        self.train_ratio = 0.8

        # Set the directories of the source images, target sorted images, and target val images
        self.dataset_dir = './dataset'
        self.raw_img_dir = './dataset/deduped_images'
        self.sorted_img_dir = './dataset/sorted_images'
        self.sorted_img_train = './dataset/sorted_images_train'
        self.sorted_img_val = './dataset/sorted_images_val'

    def get_files_from_folder(self, path):

        files = sorted(os.listdir(path))
        return np.asarray(files)

    def get_files_from_folder_shuffled(self, path):
        np.random.seed(42) 

        files = sorted(os.listdir(path))
        files = np.asarray(files)
        np.random.shuffle(files)
        return files

    def main(self):
        # get dirs
        _, dirs, _ = next(os.walk(self.sorted_img_dir))

        # calculates how many train data per class
        data_counter_per_class = np.zeros((len(dirs)))
        for i in range(len(dirs)):
            path = os.path.join(self.sorted_img_dir, dirs[i])
            files = self.get_files_from_folder(path)
            data_counter_per_class[i] = len(files)
        train_counter = np.round(data_counter_per_class * (self.train_ratio))
        val_counter = np.round(data_counter_per_class * (1 - self.train_ratio))

        ## Created directory for train and val
        #creates dir for train
        if os.path.exists(self.sorted_img_train):
            shutil.rmtree(self.sorted_img_train)
        
        #creates dir for val
        if os.path.exists(self.sorted_img_val):
            shutil.rmtree(self.sorted_img_val)
        
        '''
        ## Transfer files for train dataset
        for i in range(len(dirs)):
            path_to_original = os.path.join(self.sorted_img_dir, dirs[i])
            path_to_save = os.path.join(self.sorted_img_train, dirs[i])

            #creates dir
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            files = self.get_files_from_folder(path_to_original)
            # moves data
            for j in range(int(train_counter[i])):
                dst = os.path.join(path_to_save, files[j])
                src = os.path.join(path_to_original, files[j])
                shutil.copy(src, dst)

        ## Transfer files for val dataset
        for i in range(len(dirs)):
            path_to_original = os.path.join(self.sorted_img_dir, dirs[i])
            path_to_save = os.path.join(self.sorted_img_val, dirs[i])

            #creates dir
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            files = self.get_files_from_folder(path_to_original)
            # moves data
            for j in range(int(train_counter[i]), int(train_counter[i]) + int(val_counter[i])):
                dst = os.path.join(path_to_save, files[j])
                src = os.path.join(path_to_original, files[j])
                shutil.copy(src, dst)
        '''

        ## Transfer files for train and val dataset
        for i in range(len(dirs)):
            path_to_original = os.path.join(self.sorted_img_dir, dirs[i])
            path_to_train = os.path.join(self.sorted_img_train, dirs[i])
            path_to_val = os.path.join(self.sorted_img_val, dirs[i])

            ## Creates dir
            # train dir
            if not os.path.exists(path_to_train):
                os.makedirs(path_to_train)
            # val dir
            if not os.path.exists(path_to_val):
                os.makedirs(path_to_val)

            files = self.get_files_from_folder_shuffled(path_to_original)
            # moves train data
            for j in range(int(train_counter[i])):
                dst = os.path.join(path_to_train, files[j])
                src = os.path.join(path_to_original, files[j])
                shutil.copy(src, dst)
            
            # moves val data
            for j in range(int(train_counter[i]), int(train_counter[i]) + int(val_counter[i])):
                dst = os.path.join(path_to_val, files[j])
                src = os.path.join(path_to_original, files[j])
                shutil.copy(src, dst)

if __name__ == "__main__":
  a = data_prep()
  a.main()