import shutil
import os
import pandas as pd

class sort_img:
    def __init__(self):
        # Set the directories of the source images, target sorted images, and target test images
        self.dataset_dir = './dataset'
        self.raw_img_dir = './dataset/images'
        self.sorted_img_dir = './dataset/sorted_images'
        self.test_img_dir = './dataset/sorted_test'
    
    def main(self):

        ## TRAINING IMAGES
        # Get the location information to be used for setting the sorted images
        train_info = pd.read_csv(self.dataset_dir+ '/train.csv', dtype = 'string')
        train_info['filename'] = train_info['Id'].astype(str) + '.jpg'

        sorted_img_loc = dict(zip(train_info['filename'], train_info['label']))

        # Move all the training images
        for file, label in sorted_img_loc.items():
            
            src = self.raw_img_dir+ '/' + file
            dst = self.sorted_img_dir  + '/' + label + '/' + file
            
            if os.path.exists(self.sorted_img_dir   + '/' + label):
                shutil.copy(src, dst)
            else:
                os.makedirs(self.sorted_img_dir + '/' + label)
                shutil.copy(src, dst)

        ## TEST IMAGES
        # Get the location information to be used for setting the test images
        test_info = pd.read_csv(self.dataset_dir+ '/test.csv', dtype = 'string')
        test_list = test_info['Id'].tolist()
        test_list = [s + '.jpg' for s in test_list]

        # Move all the test images
        for file in test_list:
            src = self.raw_img_dir+ '/' + file
            dst = self.test_img_dir + '/' + file
            
            if os.path.exists(self.test_img_dir):
                shutil.copy(src, dst)
            else:
                os.makedirs(self.test_img_dir)
                shutil.copy(src, dst)

if __name__ == "__main__":
  a = sort_img()
  a.main()