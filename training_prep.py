"""

Class to help augment images for training ML algorithms.
 See bottom of script for before and after directory tree structure.

    Author : Jonathan Sanabria

"""

from collections import defaultdict
import os
#https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=9NRlYXKQy3Kx
import PIL
import numpy as np
import torch
import torchvision
import argparse 

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  
  

class TrainingPrep():
  """
  Make a IMAGE_NAME.txt file for YOLOv5 training, and augment images using pytorch .   
  """
  PATH_IDX = 0
  DIR_IDX = 1
  FIL_IDX = 2

  def __init__(self , path_to_data = None, label_format_type = "YOLOv5"):
    self.path_to_data = path_to_data # the root where all the data directories will be made
    self.image_dict = defaultdict(list)
    self.label_type = label_format_type
  def last_dir_in_path(self,_path ):
    """Parse given path for last directory

    Args:
      _path: path as str.

    Returns:
      Last directory in dirpath.
    """
    return _path[self.PATH_IDX].split("/")[-1] 
    
  def dir_walk(self ):
    """Walk through directory and fill image_dict as { class_dir : im_path } . 
    """  
    empty = lambda l : len(l) == 0

    for _path in os.walk( self.path_to_data ):
      dir_list = _path[self.DIR_IDX]
      img_list = _path[self.FIL_IDX]
      if( empty(dir_list) and
          not empty(img_list) ):
        class_dir = self.last_dir_in_path(_path)
        list_of_files = _path[self.FIL_IDX]
        self.image_dict[class_dir] = list_of_files
            
  def prepare_data(self, generate_txt = True ,
                         augmenting_images = False ,
                         total_img_count = 10 ):
    """Generate a list of directories which only have files,  
        and Fill image_dict as { class_dir : im_path } . 

    Args:
      generate_txt: boolean for *.txt only use case.
      
      augmenting_images: boolean for augmentation decision.
      
      total_img_count: Number of images pytorch "Subset" will produce for all the classes

    Returns:
      List of nested directories which only have files.
    """
    if( augmenting_images ):
      self.dir_walk( )
      self.write_augmented_image( total_img_count )
    if( generate_txt ):
      self.dir_walk( )
      self.txt_gen( )

    
  def txt_gen(self):
    """generate *.txt label for training. From all images in "images" directory.
       ----images must be in the following format :
               {class_}_{class_li[class_]}_{counter}.jpg
    """  
    class_num  = lambda image_name : image_name.split("_")[0]
    class_name = lambda image_name : image_name.split("_")[1]
    image_num  = lambda image_name : image_name.split("_")[2]
    DIR_NAME = os.path.join( self.path_to_data , "labels")
    os.makedirs(DIR_NAME, exist_ok=True)
    counter = 0 
    
    for image_name in self.image_dict["images"]:
      os.makedirs(DIR_NAME, exist_ok=True)
      new_fname = image_name.strip("jpeg") + "txt"
      path = os.path.join(self.path_to_data,DIR_NAME,new_fname)
      
      if( self.label_type != "YOLOv5"):
        pass # TODO other label formats
      else:# default YOLOv5
        label = self.label_yolo_format( _class = class_num( image_name ) )
        
      with open(path,"w") as f:
        f.write(label)
        counter += 1
    print( f"OK : {counter} labels made")   

  def label_yolo_format( self , _class = 0 ,
                                _mx = 0.5 , _my = 0.5 , width = 1.0 , height = 1.0): 
    return f"{_class} {_mx} {_my} {width} {height}"
    
    
  def write_augmented_image( self , total_img_count ):
    """Steps within method : First get augmented images, Then save the images.

    Args:
      total_img_count: Number of images pytorch "Subset" will produce for all the classes
    """
    DIR_NAME = os.path.join( self.path_to_data , "images")
    class_li = list(self.image_dict.keys())
        
    dataset = self.get_augmented_images( self.path_to_data , total_img_count )
    
    # after get_augmented_images to prevent extra class count from images dir.
    os.makedirs(DIR_NAME, exist_ok=True) 
    counter = 0
    print( class_li)
    for image , class_ in dataset:
      path = os.path.join( self.path_to_data , DIR_NAME,
                             f"{class_}_{class_li[class_]}_{counter}.jpeg" )
      image.save( path )
      counter += 1

    print( f"OK : {counter} images made")   
     
  def get_augmented_images(self , directory , total_img_count ):
    """Generate augmented images for training. 

    Args:
      total_img_count: Number of images pytorch "Subset" will produce for all the classes
      
    Returns:
      Augmented images, dataset of size total_img_count 
    """  
    CJ = (0.8,1.2)
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.Pad(400, padding_mode="edge"),
      torchvision.transforms.RandomRotation(30, resample=PIL.Image.BICUBIC),
      torchvision.transforms.ColorJitter(brightness=CJ,contrast=CJ,saturation=CJ,hue=(-.1,.1)),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.RandomAffine(30, scale=(.80,1.0), resample=PIL.Image.BICUBIC),
      torchvision.transforms.CenterCrop(800)
      ])
    dataset = torchvision.datasets.ImageFolder( self.path_to_data , transform=transforms)
    dataset_subset = torch.utils.data.Subset(dataset,
                               np.random.choice(len(dataset), total_img_count, replace=True))
    #show_dataset(dataset)
    return dataset_subset

    
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("path_to_data", type=str )
  args = parser.parse_args()
  return args
    
if __name__ == '__main__':
  args = parse_args()
  TP = TrainingPrep( path_to_data = args.path_to_data)
  TP.prepare_data(  generate_txt = True , augmenting_images = True , total_img_count= 100)
        
     
