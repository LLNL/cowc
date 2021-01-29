# ================================================================================================
# 
# Cars Overhead With Context
#
# http://gdo-datasci.ucllnl.org/cowc/
#
# T. Nathan Mundhenk, Goran Konjevod, Wesam A. Sakla, Kofi Boakye 
#
# Lawrence Livermore National Laboratory
# Global Security Directorate
#
# February 2018
#
# mundhenk1@llnl.gov
#
# ================================================================================================
#
#    Copyright (C) 2018 Lawrence Livermore National Security
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ================================================================================================
#
#   This work performed under the auspices of the U.S. Department of Energy by Lawrence Livermore
#   National Laboratory under Contract DE-AC52-07NA27344.
#
#   LLNL-MI-702797
#
# ================================================================================================

import pickle
import math
import cv2
import numpy as np
import string
import os
import shutil
import sys
import copy

# ftp://gdo152.ucllnl.org/cowc-m/datasets/Objects_All.pickle
#unique_list         = '/Users/mundhenk1/Downloads/temp/Objects_All.pickle'
unique_list         = r"C:\Users\mundhenk1\Downloads\COWC\cowc-m\datasets\Objects_All.pickle" # Windows Example
# ftp://gdo152.ucllnl.org/cowc-m/datasets/Organized_Raw_Files.tgz
#raw_image_root      = '/Users/mundhenk1/Downloads/temp/Organized_Raw_Files'
raw_image_root      = r"C:\Users\mundhenk1\Downloads\COWC\cowc-m\datasets\Organized_Raw_Files" # Windows Example
# Somewhere on your local drive
#output_image_root   = '/Users/mundhenk1/Downloads/temp/DetectionPatches_256x256'
output_image_root   = r"C:\Users\mundhenk1\Downloads\COWC\cowc-m\outputs\DetectionPatches_256x256" # Windows Example

# How large should each example patch be
patch_size          = 256
# striding step for extract patches from the large orignal image
step_size           = 128
# Should we also extract negative examples (no you shouldn't)
cars_only           = True
# How many pixels in size is the typical car?
car_size            = 32

# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# Dont edit after here
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 

#========================================================================================================================

class CarProp:
    def __init__(self,phase,type,loc_1,loc_2,obj_class):
        r"""    
            This stores attributes for each car in the dataset.
        """
        self.phase      = phase
        self.type       = type
        self.loc_1      = loc_1
        self.loc_2      = loc_2
        self.obj_class  = obj_class

#========================================================================================================================
       
def create_zoom_crop_image(in_image, patch_size, marg_size, visible_size, mean_color, zoom):
    r"""    
        This will crop out an image and optionally add a margin or zoom in on it.
    """
    out_image           = np.empty((patch_size,patch_size,3),dtype=np.uint8)
    out_image[:,:,0]    = mean_color[0]
    out_image[:,:,1]    = mean_color[1]
    out_image[:,:,2]    = mean_color[2]
    
    
    if zoom != 1.0:
        in_image_scaled     = cv2.resize(in_image,(0,0),fx=float(zoom),fy=float(zoom))
    else:
        in_image_scaled     = in_image
        
    out_center          = int(patch_size//2)
    in_center           = int(in_image_scaled.shape[0]//2)
    
    x1_out              = int(out_center-visible_size//2)
    x2_out              = int(out_center+visible_size//2+1)
    y1_out              = int(out_center-visible_size//2)
    y2_out              = int(out_center+visible_size//2+1)
    
    x1_in               = int(in_center-visible_size//2)
    x2_in               = int(in_center+visible_size//2+1)
    y1_in               = int(in_center-visible_size//2)
    y2_in               = int(in_center+visible_size//2+1)
    
    out_image[y1_out:y2_out,x1_out:x2_out,:] = in_image[y1_in:y2_in,x1_in:x2_in,:]
    
    return out_image.astype(np.uint8)

#========================================================================================================================

def permute_affine(in_img, r_rotate):
    r"""    
        This will properly rotate an image patch.
    """
    rot         = cv2.getRotationMatrix2D((in_img.shape[1]//2, in_img.shape[0]//2), r_rotate, 1.0) 
    out_img     = cv2.warpAffine(in_img, rot, (in_img.shape[1], in_img.shape[0])) 

    return out_img.astype(np.uint8)

#========================================================================================================================
#========================================================================================================================

assert(patch_size%step_size==0)
part_steps = int(patch_size / step_size)

# patch required is the required image for rotation. We force it to be even
patch_required      = int( round( math.sqrt(patch_size*patch_size + patch_size*patch_size)/2.0 ) )*2
if patch_required%2 != 0:
    patch_required = patch_required + 1

# Get the list of all pre-annotated objects. 
print("Loading: " + unique_list)

in_file             = open(unique_list, 'rb')

item_list           = pickle.load(in_file)

if not os.path.isdir(output_image_root):
    os.mkdir(output_image_root)

# We will store a single CSV file with the counts of cars from all patches we generate. 
count_file          = os.path.join(output_image_root,"object_count.csv")
count_file_handle   = open(count_file,'w')
count_file_handle.write("Folder_Name,File_Name,Neg_Count,Other_Count,Pickup_Count,Sedan_Count,Unknown_Count\n")

# For each large sub directory set we have (e.g. Utah, Selwyn)
for file_dir in sorted(item_list):
    
    print("Processing Dir:\t" + file_dir)
    
    set_raw_root            = os.path.join(raw_image_root    , file_dir)
    set_output_root         = os.path.join(output_image_root , file_dir)
    
    if not os.path.isdir(set_raw_root):
        print(">>> WARNING \"{}\" NOT FOUND, Skipping since it may be \"held-out\". You probably don\'t want this.".format(set_raw_root))
        continue
    
    if not os.path.isdir(set_output_root):
        os.mkdir(set_output_root)
     
    # For each large raw file in this sub directory set...    
    for file_root in sorted(item_list[file_dir]):        

        raw_file = os.path.join(set_raw_root, "{}.png".format(file_root))
        
        pstring =  "\tReading Raw File:\t" + raw_file + " ... "
        
        sys.stdout.write(pstring)
        sys.stdout.flush()
        
        # Read the large raw overhead file
        raw_image = cv2.imread(raw_file)
        
        print("Image Size: ")
        print(raw_image.shape)
        
        print("Done")
        
        print("Processing:")
        
        counter = 0
        
        # Get how many patches/steps we will have using patch size and stride
        steps_x = int(int(raw_image.shape[1])//int(step_size))
        steps_y = int(int(raw_image.shape[0])//int(step_size))
        
        step_locs = []
        
        for y in range(steps_y + 1):
            ts = []
            for x in range(steps_x + 1):
                ts.append([])
                
            step_locs.append(ts)
        
        # stuff objects into location bins
        for locs in item_list[file_dir][file_root]:
            
            loc_1 = int(locs.loc_1)
            loc_2 = int(locs.loc_2)
            
            step_loc_1 = int(loc_1)//int(step_size)
            step_loc_2 = int(loc_2)//int(step_size)
            
            step_locs[step_loc_2][step_loc_1].append(locs)
            
        for y in range(steps_y):
            # Patch bounds along Y
            y1 = y * step_size
            y2 = y1 + patch_size 
            
            if y2 > raw_image.shape[0]:
                break
            
            for x in range(steps_x):
                # Patch bounds along X
                x1 = x * step_size
                x2 = x1 + patch_size             
                
                if x2 > raw_image.shape[1]:
                    break 
                
                # File names for things we will save
                im_name_base    = "{}.{}.{}.jpg".format(file_root,x,y)
                bb_name         = os.path.join(set_output_root,"{}.{}.{}.txt".format(file_root,x,y))         
                im_name         = os.path.join(set_output_root,im_name_base)
                ck_name         = os.path.join(set_output_root,"{}.{}.{}.check.jpg".format(file_root,x,y))
                
                img_patch = raw_image[y1:y2,x1:x2,:]
                
                # Get a list of all objects within this image patch
                obj_list = []
                
                for sy in range(part_steps):
                    for sx in range(part_steps):
                        for locs in step_locs[y + sy][x + sx]:                         
                            if locs.obj_class != 0:
                                obj_list.append(locs)
                
                count = [0,0,0,0,0]
                
                # Write out the raw unlabeled image patch
                cv2.imwrite(im_name, img_patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])   
                img_patch_cp = copy.deepcopy(img_patch)
                
                r"""
                    Is there at least on object in the patch? Otherwise don't bother trying
                    to draw the bounding box images with boxes. We also skip creating the
                    bounding box list file. 
                """
                if len(obj_list) > 0:                
                                 
                    bb_file = open(bb_name,'w')                
                    
                    # in case we want to do something else, we keep the obj_list list
                    for l in obj_list:
                        x_loc   = float(int(l.loc_1) - x1)/float(patch_size)
                        y_loc   = float(int(l.loc_2) - y1)/float(patch_size)
                        h       = float(car_size)/float(patch_size)
                        w       = float(car_size)/float(patch_size)
                        
                        # Write out the bounding boxes
                        if cars_only:
                            if l.obj_class != 0:        
                                bb_file.write("{} {} {} {} {}\n".format(l.obj_class,x_loc,y_loc,h,w))
                        else:
                            bb_file.write("{} {} {} {} {}\n".format(l.obj_class,x_loc,y_loc,h,w))
                
                        # Count this object
                        count[l.obj_class] += 1
                
                        # Get the color which goes with this class
                        if l.obj_class == 0:
                            # white
                            col = (255, 255, 255)
                        elif l.obj_class == 1:
                            # red
                            col = (0, 0, 255)
                        elif l.obj_class == 2:
                            # green
                            col = (0, 255, 0)
                        elif l.obj_class == 3:
                            # blue
                            col = (255, 0, 0)
                        elif l.obj_class == 4:
                            # purple
                            col = (150, 0, 200) 
                        
                        # Get the bounding box for drawing in OpenCV
                        x_1 = int(int(l.loc_1) - x1 + (car_size // 2))
                        y_1 = int(int(l.loc_2) - y1 + (car_size // 2))
                        x_2 = int(int(l.loc_1) - x1 - (car_size // 2))
                        y_2 = int(int(l.loc_2) - y1 - (car_size // 2))

                        coords = [x_1, y_1, x_2, y_2]
                        # Check bounds
                        for i in range(len(coords)):
                            coord = coords[i]
                            if coord < 0:
                                coords[i] = 0
                            if coord > patch_size:
                                coords[i] = patch_size

                        # Draw the box
                        img_patch_cp = cv2.rectangle(img_patch_cp, (coords[0], coords[1]), (coords[2], coords[3]), col, 1)
                
                # Write the image with the bounding boxes in it.         
                cv2.imwrite(ck_name, img_patch_cp) 
                
                r"""   
                    Write out a central file with the count of each vehicle type for each patch.
                    
                    Count File Format: Folder_Name,File_Name,Neg_Count,Other_Count,Pickup_Count,Sedan_Count,Unknown_Count
                """
                count_file_handle.write("{},{}".format(file_dir,im_name_base))
                for i in count:
                    count_file_handle.write(",{}".format(i)) 
                count_file_handle.write("\n")
                
                # Show a very simple progress counter
                if counter > 0:
                    if counter%100 == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                    if counter%5000 == 0:
                        sys.stdout.write('\n')
                        sys.stdout.flush()
                counter += 1
            
        print('x')
            
                    
            
            
        
