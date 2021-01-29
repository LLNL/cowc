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

# The pickle file with the list of cars
# ftp://gdo152.ucllnl.org/cowc-m/datasets/Objects_15cm_24px-exc_v5-marg-32.pickle
unique_list         = '/Users/mundhenk1/Downloads/temp/Objects_15cm_24px-exc_v5-marg-32.pickle'
# The location of the raw scenes
# ftp://gdo152.ucllnl.org/cowc-m/datasets/Organized_Raw_Files.tgz
raw_image_root      = '/Users/mundhenk1/Downloads/temp/Organized_Raw_Files'
# Where to save the new patches we create on our local drive
output_image_root   = '/Users/mundhenk1/Downloads/temp/64x64_15cm_24px-exc_v5-marg-32_expanded'


# Here we specify the size of each patch along with a margin of mean gray to wrap the image in
# We give four suggestions for sizes 

# Suggestion 1
#patch_size          = 256
#marg_size           = 32

# Suggestion 2
#patch_size          = 232
#marg_size           = 20

# Suggestion 3
#patch_size          = 120
#marg_size           = 4

# Suggestion 4
patch_size          = 64
marg_size           = 4

# If we set this to more than zero, we will create that many extra patches with color permutations
color_permutes      = 0

# For each patch, we will also create its rotation. Here we list all the rotations we would like to create
rotation_set        = [0,15,30,45,60,75,90,105,120,135,150,165,180]
# For testing we do the same, but use fewer rotations to keep the testing set more compact
test_rotations      = [0,15,30,45]
# We can create multiple scales in addition the just the standard one
# Scales must be >= 1.0
scale_set           = [1.0]
# This is the mean color used in the margin. 
mean_color          = [104, 117, 123]

# ******************************************************************************************************************* 
# ******************************************************************************************************************* 
# Dont edit after here
# ******************************************************************************************************************* 
# ******************************************************************************************************************* 


#========================================================================================================================

class CarProp:
    def __init__(self,phase,type,loc_1,loc_2):
        self.phase  = phase
        self.type   = type
        self.loc_1  = loc_1
        self.loc_2  = loc_2
 
#========================================================================================================================
       
def create_zoom_crop_image(in_image, patch_size, marg_size, visible_size, mean_color, zoom):
    
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
    rot         = cv2.getRotationMatrix2D((in_img.shape[1]//2, in_img.shape[0]//2), r_rotate, 1.0) 
    out_img     = cv2.warpAffine(in_img, rot, (in_img.shape[1], in_img.shape[0])) 
    return out_img.astype(np.uint8)

#========================================================================================================================

def rotate_hue(img):

    npermute = np.random.randint(0,5)
    
    if npermute == 0:
        nimg = img[:,:,[1,0,2]]
    elif npermute == 1:
        nimg = img[:,:,[2,1,0]]
    elif npermute == 2:
        nimg = img[:,:,[0,2,1]]
    elif npermute == 3:
        nimg = img[:,:,[1,2,0]]
    elif npermute == 4:
        nimg = img[:,:,[2,0,1]]

    return nimg

#========================================================================================================================
#========================================================================================================================

# patch required is the required image for rotation. We force it to be even.
# We use this to get our initial crop from the large raw scene. It's over sized so we can
# then crop out the rotated patch without running out of bounds. 
patch_required      = int( round( math.sqrt(patch_size*patch_size + patch_size*patch_size)/2.0 ) )*2
if patch_required%2 != 0:
    patch_required = patch_required + 1
    
visible_size        = patch_size - 2*marg_size

# load in the list of car locations and negatives for creating patches
print("Loading: " + unique_list)

in_file             = open(unique_list, 'rb')

item_list           = pickle.load(in_file)

# Create the output directory if we don't have one yet
if not os.path.isdir(output_image_root):
    os.mkdir(output_image_root)

# we will run through all sample locations which are sorted by the original dataset (e.g. CSUAV or Utah)
for file_dir in sorted(item_list):
    
    print("Processing Dir:\t" + file_dir)
    
    set_raw_root            = os.path.join(raw_image_root, file_dir)
    set_output_root         = os.path.join(output_image_root, file_dir)
    
    if not os.path.isdir(set_output_root):
        os.mkdir(set_output_root)
        
    set_output_root_test    = os.path.join(set_output_root, 'test')
    set_output_root_train   = os.path.join(set_output_root, 'train')
    
    if not os.path.isdir(set_output_root_test):
        os.mkdir(set_output_root_test)
        
    if not os.path.isdir(set_output_root_train):
        os.mkdir(set_output_root_train)
    
    # For each of the large raw scene images
    for file_root in sorted(item_list[file_dir]):        

        raw_file = os.path.join(set_raw_root, "{}.png".format(file_root))
        
        pstring =  "\tReading Raw File:\t" + raw_file + " ... "
        
        sys.stdout.write(pstring)
        sys.stdout.flush()
        
        # load the large raw scene image
        raw_image = cv2.imread(raw_file)
        
        print("Image Size: ")
        print(raw_image.shape)
        
        print("Done")
        
        print("Processing:")
        
        counter = 0
        
        # for each sample in this scene image
        for locs in item_list[file_dir][file_root]:
            
            # Get the location of this sample
            loc_1 = int(locs.loc_1)
            loc_2 = int(locs.loc_2)
            
            temp_name       = "{}.{}.{:05d}.{:05d}".format(locs.type, file_root, loc_1, loc_2)
            full_temp_name  = os.path.join(set_output_root, locs.phase, temp_name)
            
            # detemine the window location of this patch within the large raw scene
            x1 = int(loc_1-patch_required//2)
            x2 = int(loc_1+patch_required//2)
            y1 = int(loc_2-patch_required//2)
            y2 = int(loc_2+patch_required//2)
            
            # make sure we're in bounds, if not we can just add more gray area
            if x1 < 0 or y1 < 0 or x2 >= raw_image.shape[1] or y2 >= raw_image.shape[0]:
                
                # We are running outside the large raw scene image, so we need to get the visible
                # area and then make the part of the patch that lies outside the image gray
                temp_image           = np.empty((patch_required,patch_required,3),dtype=np.uint8)
                temp_image[:,:,0]    = mean_color[0]
                temp_image[:,:,1]    = mean_color[1]
                temp_image[:,:,2]    = mean_color[2]
                ty1 = 0 
                tx1 = 0
                ty2 = patch_required
                tx2 = patch_required
                ny1 = y1
                nx1 = x1
                ny2 = y2
                nx2 = x2

                if y1 < 0:
                    ty1 = -y1
                    ny1 = 0
                    
                if x1 < 0:
                    tx1 = -x1
                    nx1 = 0
                    
                if x2 >= raw_image.shape[1]:
                    tx2 = patch_required + (raw_image.shape[1] - x2) - 1
                    nx2 = raw_image.shape[1] - 1
                    
                if y2 >= raw_image.shape[0]:
                    ty2 = patch_required + (raw_image.shape[0] - y2) - 1
                    ny2 = raw_image.shape[0] - 1

                # Get the part of the image that is inside the scene
                temp_image[ty1:ty2,tx1:tx2,:]   = raw_image[ny1:ny2,nx1:nx2,:]
                cropped_raw_image               = temp_image
            else:
                # Patch is totally inside the image, we just crop out it. We leave some slack so we can
                # crop out a final rotated image later
                cropped_raw_image           = np.empty((patch_required,patch_required,3),dtype=np.uint8)
                cropped_raw_image[:,:,:]    = raw_image[y1:y2,x1:x2,:]
            
            r_set = []
            
            # if we are using a training image, we load a different set of rotation permutions than
            # if we are using testing images
            if locs.phase == "test":
                r_set = test_rotations
            else:
                r_set = rotation_set
            
            # for each rotation permutation
            for rot in r_set:
            
                # create the rotated patch image
                rot_img = permute_affine(cropped_raw_image, rot)
                
                for scale in scale_set:
                    
                    file_name       = "{}.{}.{:05d}.{:05d}.{:04.2f}-{:03d}.jpg".format(locs.type, file_root, loc_1, loc_2, scale, rot)
                    
                    full_file_name  = os.path.join(set_output_root, locs.phase, file_name)
                    
                    # Zoom in if requested and do a final crop. 
                    out_image       = create_zoom_crop_image(rot_img, patch_size, marg_size, visible_size, mean_color, scale)
                    
                    # Write the image patch out
                    cv2.imwrite(full_file_name, out_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    # only do color permutions on the training data
                    if locs.phase == "train":
                        
                        # if we want to create color permutations, do it here. 
                        for p in range(color_permutes):
                            
                            # We apply augmentation to the already cropped patch
                            nout            = rotate_hue(out_image)
                                            
                            file_name       = "{}.{}.{:05d}.{:05d}.{:04.2f}-{:03d}-HueRot-{}.jpg".format(locs.type, file_root, loc_1, loc_2, scale, rot, p)
                        
                            full_file_name  = os.path.join(set_output_root, locs.phase, file_name)
                            
                            # Write the image patch out
                            cv2.imwrite(full_file_name, nout, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            if counter > 0:
                if counter%100 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                if counter%5000 == 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            counter += 1
            
        print('x')
            
                    
            
            
        