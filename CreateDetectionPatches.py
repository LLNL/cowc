import pickle
import math
import cv2
import numpy as np
import string
import os
import shutil
import sys

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
        
    out_center          = patch_size/2
    in_center           = in_image_scaled.shape[0]/2
    
    x1_out              = out_center-visible_size/2
    x2_out              = out_center+visible_size/2+1
    y1_out              = out_center-visible_size/2
    y2_out              = out_center+visible_size/2+1
    
    x1_in               = in_center-visible_size/2
    x2_in               = in_center+visible_size/2+1
    y1_in               = in_center-visible_size/2
    y2_in               = in_center+visible_size/2+1
    
    out_image[y1_out:y2_out,x1_out:x2_out,:] = in_image[y1_in:y2_in,x1_in:x2_in,:]
    
    return out_image.astype(np.uint8)

#========================================================================================================================

def permute_affine(in_img, r_rotate):
    #print in_img.shape     
    rot         = cv2.getRotationMatrix2D((in_img.shape[1]/2, in_img.shape[0]/2), r_rotate, 1.0) 
    #print rot
    #print ">>> WARPING"
    out_img     = cv2.warpAffine(in_img, rot, (in_img.shape[1], in_img.shape[0])) 
    #print ">>>> DONE"
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

unique_list         = '/data/shared/datasets/CarsOverheadWithContext/UniqueSet_256x256_15cm_24px-exc_v5-marg-32.pickle'
raw_image_root      = '/data/shared/datasets/CarsOverheadWithContext/Organized_Raw_Files'
#output_image_root   = '/data/shared/datasets/CarsOverheadWithContext/64x64_15cm_24px-exc_v5-marg-32_expanded'
output_image_root   = '/data/shared/datasets/CarsOverheadWithContext/64x64_15cm_24px-exc_v5-marg-32_expanded_hue-rot'
#patch_size          = 256
#marg_size           = 32
#patch_size          = 232
#marg_size           = 20
#patch_size          = 120
#marg_size           = 4
patch_size          = 64
marg_size           = 4
color_permutes      = 1

rotation_set        = [0,15,30,45,60,75,90,105,120,135,150,165,180]
test_rotations      = [0,15,30,45]
# Scales must be >= 1.0
scale_set           = [1.0]
mean_color          = [104, 117, 123]


# patch required is the required image for rotation. We force it to be even
patch_required      = int( round( math.sqrt(patch_size*patch_size + patch_size*patch_size)/2.0 ) )*2
if patch_required%2 != 0:
    patch_required = patch_required + 1
    
visible_size        = patch_size - 2*marg_size

print "Loading: " + unique_list

in_file             = open(unique_list)

item_list           = pickle.load(in_file)

if not os.path.isdir(output_image_root):
    os.mkdir(output_image_root)

for file_dir in sorted(item_list):
    
    print "Processing Dir:\t" + file_dir
    
    set_raw_root            = raw_image_root    + '/' + file_dir
    set_output_root         = output_image_root + '/' + file_dir
    
    if not os.path.isdir(set_output_root):
        os.mkdir(set_output_root)
        
    set_output_root_test    = set_output_root + '/test'
    set_output_root_train   = set_output_root + '/train'
    
    if not os.path.isdir(set_output_root_test):
        os.mkdir(set_output_root_test)
        
    if not os.path.isdir(set_output_root_train):
        os.mkdir(set_output_root_train)
    

    for file_root in sorted(item_list[file_dir]):        

        raw_file = set_raw_root + '/' + file_root + '.png'
        
        pstring =  "\tReading Raw File:\t" + raw_file + " ... "
        
        sys.stdout.write(pstring)
        sys.stdout.flush()
        
        raw_image = cv2.imread(raw_file)
        
        print "Image Size: "
        print raw_image.shape
        
        print "Done"
        
        print "Processing:"
        
        counter = 0
        
        for locs in sorted(item_list[file_dir][file_root]):
            
            loc_1 = int(locs.loc_1)
            loc_2 = int(locs.loc_2)
            
            temp_name       = "{}.{}.{:05d}.{:05d}".format(locs.type, file_root, loc_1, loc_2)
            full_temp_name  = set_output_root + '/' + locs.phase + '/' + temp_name
            #print "Processing: " + full_temp_name
            
            x1 = loc_1-patch_required/2
            x2 = loc_1+patch_required/2
            y1 = loc_2-patch_required/2
            y2 = loc_2+patch_required/2
            
            if x1 < 0 or y1 < 0 or x2 >= raw_image.shape[1] or y2 >= raw_image.shape[0]:
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
                '''    
                print "{}".format(patch_required)
                print "{},{} {},{}".format(x1,x2,y1,y2)
                    
                print "FROM {},{} {},{}".format(nx1,nx2,ny1,ny2)
                print "TO {},{} {},{}".format(tx1,tx2,ty1,ty2)
                '''
                temp_image[ty1:ty2,tx1:tx2,:]   = raw_image[ny1:ny2,nx1:nx2,:]
                cropped_raw_image               = temp_image
            else:
                cropped_raw_image           = np.empty((patch_required,patch_required,3),dtype=np.uint8)
                cropped_raw_image[:,:,:]    = raw_image[y1:y2,x1:x2,:]
            
            #print "Running Permutations"
            
            r_set = []
            
            if locs.phase == "test":
                r_set = test_rotations
                #continue
            else:
                #continue
                r_set = rotation_set
            
            for rot in r_set:
            
                rot_img = permute_affine(cropped_raw_image, rot)
                
                #print "\tRunning Scales"
                
                for scale in scale_set:
                    
                    file_name       = "{}.{}.{:05d}.{:05d}.{:04.2f}-{:03d}.jpg".format(locs.type, file_root, loc_1, loc_2, scale, rot)
                    
                    full_file_name  = set_output_root + '/' + locs.phase + '/' + file_name
                    
                    #print "\tDoing: " + file_name
                    
                    out_image       = create_zoom_crop_image(rot_img, patch_size, marg_size, visible_size, mean_color, scale)
                    
                    cv2.imwrite(full_file_name, out_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    if locs.phase == "train":
                        
                        for p in range(color_permutes):
                            
                            nout            = rotate_hue(out_image)
                                            
                            file_name       = "{}.{}.{:05d}.{:05d}.{:04.2f}-{:03d}-HueRot-{}.jpg".format(locs.type, file_root, loc_1, loc_2, scale, rot, p)
                        
                            full_file_name  = set_output_root + '/' + locs.phase + '/' + file_name
                            
                            cv2.imwrite(full_file_name, nout, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        
                    #print "\tDONE"
            
            if counter > 0:
                if counter%100 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                if counter%5000 == 0:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            counter += 1
            
        print 'x'
            
                    
            
            
        