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
# February 2016
#
# ================================================================================================
#
#    Copyright (C) 2016 Lawrence Livermore National Security
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
#
# Sorry about the ugly python. I just joined the pyrty recently. Enjoy. 
#
# 
# ================================================================================================

import caffe
import numpy as np
import string
import cv2
import os
import math
import time 

# ================================================================================================
class WinProp:
    def __init__(self):
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0

# ================================================================================================

# The directory of your images        
input_root_dir = "LOCATION OF MY IMAGES"

# The list of your input images           
input_image_file_list    = []

input_image_file_list.append(input_root_dir + '12TVK440540_CROP_01.png') # 628

input_image_file_list.append(input_root_dir + '12TVK440540_CROP_02.png') # 285
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_03.png') # 140
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_04.png') # 596
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_05.png') # 881
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_06.png') # 94
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_07.png') # 28
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_08.png') # 208       
input_image_file_list.append(input_root_dir + '12TVL180480_CROP_09.png') # 249
input_image_file_list.append(input_root_dir + '12TVL180480_CROP_10.png') # 215
input_image_file_list.append(input_root_dir + '12TVK460400_CROP_11.png') # 498

input_image_file_list.append(input_root_dir + '12TVL220060_CROP_1.png') # 51
input_image_file_list.append(input_root_dir + '12TVL220060_CROP_2.png') # 42
input_image_file_list.append(input_root_dir + '12TVL460100_CROP_1.png') # 22
input_image_file_list.append(input_root_dir + '12TVL460100_CROP_2.png') # 23
input_image_file_list.append(input_root_dir + '12TVK220780_CROP_1.png') # 20
input_image_file_list.append(input_root_dir + '12TVK220780_CROP_2.png') # 20
input_image_file_list.append(input_root_dir + '12TVL240360_CROP_1.png') # 28
input_image_file_list.append(input_root_dir + '12TVL240360_CROP_2.png') # 36
input_image_file_list.append(input_root_dir + '12TVL160120_CROP_1.png') # 10
input_image_file_list.append(input_root_dir + '12TVL160120_CROP_2.png') # 10

# The ground truth count for each image
ground_truth_list   = [628,285,140,596,881,94,28,208,249,215,498,51,42,22,23,20,20,28,36,10,10]
# Which images should be included in the final statistics
use_image_list      = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# ================================================================================================

# Give a list of strides to use at different offsets. 
stride_list = []
stride_list.append([0,0])
#stride_list.append([0,4])
#stride_list.append([4,0])
#stride_list.append([4,4])

# render/save some nifty output images showing how things are going?
render_images       = True
# How large (in pixels) should each stride be for a sliding window
window_stride       = 167  
# Padding to add at the border of the image in pixels
window_pad          = 128 #104 or 128
# The batch size to use in Caffe
batch_size          = 50  # between 16 and 64
# The GPU device to use in Caffe
gpu_device          = 3
# What is the first layer in the network the images will be sent to?
start_layer         = 'Layer1_7x7/Convolution_Stride_2'
# What is the softmax output layer?
softmax_layer       = 'loss3/Softmax_plain'
# The location of your prototxt network training file. 
prototxt_net_file   = 'LOCATION OF MY TRAINING FILE'
# The location of your trained caffe model
caffemodel_file     = 'LOCATION OF MY MODEL FILE'

window_scale        = 1.0 / float(len(stride_list))

# ================================================================================================

# Initialize your caffe network

caffe.set_device(gpu_device)
caffe.set_mode_gpu()

print 'Load network'

net = caffe.Net(prototxt_net_file, caffemodel_file, caffe.TEST)

print 'Create Transformer'

net.blobs['data'].reshape(batch_size,3,224,224)

# Set up the image transformer

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.float32([104.0, 117.0, 123.0])) # mean pixel

# Init a bunch of counters and things

mean_caffe_time = 0.0
image_count     = 0
MAE             = 0.0
RMSE            = 0.0
sum_gt          = 0.0
sum_count       = 0.0
N               = 0.0
max_error       = 0.0

print "START COUNTING ...."

# For each image stride over and count
for input_image_file in input_image_file_list:
    
    total_counter   = 0.0
    caffe_runs      = 0
    final_bins      = 0
    caffe_time      = 0.0
                     
   # Open up our image. This is lazy and does not check the image to see if it's valid.
    input_image             = cv2.imread(input_image_file)     
       
     # This is the list of different stride offsets. 
    for s in stride_list:
    
        # Set the current from the list offset
        xs = s[0]
        ys = s[1]
        
        # create a padded blank image 
        input_image_pad         = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.uint8)
        
        # copy the original image into the center but offset it by xs,ys
        input_image_pad[ys+window_pad:ys+window_pad+input_image.shape[0],\
                        xs+window_pad:xs+window_pad+input_image.shape[1],:] = input_image  
        
        # set up images to keep track of statistics for output after we are done. 
        count_image             = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.uint8)
        box_image               = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.uint8)
        prop_image              = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.uint8)
        
        im_cols                 = input_image_pad.shape[1]/window_stride
        im_rows                 = input_image_pad.shape[0]/window_stride
        
        py = input_image.shape[0]/window_stride
        px = input_image.shape[1]/window_stride
        
        crop_img                = np.zeros((224,224,3),dtype=np.uint8)
        crop_img[:,:,0]         = 104
        crop_img[:,:,1]         = 117 
        crop_img[:,:,2]         = 123
        
        start = time.time()
        
        frame_count = 0
        
        y = 0
        x = 0
        do_batch    = True

        # While we still have window patches to feed into caffe
        while do_batch == True:
            
            x_counter = 0;
            win_props = []
            
            # Get batch_size number of windows into Caffe blobs
            for bx in range(batch_size):
                
                # Stop when we run out of image
                if x > px:
                    x = 0
                    y += 1
                    if y > py:
                        do_batch = False
                        break
                    
                # Define the window  
                w = WinProp();    
                w.y1 = y*window_stride
                w.y2 = 255 + y*window_stride
                w.x1 = x*window_stride
                w.x2 = 255 + x*window_stride
                
                x_counter += 1 
                frame_count += 1
            
                # Process window and plut into Caffe
                cv_img                              = input_image_pad[w.y1:w.y2, w.x1:w.x2, :]
                crop_img[16:208,16:208,:]           = cv_img[32:224,32:224,:]
                net.blobs['data'].data[bx,:,:,:]    = transformer.preprocess('data', crop_img)
                
                win_props.append(w)
                
                x += 1
            
            # Process this batch of window patches
            cstart      = time.time()
            if start_layer != 'data':
                net.forward(start=start_layer)  # Provide our own data
            else:
                net.forward()
                        
            smax4       = net.blobs[softmax_layer].data
            cend        = time.time()
            
            caffe_time += cend - cstart
            caffe_runs += 1

            # For each patch in our batch, get the results. 
            for bx in range(x_counter): 
                 
                # get the max ... the lazy way                    
                max_bin     = -1
                max_val     = -1
                bin_count   = 0
                for sbin in smax4[bx]:
                    if sbin > max_val:
                        max_val = sbin
                        max_bin = bin_count     # my class as a bin construct, not same as bin_size
                    bin_count += 1
        
                if render_images:
                    w = win_props[bx]
                    w.x1 = w.x1 + 32
                    w.y1 = w.y1 + 32
                    w.x2 = w.x2 - 32
                    w.y2 = w.y2 - 32
            
                    #draw some images that shows the count
                    if bx%2 == 0:
                        cv2.rectangle(box_image, (w.x1,w.y1), (w.x2,w.y2), (0,255,0), 3)
                    else:
                        cv2.rectangle(box_image, (w.x1,w.y1), (w.x2,w.y2), (255,0,0), 3)
                        
                    cv2.putText(box_image, "{}".format(max_bin), (w.x1+32,w.y2-32), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255))
                    
                    if max_bin > 0:
                        cv2.rectangle(prop_image, (w.x1,w.y1), (w.x2,w.y2), (255,255,255), -1)
                
                total_counter += float(max_bin) * window_scale  
                
                final_bins    += 1
        
        if render_images:
            # blend the box image and the input padded image
            input_image_pad[:,:, 2] = box_image[:,:, 2]/2 + input_image_pad[:,:, 2]/2
            input_image_pad[:,:, 1] = box_image[:,:, 1]/2 + input_image_pad[:,:, 1]/2
            input_image_pad[:,:, 0] = box_image[:,:, 0]/2 + input_image_pad[:,:, 0]/2
            
            # write the images
            file_name = input_image_file + ".{}.{}.count.jpg".format(xs,ys)
            print "Saving Image: " + file_name
            cv2.imwrite(file_name,input_image_pad)
            
            file_name = input_image_file + ".{}.{}.prop.png".format(xs,ys)
            print "Saving Image: " + file_name
            cv2.imwrite(file_name,prop_image)

    end = time.time()
    
    mean_caffe_time += caffe_time 
    
    print "**************************"
    print "IMAGE: " + input_image_file
    print "Time caffe: {}".format(caffe_time) 
    print "Total windows processed: {}".format(frame_count) 
    print "Caffe batches: {}".format(caffe_runs)
    print ">>> CAR COUNT: {}".format(total_counter)
    print ">>> GROUND TRUTH: {}".format(ground_truth_list[image_count])
    
    if use_image_list[image_count]:
        error   = total_counter - ground_truth_list[image_count]
        perror  = abs(error)/ground_truth_list[image_count]
        MAE         += perror
        RMSE        += perror*perror
        sum_gt      += ground_truth_list[image_count]
        sum_count   += total_counter
        N           += 1.0
        max_error   = max(max_error,perror)

        print ">>> ERROR: {}".format(error)
        print ">>> PERCENT ERROR: {}".format(perror)
    else:
        print ">>> VALIDATION IMAGE"
    
    image_count += 1

mean_caffe_time /= len(input_image_file_list)
MAE             /= N
RMSE            = math.sqrt(RMSE/N)
total_error     = abs(sum_gt - sum_count) / sum_gt

print "**************************"
print "RESULTS"
print "**************************"
print "Mean time caffe: {}".format(mean_caffe_time) 
print "MAE: {}".format(MAE)
print "RMSE: {}".format(RMSE)
print "Max Error: {}".format(max_error)
print "Total Error: {}".format(total_error)



