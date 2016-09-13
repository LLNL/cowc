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
import cv2
import math
from copy import deepcopy

# ================================================================================================

class WinProp:
    def __init__(self):
        self.x          = -1
        self.y          = -1
        self.val        = 0
        self.have_max   = False
        
# ================================================================================================

def getNextWindow(temp_p_map, threshold):
    
    p = WinProp()
    
    loc     = np.argmax(temp_p_map)
    p.y     = loc / temp_p_map.shape[1]
    p.x     = loc % temp_p_map.shape[1]
    p.val   = temp_p_map[p.y,p.x]
    
    if p.val > threshold:
        p.have_max = True
    else:
        p.have_max = False
    
    return p

# ================================================================================================
        
def getWindows(win_excl_size, temp_p_map, threshold):
    
    winPropSet = []
    
    have_more = True
    
    while have_more:
        p = getNextWindow(temp_p_map, threshold)
        
        if p.have_max == False:
            have_more = False
            break
        
        winPropSet.append(p)
        
        minx = max(p.x-win_excl_size/2, 0)
        miny = max(p.y-win_excl_size/2, 0)
        
        maxx = min(p.x+win_excl_size/2,temp_p_map.shape[1])
        maxy = min(p.y+win_excl_size/2,temp_p_map.shape[0])
        
        temp_p_map[miny:maxy,minx:maxx] = 0

    return winPropSet

# ================================================================================================
        
def drawWindows(img, winPropSet, win_size, rescale=1):   
    
    for p in winPropSet:
        if  (rescale*p.x)-win_size/2 >= 0 and (rescale*p.y)-win_size/2 >= 0 and \
            (rescale*p.y)+win_size/2 < img.shape[0] and (rescale*p.x)+win_size/2 < img.shape[1]:
            cv2.rectangle(img,\
                          ((rescale*p.x)-win_size/2,(rescale*p.y)-win_size/2),\
                          ((rescale*p.x)+win_size/2,(rescale*p.y)+win_size/2),\
                          (255,255,0))
            cv2.rectangle(img,\
                          ((rescale*p.x)-win_size/2-2,(rescale*p.y)-win_size/2-2),\
                          ((rescale*p.x)+win_size/2+2,(rescale*p.y)+win_size/2+2),\
                          (0,255,255))
        else:
            cv2.rectangle(img,\
                          ((rescale*p.x)-win_size/2,(rescale*p.y)-win_size/2),\
                          ((rescale*p.x)+win_size/2,(rescale*p.y)+win_size/2),\
                          (0,0,255))       
        
    return img


# ================================================================================================

# The Sampling stride over the image
window_stride       = 8
# how many patches can we fit in one batch
batch_size          = 64
# which GPU device to use
gpu_device          = 0
# skew placed on probability 
log_power           = 16
# numeric threshold for a detection (0 to 255)
threshold           = 196 
# The starting layer in your network
start_layer         = 'Layer1_7x7/Convolution_Stride_2'
# The final softmax output layer
softmax_layer       = 'loss3/Softmax_plain'
# The network prototxt file for Caffe
prototxt_net_file   = 'LOCATION OF MY TRAINING PROTOTXT'
# The trained caffe model
caffemodel_file     = 'LOCATION OF MY CAFFE MODEL'
# name to append to result files
method_label        = 'train_val_COWC_v3'


# Mean image values
mean_image_vals     = [104.0, 117.0, 123.0]
# detection window size
detect_win_size     = 48
# window exclusion size
win_excl_size       = 64/window_stride

# don't mess with these
window_size         = 255
window_pad          = window_size/2
patch_size          = 224

# ================================================================================================

# The directory of your images        
input_root_dir = "LOCATION OF MY IMAGES"

# The list of your input images           
input_image_file_list    = []

input_image_file_list.append(input_root_dir + '12TVK440540_CROP_01.png') 

input_image_file_list.append(input_root_dir + '12TVK440540_CROP_02.png')
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_03.png')
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_04.png') 
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_05.png') 
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_06.png') 
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_07.png') 
input_image_file_list.append(input_root_dir + '12TVK440540_CROP_08.png')       
input_image_file_list.append(input_root_dir + '12TVL180480_CROP_09.png') 
input_image_file_list.append(input_root_dir + '12TVL180480_CROP_10.png')
input_image_file_list.append(input_root_dir + '12TVK460400_CROP_11.png') 

input_image_file_list.append(input_root_dir + '12TVL220060_CROP_1.png') 
input_image_file_list.append(input_root_dir + '12TVL220060_CROP_2.png') 
input_image_file_list.append(input_root_dir + '12TVL460100_CROP_1.png') 
input_image_file_list.append(input_root_dir + '12TVL460100_CROP_2.png') 
input_image_file_list.append(input_root_dir + '12TVK220780_CROP_1.png') 
input_image_file_list.append(input_root_dir + '12TVK220780_CROP_2.png') 
input_image_file_list.append(input_root_dir + '12TVL240360_CROP_1.png')
input_image_file_list.append(input_root_dir + '12TVL240360_CROP_2.png') 
input_image_file_list.append(input_root_dir + '12TVL160120_CROP_1.png') 
input_image_file_list.append(input_root_dir + '12TVL160120_CROP_2.png') 

# ================================================================================================

caffe.set_device(gpu_device)
caffe.set_mode_gpu()

print 'Load network'

net = caffe.Net(prototxt_net_file, caffemodel_file, caffe.TEST)
 
print 'Create Transformer'

net.blobs['data'].reshape(batch_size,3,patch_size,patch_size)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.float32(mean_image_vals)) # mean pixel

batch_loc_x = []
batch_loc_y = []

for x in range(batch_size):
    batch_loc_x.append(0)
    batch_loc_y.append(0)

for input_image_file in input_image_file_list:
    
    print "Running: " + input_image_file
        
    results_prefix          = input_image_file + '.' + method_label + ".Stride-{}.".format(window_stride)
    input_image             = cv2.imread(input_image_file)
    input_image_pad         = np.zeros((input_image.shape[0]+window_pad*2, input_image.shape[1]+window_pad*2, 3), dtype=np.uint8)
    
    input_image_pad[window_pad:window_pad+input_image.shape[0],window_pad:window_pad+input_image.shape[1],:] = input_image  
    
    im_cols         = input_image_pad.shape[1]/window_stride
    im_rows         = input_image_pad.shape[0]/window_stride
    
    p_img           = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,1),dtype=float)
    lp_img          = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,1),dtype=float)
    r_img           = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,3),dtype=np.uint8)
    pr_img          = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,3),dtype=np.uint8)
    thresh_img      = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,3),dtype=np.uint8)
    pdisp_img       = np.zeros((input_image.shape[0]/window_stride,input_image.shape[1]/window_stride,3),dtype=np.uint8)
     
    crop_img        = np.zeros((patch_size ,patch_size ,3),dtype=np.uint8)
    crop_img[:,:,0] = mean_image_vals[0]
    crop_img[:,:,1] = mean_image_vals[1]
    crop_img[:,:,2] = mean_image_vals[2]
    
    x_range_size    = int(math.ceil(float(p_img.shape[1])/float(batch_size)))
    
    pix             = np.zeros((batch_size,3,1),dtype=np.uint8)
    
    image_counter   = 0
    
    for y in range(p_img.shape[0]):
        
        print 'Row ' + "{}".format(y*window_stride)
        
        y1 = y*window_stride
        y2 = window_size + y*window_stride
                        
        for sx in range(x_range_size):
    
            bx          = 0
            
            while True:
                x = (bx + batch_size*sx)
                if x > p_img.shape[1]:
                    break
                if bx == batch_size:
                    break
                
                use_location = False
                
                x1 = x*window_stride
                x2 = window_size + x*window_stride
                
                cv_img                              = input_image_pad[y1:y2, x1:x2, :]
                crop_img[16:208,16:208,:]           = cv_img[32:224,32:224,:]
                net.blobs['data'].data[bx,:,:,:]    = transformer.preprocess('data', crop_img)
                batch_loc_x[bx]                     = x
                batch_loc_y[bx]                     = y
                pix[bx,:,0]                         = cv_img[window_size/2,window_size/2,:]
                     
                bx          += 1
            
            if start_layer != 'data':
                net.forward(start=start_layer)  # Provide our own data
            else:
                net.forward()
    
            smax4       = net.blobs[softmax_layer].data
        
            for rx in range(bx):
                yy = batch_loc_y[rx]
                x  = batch_loc_x[rx]

                p1          = np.math.pow(smax4[rx][0],10.0)
                p2          = np.math.pow(smax4[rx][1],10.0)
                p           = (p2 - p1 + 1.0) / 2.0                 
                lp          = pow(p2 - p1 + 1.0,log_power)/pow(2,log_power)
    
                p_img[yy,x,0]       = p*255.0
                lp_img[yy,x,0]      = lp*255.0
                r_img[yy,x,:]       = pix[rx,:,0]  
                pdisp_img[yy,x,1]   = int(p*255.0)
            
                if p > 0.5:
                    pr_img[yy,x,2]          = np.round(p*255.0)
                    pr_img[yy,x,0]          = 0
                    thresh_img[yy,x,2]      = 255
                    if p < 0.75:
                        thresh_img[yy,x,2]      = 255
                        thresh_img[yy,x,1]      = 255 
                    else:
                        thresh_img[yy,x,2]      = 255
                elif p < 0.5:
                    if p > 0.25:
                        thresh_img[yy,x,1]      = 255
                    pr_img[yy,x,0]          = np.round(p*255.0)
                    pr_img[yy,x,2]          = 0

            
                pr_img[yy,x,1] = pix[rx,1,0]
                
            cv2.imshow("pr img",pr_img)
            cv2.imshow("Raw Prob",pdisp_img)
            cv2.imshow("P > 0.75;0.5;0.25",thresh_img)
            cv2.waitKey(1)        
    
    # ================================================================================================
   
    results_prefix          = input_image_file + '.' + method_label + ".Stride-{}.".format(window_stride)
    out_root                = results_prefix + "detect."
    pbyte_img               = p_img.astype(np.uint8)
    
    print "Getting Windows"
    
    retval,pthresh_img      = cv2.threshold(pbyte_img,threshold,255,cv2.THRESH_TOZERO)  
    winPropSet              = getWindows(win_excl_size, pthresh_img, 1)
    
    print "Drawing Windows"
    
    iiw_img                 = deepcopy(input_image)
    iiw_img                 = drawWindows(iiw_img, winPropSet, detect_win_size, iiw_img.shape[0]/r_img.shape[0]) 
    iiw_img_sm              = cv2.resize(iiw_img,(0,0),fx=0.25,fy=0.25)
    
    cv2.imshow("detections",iiw_img_sm)
    cv2.waitKey(1) 
    
    p_name      = results_prefix + 'p_img.jpg'
    lp_name     = results_prefix + 'lp_img.jpg'
    r_name      = results_prefix + 'r_img.jpg'
    pr_name     = results_prefix + 'pr_img.jpg'
    iiw_name    = out_root + 'iiw_img.jpg'
    
    print 'Saving ' + iiw_name
    
    cv2.imwrite(iiw_name,iiw_img)
           
    print 'Saving ' + p_name
    
    cv2.imwrite(p_name,p_img)
    
    print 'Saving ' + lp_name
    
    cv2.imwrite(lp_name,lp_img)
    
    print 'Saving ' + r_name
    
    cv2.imwrite(r_name,r_img)
    
    print 'Saving ' + pr_name
    
    cv2.imwrite(pr_name,pr_img)

