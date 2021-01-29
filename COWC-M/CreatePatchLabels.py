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

import os
import shutil

# ================================================================================================        

# This is a directory where we have a set of uniquely labeled car images
# ftp://gdo152.ucllnl.org/cowc-m/datasets/Sorted_Cars_By_Type_15cm_24px-exc_v5-marg-32_expanded.tgz
type_dir            = '/Users/mundhenk1/Downloads/temp/Sorted_Cars_By_Type_15cm_24px-exc_v5-marg-32_expanded'

# These are raw images we will match to the labeled car images
# ftp://gdo152.ucllnl.org/cowc-m/datasets/64x64_15cm_24px-exc_v5-marg-32_expanded.tgz
# OR
# You can create your own with CreateDetectionPatches.py
data_dir            = '/Users/mundhenk1/Downloads/temp/64x64_15cm_24px-exc_v5-marg-32_expanded'

# Where should we 
train_list_file     = os.path.join(data_dir, 'train_list_TARANTO.txt')
test_list_file      = os.path.join(data_dir, 'test_list_TARANTO.txt')

# Ignore the unknown labeled cars? Set this to {4}
ignore_list         = {4}
# Otherwise, leave blank to use the unknown 
#ignore_list         = {}

# For old style 2, for expanded 3
endlen = 3

# What label should we give each item?
# The order here is {Not Car, Other, Pickup, Sedan, Unknown 
label_str_list  = ["0","1","2","3","4"]

# If zero, we give no extra samples, otherwise we do that many extra (times) for each sample
# The order here is {Not Car, Other, Pickup, Sedan, Unknown 
extra_samples   = [0,0,0,0,0]

# ================================================================================================ 
# ================================================================================================ 
# DONT'T EDIT BELOW HERE
# ================================================================================================ 
# ================================================================================================ 

def getClass(file_name, label_set, car_prefix, endlen):    
    
    file_part = file_name.split('.')
    
    if file_part[0] == car_prefix:
    
        unique_name = ''

        for n in range(len(file_part)-endlen):
            unique_name =  unique_name + file_part[n] + '_'
    
        return label_set[unique_name]
    
    else:
        return 0
                
# ================================================================================================ 
def getLabels(file_root, label_set, label_num):
    
    # file_root:
    # e.g. /g/g17/mundhetn/data/CarsOverheadWithContext/Sorted_Cars_By_Type/CSUAV/Pickup
    
    file_list     = os.listdir(file_root)     
    
    for file in sorted(file_list):
        
        # file:
        # e.g. car.Columbus_EO_Run01_s2_301_15_00_42.40561128-Oct-2007_11-00-47.194_Frame_74.02085.01928.000.png
        
        file_part = file.split('.')
         
        unique_name = ''
    
        for n in range(len(file_part)-2):
            
            # e.g. car.Columbus_EO_Run01_s2_301_15_00_42.40561128-Oct-2007_11-00-47.194_Frame_74.02085.01928
            unique_name =  unique_name + file_part[n] + '_'
        
        label_set[unique_name] = label_num
    
    return label_set  

# ================================================================================================
# Get the lable of each car by directory
# ================================================================================================
 
files_root  = os.listdir(type_dir)

print('Do Lists')

label_set = {}

for file_dir in sorted(files_root):
        
    if os.path.isdir(type_dir + '/' + file_dir):
        
        # e.g. /g/g17/mundhetn/data/CarsOverheadWithContext/Sorted_Cars_By_Type/CSUAV
        
        print("Doing Type Directory: {}".format(os.path.join(type_dir, file_dir)))
        
        other_loc       = os.path.join(type_dir, file_dir, 'Other')
        pickup_loc      = os.path.join(type_dir, file_dir, 'Pickup') 
        sedan_loc       = os.path.join(type_dir, file_dir, 'Sedan')
        unknown_loc     = os.path.join(type_dir, file_dir, 'Unknown')
        
        label_set       = getLabels(other_loc,    label_set, 1)
        label_set       = getLabels(pickup_loc,   label_set, 2)
        label_set       = getLabels(sedan_loc,    label_set, 3)
        label_set       = getLabels(unknown_loc,  label_set, 4)
        
# ================================================================================================
# Get each sample and match to label
# ================================================================================================

test_count      = [0,0,0,0,0]
train_count     = [0,0,0,0,0]
test_samples    = 0
train_samples   = 0
car_prefix      = 'car'

files_root  = os.listdir(data_dir)

print('Do Lists')

test_out_list       = open(test_list_file, 'w')
train_out_list      = open(train_list_file, 'w')

for file_dir in sorted(files_root):
        
    if os.path.isdir(os.path.join(data_dir, file_dir)):
        
        print("Doing Data Directory: {}".format(os.path.join(data_dir,file_dir)))
        
        test_loc    = os.path.join(data_dir, file_dir, 'test')
        test_files  = os.listdir(test_loc)
        
        for test_file in sorted(test_files):
            car_class   = getClass(test_file, label_set, car_prefix, endlen) 
            test_count[car_class] += 1 
            if car_class in ignore_list:
                continue
            line        = os.path.join(test_loc, test_file) + '\t' + "{}".format(label_str_list[car_class]) + "\n"
            test_out_list.write(line)
            test_samples += 1
           
        train_loc   = os.path.join(data_dir, file_dir, 'train')
        train_files = os.listdir(train_loc)
                    
        for train_file in sorted(train_files):
            car_class   = getClass(train_file, label_set, car_prefix, endlen)  
            train_count[car_class] += 1          
            if car_class in ignore_list:
                continue
            line        = os.path.join(train_loc, train_file) + '\t' + "{}".format(label_str_list[car_class]) + "\n"
            
            train_out_list.write(line)
            train_samples += 1
             
            for e in range(extra_samples[car_class]):            
                train_out_list.write(line)
                train_samples += 1
                train_count[car_class] += 1


print("Writing: {}".format(test_list_file))
test_out_list.close()
print("Writing: {}".format(train_list_file))
train_out_list.close()

print("Train Neg: "     + "{}".format(train_count[0]))
print("Train Other: "   + "{}".format(train_count[1]))
print("Train Pickup: "  + "{}".format(train_count[2]))
print("Train Sedan: "   + "{}".format(train_count[3]))
print("Train Unknown: " + "{}".format(train_count[4]))
print("Train SAMPLES: " + "{}".format(train_samples))

print("Test Neg: "     + "{}".format(test_count[0]))
print("Test Other: "   + "{}".format(test_count[1]))
print("Test Pickup: "  + "{}".format(test_count[2]))
print("Test Sedan: "   + "{}".format(test_count[3]))
print("Test Unknown: " + "{}".format(test_count[4]))
print("Test SAMPLES: " + "{}".format(test_samples))

           
            