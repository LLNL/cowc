import string
import os
import shutil

def getClass(file_name, label_set, car_prefix, endlen):    
    
    file_part = string.split(file_name, '.')
    
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
        
        file_part = string.split(file, '.')
         
        unique_name = ''
    
        for n in range(len(file_part)-2):
            
            # e.g. car.Columbus_EO_Run01_s2_301_15_00_42.40561128-Oct-2007_11-00-47.194_Frame_74.02085.01928
            unique_name =  unique_name + file_part[n] + '_'
        
        label_set[unique_name] = label_num
    
    return label_set  

# ================================================================================================        

#type_dir            = '/data/shared/datasets/CarsOverheadWithContext/Sorted_Cars_By_Type'
#data_dir            = '/data/shared/datasets/CarsOverheadWithContext/120x120_15cm_24px-exc_v5-marg-32_expanded'
type_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/Sorted_Cars_By_Type'

#data_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/64x64_15cm_24px-exc_v5-marg-32_expanded_25p'
#data_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/64x64_15cm_24px-exc_v5-marg-32_expanded'
data_dir            = '/data/shared/datasets/CarsOverheadWithContext/64x64_15cm_24px-exc_v5-marg-32_expanded_hue-rot'
#data_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/120x120_15cm_24px-exc_v5-marg-32_expanded'
#data_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/232x232_15cm_24px-exc_v5-marg-32_expanded'
#data_dir            = '/g/g17/mundhetn/data/CarsOverheadWithContext/116x116_15cm_expanded_66p_3-scales_gray_r-motion'

train_list_file     = data_dir + '/train_3types_extra-trucks-3x.txt'
test_list_file      = data_dir + '/test_3types_extra-trucks-3x.txt'
ignore_list         = {4}
#ignore_list         = {}

# For old style 2, for expanded 3
endlen = 3

# ================================================================================================
# Get the lable of each car by directory
# ================================================================================================
 
files_root  = os.listdir(type_dir)

print 'Do Lists'

label_set = {}

for file_dir in sorted(files_root):
        
    if os.path.isdir(type_dir + '/' + file_dir):
        
        # e.g. /g/g17/mundhetn/data/CarsOverheadWithContext/Sorted_Cars_By_Type/CSUAV
        
        print "Doing Type Directory: " + type_dir + '/' + file_dir
        
        other_loc    = type_dir + '/' + file_dir + '/Other'
        pickup_loc   = type_dir + '/' + file_dir + '/Pickup' 
        sedan_loc    = type_dir + '/' + file_dir + '/Sedan'
        unknown_loc  = type_dir + '/' + file_dir + '/Unknown'
        
        label_set = getLabels(other_loc,    label_set, 1)
        label_set = getLabels(pickup_loc,   label_set, 2)
        label_set = getLabels(sedan_loc,    label_set, 3)
        label_set = getLabels(unknown_loc,  label_set, 4)
        
# ================================================================================================
# Get each sample and match to label
# ================================================================================================

test_count      = [0,0,0,0,0]
train_count     = [0,0,0,0,0]
test_samples    = 0
train_samples   = 0
car_prefix      = 'car'

#label_str_list  = ["0 -1 -1 -1", "1 -1 -1 -1", "2 -1 -1 -1", "3 -1 -1 -1", "1 2 3 4"]
label_str_list  = ["0","1","2","3","4"]

# If zero, we give no extra samples, otherwise we do that many extra (times) for each sample
extra_samples   = [0,0,3,0,0]

files_root  = os.listdir(data_dir)

print 'Do Lists'

test_out_list       = open(test_list_file, 'w')
train_out_list      = open(train_list_file, 'w')

for file_dir in sorted(files_root):
        
    if os.path.isdir(data_dir + '/' + file_dir):
        
        print "Doing Data Directory: " + data_dir + '/' + file_dir
        
        test_loc    = data_dir + '/' + file_dir + '/test'
        test_files  = os.listdir(test_loc)
        
        for test_file in sorted(test_files):
            car_class   = getClass(test_file, label_set, car_prefix, endlen) 
            test_count[car_class] += 1 
            if car_class in ignore_list:
                continue
            line        = test_loc + '/' + test_file + '\t' + "{}".format(label_str_list[car_class]) + "\n"
            test_out_list.write(line)
            test_samples += 1
           
        train_loc   = data_dir + '/' + file_dir + '/train'
        train_files  = os.listdir(train_loc)
                    
        for train_file in sorted(train_files):
            car_class   = getClass(train_file, label_set, car_prefix, endlen)  
            train_count[car_class] += 1          
            if car_class in ignore_list:
                continue
            line        = train_loc + '/' + train_file + '\t' + "{}".format(label_str_list[car_class]) + "\n"
            
            train_out_list.write(line)
            train_samples += 1
             
            for e in range(extra_samples[car_class]):            
                train_out_list.write(line)
                train_samples += 1
                train_count[car_class] += 1


print "Writing: " + test_list_file
test_out_list.close()
print "Writing: " + train_list_file
train_out_list.close()

print "Train Neg: "     + "{}".format(train_count[0])
print "Train Other: "   + "{}".format(train_count[1])
print "Train Pickup: "  + "{}".format(train_count[2])
print "Train Sedan: "   + "{}".format(train_count[3])
print "Train Unknown: " + "{}".format(train_count[4])
print "Train SAMPLES: " + "{}".format(train_samples)

print "Test Neg: "     + "{}".format(test_count[0])
print "Test Other: "   + "{}".format(test_count[1])
print "Test Pickup: "  + "{}".format(test_count[2])
print "Test Sedan: "   + "{}".format(test_count[3])
print "Test Unknown: " + "{}".format(test_count[4])
print "Test SAMPLES: " + "{}".format(test_samples)

           
            