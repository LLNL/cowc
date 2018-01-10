(1)	Introduction

	These are scripts for extracting training patches of different types from the COWC-M dataset. Two of them will extract patches and label them for use with Caffe in a way that is similar to our original patches from ECCV ’16. The main differences is that now the type of car is labeled as Sedan, Pickup, Other and Unknown rather than as just ‘car’. We describe them more later, but the scripts are:
	
	CreateDetectionPatches.py
	CreatePatchLabels.py
	
	Note that you can just download pre-extracted patches from our ftp at:
	
	ftp://gdo152.ucllnl.org/cowc-m/datasets/64x64_15cm_24px-exc_v5-marg-32_expanded.tgz
	
	
	The other script will extract cars with detection locations in a manner more compatible with detection methods such as Faster-RCNN. This creates multiple patch scenes and a location label for each scene. This is called:
	
	CreatePatchScenes.py

(2)	Creating new patches

	There are only three steps to creating your own training patches:
	
	1)	Open up the python scripts and edit the path locations at the top
	2)	Download the data files shown at the top of the script
	3)	Run the script.
	
	I have left the paths as I would use them so you can see an example of usage, but you need to change these. For example:
	
	unique_list         = '/Users/mundhenk1/Downloads/temp/Objects_15cm_24px-exc_v5-marg-32.pickle'
	
	You might change to:
	
	unique_list         = '/my/directory/on/my/machine/Objects_15cm_24px-exc_v5-marg-32.pickle'
	 
	Don’t literally do this, just point it towards your local copy. 
	
	You then create new patches by first calling CreateDetectionPatches.py. This will create a set of training images. Next you will need to run CreatePatchLabels.py to create a set of patch labels. Once you have called both, you should have all you need to train a Caffe network on your data. 

(3)	Creating new scenes

	You run this in the same way as for creating training patches, but you don’t need to run a script to extract labels. This is in part because different detection engines use different labeling schemes. You should use the script CreatePatchScenes.py more as an example for how to do this and make changes as needed. 


