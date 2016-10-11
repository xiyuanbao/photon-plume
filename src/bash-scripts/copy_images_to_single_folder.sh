# this script copies all the image pairs stored in different folders, 
# to a single folder and renames them in the process


# this is the case name
case_name=150x150-f16-disp2

# this is the destination where the images will be copied to
destination=/home/barracuda/a/lrajendr/Projects/camera_simulation/results/images/bos/error-analysis/dot-size/processing/$case_name/reordered-images

# if the destination directory does not exist, create one; it it does, then delete it
rm -r $destination
mkdir $destination

# this is the source filepath
filepath=/home/barracuda/a/lrajendr/Projects/camera_simulation/results/images/bos/error-analysis/dot-size/$case_name

# find the number of folders in this path
num_folders=$(find $filepath/* -maxdepth 0 -type d | wc -l)

# print out the number of folders
echo "number of folders: " $num_folders

# find the names of the folders in the path
folder_names=$(find $filepath/* -maxdepth 0 -type d | sort --version-sort)

# iterate through the folder names
for dir in $folder_names
do
  # print out the folder names
  echo $dir
  echo
  # create a copy of the .tif files with a different name
  for file in $(find $dir/* -name *.tif)
  do
    echo $file
    echo
    

    # set the new filename under which the image will be saved
    # this is the first part of the name
    new_filename_1="dotsize_"
    
    echo "new_filename_1", $new_filename_1
    
    # this is the second part of the name
    new_filename_2="$(basename $dir)"

    echo "new_filename_2", $new_filename_2

    # this is the third part of the name
    #string=$(basename $file)
    #new_filename_3=${string: -6}
    new_filename_3=$(($(find $destination/* -maxdepth 0 -type f | wc -l) + 1))  
     
    echo "new_filename_3", $new_filename_3

    # this is the full name of the file including the path
    new_filename="$destination/$new_filename_1$new_filename_2_$(printf %02d $new_filename_3).tif"
    
    # print out new filename
    echo $new_filename
    
    # copy file to the destination but under the new file name
    cp $file $new_filename
     
  done
  # copy .tif files inside the folder to the destination
  #cp "$dir *.tif" 
done




