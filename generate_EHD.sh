cd ./mpeg7fexlin/

echo 'Generating dataset lists'
find -L .. -path "*IRMA*train*png" > ./tmp_train.txt
head tmp_train.txt -n 300 > train_dataset.txt
echo 'done'
echo ''

echo 'Converting images to JPG'
mkdir convert_dir
for img in $(cat train_dataset.txt); do 
    filename=$(basename $img .png)
    convert $img ./convert_dir/$filename.jpg
done
echo 'done'
echo ''

echo 'Generated new imageList file for the converted images'
find . -path "*jpg" > ./train_dataset-converted.txt
echo 'done'
echo ''

echo 'Generating image descriptors'
export LD_LIBRARY_PATH=$(pwd)/solibs/
./MPEG7Fex EHD train_dataset-converted.txt ehd_out.txt
echo 'done'
echo ''

