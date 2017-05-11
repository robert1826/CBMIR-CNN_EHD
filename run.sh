cd ./src/

echo 'Generating labels pickle files'
python generate_labels.py ../codes/train_codes_02.csv
python generate_labels.py ../codes/test_codes_03.csv
mv ../codes/*.pickle .
echo 'done'
echo ''

echo 'Generating dataset lists'
find -L .. -path "*IRMA*train*png" > ./tmp_train.txt
find -L .. -path "*IRMA*test*png" > ./tmp_test.txt
head tmp_train.txt -n 1000 > train_dataset.txt
head tmp_test.txt -n 100 > test_dataset.txt
echo 'done'
echo ''

echo 'Generating descriptors files'
optirun python myalexnet_forward.py ./ train_dataset.txt train_codes_02.csv_labels.pickle
optirun python myalexnet_forward.py ./ test_dataset.txt test_codes_03.csv_labels.pickle
echo 'done'
echo ''

echo 'Generating retrieval files'
python search_euclidean_pool.py train_dataset.txt_desc test_dataset.txt_desc ../IRMA/ImageCLEFmed2009_train.02/ ../IRMA/ImageCLEFmed2009_test.03/ 
echo 'done'
echo ''

# EHD part
cd ../mpeg7fexlin/

echo 'Getting dataset lists'
cp ../src/*dataset*txt .
echo 'done'
echo ''

echo 'Converting images to JPG'
mkdir convert_dir
for img in $(cat train_dataset.txt test_dataset.txt); do 
    filename=$(basename $img .png)
    convert $img ./convert_dir/$filename.jpg
done
echo 'done'
echo ''

echo 'Generated new imageList file for the converted images'
find . -path "*jpg" > ./dataset-converted.txt
echo 'done'
echo ''

echo 'Generating image descriptors'
export LD_LIBRARY_PATH=$(pwd)/solibs/
./MPEG7Fex EHD dataset-converted.txt ehd_out.txt > mpeg7fexlin_out.txt
rm mpeg7fexlin_out.txt
echo 'done'
echo ''

cd ../src/
echo 'EHD retrieval'
python re_rank_EHD.py ../mpeg7fexlin/ehd_out.txt
echo 'done'
echo ''

rm tmp*.txt
mkdir ../generated_files
mv train* ../generated_files -f
mv test* ../generated_files -f
mv Retrieval ../generated_files -f
mv retrieval_result_fc7 ../generated_files -f

# clean mpeg dir
cd ../mpeg7fexlin/
rm *dataset*
rm -r convert_dir
mv ehd_out.txt ../generated_files
