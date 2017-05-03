cd ./src/

echo 'Generating labels pickle files'
python generate_labels.py ../codes/train_codes_02.csv
python generate_labels.py ../codes/test_codes_03.csv
mv ../codes/*.pickle .
echo 'done'
echo ''

echo 'Generating dataset lists'
cd ..
find -L . -path "*IRMA*train*png" > ./src/tmp_train.txt
find -L . -path "*IRMA*test*png" > ./src/tmp_test.txt
cd ./src/
head tmp_train.txt -n 300 > train_dataset.txt
head tmp_test.txt -n 20 > test_dataset.txt
echo 'done'
echo ''

echo 'Generating training descriptors file'
optirun python myalexnet_forward.py ../ train_dataset.txt train_codes_02.csv_labels.pickle
optirun python myalexnet_forward.py ../ test_dataset.txt test_codes_03.csv_labels.pickle
echo 'done'
echo ''

echo 'Generating retrieval files'
python search_euclidean.py train_dataset.txt_desc test_dataset.txt_desc ../IRMA/ImageCLEFmed2009_train.02/ ../IRMA/ImageCLEFmed2009_test.03/ 
echo 'done'
echo ''

rm tmp*.txt
mkdir ../generated_files
mv train* ../generated_files -f
mv test* ../generated_files -f
mv Retrieval ../generated_files -f
mv retrieval_result_fc7 ../generated_files -f

