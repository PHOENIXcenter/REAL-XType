#!/bin/bash
# application
seed=0
device="/cpu"
data_path="/data/linhai/HCC/data"
method="XType"
for ((i=0; i<6; i++))
do
	for test_cohort in "SH" "GZ" "FZ"
	do
		python application.py --method $method --test_cohort $test_cohort --para_id $i --seed $seed --data_path $data_path --device $device
	done
done
python application.py --evaluate True --method $method --km_viz True

# benchmarking
for method in "XType" "XTypeNoCDAN" "XTypeNoCox" "XTypeSuperviseOnly" "RandomForest" "LogisticRegression"
do
	for test_cohort in "SH" "GZ" "FZ" "Gao"
	do
		for ((i=0; i<9; i++))
		do
			for ((seed=0; seed<10; seed++))
			do
				python benchmarking.py --method $method --test_cohort $test_cohort --para_id $i --seed $seed --data_path $data_path --device $device
			done
		done
	done
done
python benchmarking.py --evaluate True
