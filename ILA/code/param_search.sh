#!/bin/bash

for z in 0.4
do
	for w in 33
	do
		for g in 12
		do
			new_dir=_z${z}_w${w}_g${g}
			python imgFeatures_float.py -trDir ../DataCorpus/train/ -teDir ../DataCorpus/test/ -z $z -w $w -g $g -o ./work/ -s > ./work/${new_dir}/imgFeatures.log
			python struct_sentByrow.py -i ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/train/
			cd ./work/${new_dir}/train/
			cat Mss* > train_features_22_z${z}_w${w}_g${g}.txt
			cd ../../../
			python struct_sentByrow.py -i ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/test/
			cd ./work/${new_dir}/test/
			cat Mss* > test_features_17_z${z}_w${w}_g${g}.txt
			cd ../../../
			crfsuite learn -a arow -L AROWLOG -l -m ./work/${new_dir}/z${z}_w${w}_g${g}_arow.model ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.txt
			crfsuite learn -a lbfgs -L LBFGSLOG -l -m ./work/${new_dir}/z${z}_w${w}_g${g}_lbfgs.model ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.txt
			crfsuite tag -m ./work/${new_dir}/z${z}_w${w}_g${g}_arow.model -qt ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/results_arrow.txt
			crfsuite tag -m ./work/${new_dir}/z${z}_w${w}_g${g}_lbfgs.model -qt ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/results_arrow.txt
		done
	done
done
