#!/bin/bash

for z in 0.1 0.2 0.3 0.4 0.5
do
	for w in 8 16 32 64
	do
		for g in 1 3 6 9 12
		do
			new_dir=z${z}_w${w}_g${g}
			mkdir -p work/${new_dir}/train work/${new_dir}/test
			python imgFeatures.py -imgDir ../DataCorpus/train/ -z $z -w $w -g $g -o ./work/${new_dir}/train/
			python imgFeatures.py -imgDir ../DataCorpus/test/ -z $z -w $w -g $g -o ./work/${new_dir}/test/
			python struct_sentByrow.py -i ./work/${new_dir}/train/features_22_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/train/
			cd ./work/${new_dir}/train/
			cat Mss* > crfsuite_features_22_z${z}_w${w}_g${g}_pnub.txt
			cd ../../../
			python struct_sentByrow.py -i ./work/${new_dir}/test/features_17_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/test/
			cd ./work/${new_dir}/test/
			cat Mss* > crfsuite_test_17_z${z}_w${w}_g${g}_pnub.txt
			cd ../../../
			crfsuite learn -a arow -m ./work/${new_dir}/crfsuite_22_z${z}_w${w}_g${g}_pnub_arow.model ./work/${new_dir}/train/crfsuite_features_22_z${z}_w${w}_g${g}_pnub.txt
			crfsuite tag -m ./work/${new_dir}/crfsuite_22_z${z}_w${w}_g${g}_pnub_arow.model -qt ./work/${new_dir}/test/crfsuite_test_17_z${z}_w${w}_g${g}_pnub.txt > ./work/${new_dir}/results.txt
		done
	done
done
