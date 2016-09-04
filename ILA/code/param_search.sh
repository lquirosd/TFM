#!/bin/bash

echo "z,w,g,a,ft,trt,tet,p,r,f1" > ./work/summary.txt
for z in 0.1 0.2 0.3 0.4
do
	for w in 9 17 33 65
	do
		for g in 1 3 9 12
		do
			new_dir=_z${z}_w${w}_g${g}
			python imgFeatures_float.py -trDir ../DataCorpus/train/ -teDir ../DataCorpus/test/ -z $z -w $w -g $g -o ./work/ -s > ./work/${new_dir}_imgFeatures.log
			python struct_sentByrow.py -i ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/train/
			cd ./work/${new_dir}/train/
			cat Mss* > train_features_22_z${z}_w${w}_g${g}.txt
			cd ../../../
			python struct_sentByrow.py -i ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.pickle -out ./work/${new_dir}/test/
			cd ./work/${new_dir}/test/
			cat Mss* > test_features_17_z${z}_w${w}_g${g}.txt
			cd ../../../
			crfsuite learn -a arow -m ./work/${new_dir}/z${z}_w${w}_g${g}_arow.model ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/trainCRFsuite_arrow.log
			crfsuite learn -a lbfgs -m ./work/${new_dir}/z${z}_w${w}_g${g}_lbfgs.model ./work/${new_dir}/train/train_features_22_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/trainCRFsuite_lbfgs.log
			crfsuite tag -m ./work/${new_dir}/z${z}_w${w}_g${g}_arow.model -qt ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/results_arow.txt
			crfsuite tag -m ./work/${new_dir}/z${z}_w${w}_g${g}_lbfgs.model -qt ./work/${new_dir}/test/test_features_17_z${z}_w${w}_g${g}.txt > ./work/${new_dir}/results_lbfgs.txt
			echo -n "$z,$w,$g,arow" >> ./work/summary.txt
			grep "Parsing Image Data" ./work/${new_dir}_imgFeatures.log | awk '{printf ",%f",$4/39}' >> ./work/summary.txt
			grep "Total seconds required for training" ./work/${new_dir}/trainCRFsuite_arrow.log | awk '{printf ",%f",$6}' >> ./work/summary.txt
			grep "Elapsed time" ./work/${new_dir}/results_arow.txt | awk '{printf ",%f",$3}' >> ./work/summary.txt
			grep "Macro-average" ./work/${new_dir}/results_arow.txt | sed 's:[()]:,:g' | awk -F ',' '{printf ",%f,%f,%f\n",$4,$5,$6}' >> ./work/summary.txt
			echo -n "$z,$w,$g,lbfgs" >> ./work/summary.txt
			grep "Parsing Image Data" ./work/${new_dir}_imgFeatures.log | awk '{printf ",%f",$4/39}' >> ./work/summary.txt
			grep "Total seconds required for training" ./work/${new_dir}/trainCRFsuite_lbfgs.log | awk '{printf ",%f",$6}' >> ./work/summary.txt
			grep "Elapsed time" ./work/${new_dir}/results_lbfgs.txt | awk '{printf ",%f",$3}' >> ./work/summary.txt
			grep "Macro-average" ./work/${new_dir}/results_lbfgs.txt | sed 's:[()]:,:g' | awk -F ',' '{printf ",%f,%f,%f\n",$4,$5,$6}' >> ./work/summary.txt
			rm -rf ./work/${new_dir}/test
			rm -rf ./work/${new_dir}/train
		done
	done
done
