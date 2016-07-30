#!/bin/bash
#--- run tagger 
for file in Mss*
    do
        crfsuite tag -m ../z0.3_w32_g3_arow_pos.model -t $file | sed '/^$/d' | sed 's:^\([^0-9]\):#\1:g' > ./test_pos/${file}.results

#--- Gen imges 
python process_results.py -imgData ./test_features_17_z0.3_w32_g3.pickle -t ./  
