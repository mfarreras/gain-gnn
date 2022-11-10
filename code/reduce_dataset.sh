#!/bin/bash

#start from root of dataset (currently in code folder)
cd ../data/gnnet-ch21-dataset-train
#create folder reduced_train
mkdir ../reduced_train
for d in */ ; do #for each folder (number of nodes name)
    #get original graphs folder and routings and copy them to  ../reduced_train/(nº nodes)
    mkdir ../reduced_train/$d
    d=${d%/}
    cp -r $d/graphs ../reduced_train/$d/graphs
    cp -r $d/routings ../reduced_train/$d/routings
    cd $d
    n=7
    for file in ./*
    do
        if [ ! -d "$file" ]; then
            if [ $n -eq 7 ]; then
                #recòrrer fitxers results i copiar-ne un de cada 7 a ../reduced_train/(nº nodes)
                cp $file ../../reduced_train/$d
                n=0
            fi
            n=$(($n + 1))
        fi
    done
    cd ..
done

    