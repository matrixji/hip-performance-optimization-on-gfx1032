#!/usr/bin/bash

this_dir=$(cd $(dirname $0); pwd)

cd $this_dir

for cu_file in *.cu ; do
    hip_file=$(echo $cu_file | sed 's/\.cu$/.hip/')
    hip_file_path=../src/$hip_file
    hipify-perl $cu_file \
        | sed 's/CUDA_CHECK/HIP_CHECK/g' \
        > $hip_file_path
done
