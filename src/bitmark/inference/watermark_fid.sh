#!/bin/bash

eval "$(conda shell.bash hook)"
source ~/.bashrc

architecture=infinity # infinity, instella_iar, big_r



case "$architecture" in
    "infinity")
        conda activate inf_water
        ;;
    "instella_iar")
        conda activate instella
        ;;
    "big_r")
        conda activate inf_water
        ;;
    *)
        echo "Error: Unsupported architecture: $architecture"
        echo "Supported architectures: infinity, instella_iar"
        exit 1
        ;;
esac


image_size=512 # 256
if [ $image_size -eq 256 ]; then
    model='BiGR-L-d24' # Model name
else
    model='BiGR-L-512' # Model name
fi
# debug print-outs
echo USER: $USER
which conda
which python
scales=2
delta=2
gen_data_path="/path/${architecture}/${model}/delta_${1}"
out_dir=$gen_data_path
nat_data_path="/path/datasets/val"
# run the code
PYTHONPATH=. python3 img_quality_eval.py \
--nat_data_path ${nat_data_path} \
--gen_data_path ${gen_data_path} \
--out_dir ${out_dir}