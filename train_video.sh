#!/bin/bash
dataset=$1
model_name=$2
num_objects=$3
beta=$4
communication=$5
dynamics=$6
device=$7


ExpName=${model_name}"_"${dataset}
echo "doing experiment: ${ExpName}"

nohup python train_object_discovery.py \
--dataset=${dataset} \
--model=${model_name} \
--model.vmnn_params.communication=${communication} \
--model.vmnn_params.beta=${beta} \
--model.vmnn_params.dynamics=${dynamics} \
--dataset.max_num_objects=${num_objects} \
--device=${device} \




