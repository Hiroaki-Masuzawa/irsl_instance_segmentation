#!/bin/bash

abs_script=$(readlink -f "$0")
abs_dir=$(dirname ${abs_script})

set -x

docker pull repo.irsl.eiiris.tut.ac.jp/irsl_base:cuda_12.1.0-cudnn8-devel-ubuntu22.04_one
docker build ${abs_dir}/.. -f ${abs_dir}/Dockerfile -t irsl_instance_segmentation
