ARG BASE_IMAGE=repo.irsl.eiiris.tut.ac.jp/irsl_base:cuda_12.1.0-cudnn8-devel-ubuntu22.04_one

FROM ${BASE_IMAGE}

ENV IRSL_WS=/irsl_catkin_ws
ENV IRSL_PROJECT=irsl_instance_segmentation 

WORKDIR ${IRSL_WS}

RUN apt update && apt install -y python3 python3-distutils curl git python-is-python3 python3-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/pip/get-pip.py | python3
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# install detectron
RUN pip3 install shapely 
RUN cd / && git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;8.9"
RUN cd / && pip3 install -e detectron2_repo

RUN pip install fastapi uvicorn opencv-python pillow
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y libopencv*
RUN pip install python-multipart
ENV PYTHONPATH=/detectron2_repo


# install Grounded-Segment-Anything
ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

RUN cd / \
    && git clone --recursive https://github.com/IDEA-Research/Grounded-Segment-Anything.git \
    && cd /Grounded-Segment-Anything \
    && python -m pip install -e segment_anything \
    && pip install --no-build-isolation -e GroundingDINO \
    && git clone https://github.com/xinyu1205/recognize-anything.git \
    && pip install -r ./recognize-anything/requirements.txt \
    && pip install -e ./recognize-anything/

RUN pip install supervision==0.21
RUN pip install litellm

RUN apt update && apt install -y wget
RUN cd /Grounded-Segment-Anything \
    && wget --quiet https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    && wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    && wget --quiet https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
    && wget --quiet  https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth 


ENV PYTHONPATH=/Grounded-Segment-Anything:/Grounded-Segment-Anything/segment_anything:/Grounded-Segment-Anything/recognize-anything:/detectron2_repo

RUN apt update && apt install -q -qq -y python3-catkin-tools ros-${ROS_DISTRO}-cv-bridge 
RUN pip install numpy==1.26.4

RUN mkdir -p ${IRSL_WS}/src/${IRSL_PROJECT}
COPY . ${IRSL_WS}/src/${IRSL_PROJECT}/
RUN ( cd src; vcs import --recursive < ${IRSL_PROJECT}/dot.rosinstall )
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin config --install && IRSL_BUILD_LOCAL=true catkin build"

CMD [ "bash", "-c", "source ${IRSL_WS}/install/setup.bash && roslaunch ${IRSL_PROJECT} run.launch" ]