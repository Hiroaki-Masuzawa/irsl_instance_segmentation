#!/bin/bash


# abs_script=$(readlink -f "$0")
# abs_dir=$(dirname ${abs_script})

_ROS_IP=${ROS_IP}
_ROS_HOSTNAME=${ROS_HOSTNAME}
_ROS_MASTER_URI=${ROS_MASTER_URI}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ros-setup)
            _ROS_SETUP="$2"
            shift
            shift
            ;;
        --ros-ip)
            _ROS_IP="$2"
            shift
            shift
            ;;
        --ros-hostname)
            _ROS_HOSTNAME="$2"
            shift
            shift
            ;;
        --ros-master-uri)
            _ROS_MASTER_URI="$2"
            shift
            shift
            ;;
        --help)
            echo "run.sh [-w|--workspace workspace] [--ros-setup ROS_SETUP] [--ros-ip ROS_IP] [--ros-hostname ROS_HOSTNAME] [--ros-master-uri ROS_MASTER_URI] "
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done



if [ "${_ROS_IP}" == "" ]; then
    rosip_option=" "
else 
    rosip_option=" --env ROS_IP=${_ROS_IP} "
fi

if [ "${_ROS_HOSTNAME}" == "" ]; then
    roshostname_option=" "
else 
    roshostname_option=" --env ROS_HOSTNAME=${_ROS_HOSTNAME} "
fi

if [ "${_ROS_MASTER_URI}" == "" ]; then
    rosmaster_option=" "
else 
    rosmaster_option=" --env ROS_MASTER_URI=${_ROS_MASTER_URI} "
fi

set -x 
docker run -it --rm --gpus all \
${rosip_option} ${roshostname_option} ${rosmaster_option} \
--name irsl_instance_segmentation_ram_grounding_sam \
irsl_instance_segmentation \
bash -c "source /catkin_ws/install/setup.bash && roslaunch irsl_instance_segmentation ram_grounding_sam.launch"
