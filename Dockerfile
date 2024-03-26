# This is an auto generated Dockerfile for ros:desktop-full
# generated from docker_images/create_ros_image.Dockerfile.em
FROM osrf/ros:noetic-desktop-focal

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y apt-utils curl wget git bash-completion build-essential sudo && rm -rf /var/lib/apt/lists/*

# Now create the same user as the host itself
ARG UID=1000
ARG GID=1000
RUN addgroup --gid ${GID} nn_pv
RUN adduser --gecos "ROS User" --disabled-password --uid ${UID} --gid ${GID} nn_pv
RUN usermod -a -G dialout nn_pv
RUN mkdir config && echo "nn_pv ALL=(ALL) NOPASSWD: ALL" > config/99_aptget
RUN cp config/99_aptget /etc/sudoers.d/99_aptget
RUN chmod 0440 /etc/sudoers.d/99_aptget && chown root:root /etc/sudoers.d/99_aptget

# Change HOME environment variable
ENV HOME /home/docker
RUN mkdir -p ${HOME}/nn_pv_ws/src

# Initialize the workspace
RUN cd ${HOME}/nn_pv_ws/src && /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_init_workspace"
RUN cd ${HOME}/nn_pv_ws /bin/bash -c "source source /opt/ros/${ROS_DISTRO}/setup.bash; catkin_make"

# set up environment
COPY ./update_bashrc /sbin/update_bashrc
RUN sudo chmod +x /sbin/update_bashrc ; sudo chown ros /sbin/update_bashrc ; sync ; /bin/bash -c /sbin/update_bashrc ; sudo rm /sbin/update_bashrc

# Install pip
RUN apt-get install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

# Install deepsort
RUN pip install gdown easydict 
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install segmentation models
RUN pip install albumentations
RUN pip install segmentation-models-pytorch

# Install extra tools
RUN apt-get update && apt-get install -y vim ranger nano

# Create package
COPY ./* ${HOME}/nn_pv_ws/src/nn_segmentation/
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x detect.py
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x train.py
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x utils.py
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x models.py
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x dataset.py
RUN cd ${HOME}/nn_pv_ws/src/nn_segmentation/ && chmod +x test.py

WORKDIR ${HOME}/nn_pv_ws/src/nn_segmentation/ 
