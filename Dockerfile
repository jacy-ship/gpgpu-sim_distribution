FROM ubuntu:14.04
RUN apt-get update -y
RUN apt-get install -y gcc g++ gcc-4.4 g++-4.4 make xutils bison flex doxygen python-pmw python-ply python-numpy libpng12-dev python-matplotlib doxygen graphviz git python-pmw python-ply python-numpy libpng12-dev libxi-dev libxmu-dev wget iproute2 expect xutils-dev libc-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libboost-dev libboost-system-dev libboost-filesystem-dev libboost-all-dev mpich libcuda1-304 binutils-gold vim
RUN wget http://developer.download.nvidia.com/compute/cuda/4_0/sdk/gpucomputingsdk_4.0.17_linux.run -P ~/.
RUN wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run -P ~/.
RUN wget https://gist.githubusercontent.com/Kiran-r/a7ea775e5ddc0474754be3f234c6b5f1/raw/d7e5f8bb8e2dc7db22680eb50fab27100f8af325/cuda.exp -P ~/.
RUN chmod 777 ~/gpucomputingsdk_4.0.17_linux.run ~/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
RUN echo "export CUDA_INSTALL_PATH=/usr/local/cuda" >> ~/.bashrc
RUN echo "export NVIDIA_COMPUTE_SDK_LOCATION=~/NVIDIA_GPU_Computing_SDK" >> ~/.bashrc
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.4 50
RUN git clone https://github.com/socal-ucr/gpgpu-sim_distribution ~/gpgpu-sim_distribution
RUN git clone https://github.com/socal-ucr/ispass2009-benchmarks ~/ispass2009-benchmarks
RUN expect ~/cuda.exp
