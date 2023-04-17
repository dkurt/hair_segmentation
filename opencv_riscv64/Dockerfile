FROM ubuntu:22.04

USER root

RUN dpkg --add-architecture riscv64 && \
    sed -i -E 's|^deb ([^ ]+) (.*)$|deb [arch=amd64] \1 \2\ndeb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ \2|' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-minimal \
        python3-pip \
        libpython3-dev:riscv64 \
        crossbuild-essential-riscv64 \
        cmake \
        git

RUN python3 -m pip install numpy==1.21.5 && \
    python3 -c "import numpy; print(numpy.__version__); print(numpy.__file__)"

RUN git clone --depth 1 https://github.com/opencv/opencv

RUN cmake \
        -DGNU_MACHINE=riscv64-linux-gnu \
        -DCMAKE_TOOLCHAIN_FILE=$(realpath opencv/platforms/linux/riscv64-gcc.toolchain.cmake) \
        -DCMAKE_INSTALL_PREFIX=$(realpath opencv_install) \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_LIST=dnn,python3,videoio,imgproc \
        -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.10/dist-packages/numpy/core/include/  \
        -DPYTHON3_INCLUDE_PATH="/usr/include/python3.10;/usr/include/" \
        -DPYTHON3_LIBRARIES=/usr/lib/riscv64-linux-gnu/libpython3.10.so \
        -DPYTHON3_EXECUTABLE=/usr/bin/python3.10  \
        -DPYTHON3_CVPY_SUFFIX=".cpython-310-riscv64-linux-gnu.so" \
        -S opencv -B opencv_build && \
    cmake --build opencv_build -j$(nproc --all) && \
    cmake --install opencv_build --prefix opencv_install && \
    tar -cvzf opencv_install.tar.gz opencv_install
