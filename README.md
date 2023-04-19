This repository contains a simple Python script which runs Mediapipe's hair segmentation model using OpenCV on RISC-V CPU.

### How to start

1. Download a compiler: [Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz](https://occ.t-head.cn/community/download?id=4090445921563774976) and put to `opencv_riscv64` folder without unpacking.

2. Build OpenCV:
    ```bash
    docker build -t opencv opencv_riscv64
    docker run -v $(pwd):/mnt opencv sh -c "cp *.tar.gz /mnt"
    ```

3. Download deep learning [model](https://storage.googleapis.com/mediapipe-assets/hair_segmentation.tflite?generation=1661875756623461)

4. Start a telegram bot:

    ```bash
    python bot.py --token=xxx
    ```

### Performance

HW: Sipeed Lichee RV Dock (Allwinner D1 aka XuanTie C906 CPU)

OS: [20211230_LicheeRV_debian_d1_hdmi_8723ds](https://mega.nz/folder/lx4CyZBA#PiFhY7oSVQ3gp2ZZ_AnwYA/folder/xtxkABIB)

| input | no RVV | RVV |
|---|---|---|
| 512x512 | 3.5 seconds | 2.35 seconds (x1.48) |

<img src="./images/example.jpg" height="512">
