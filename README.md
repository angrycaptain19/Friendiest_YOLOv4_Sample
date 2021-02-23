# Friendiest_YOLOv4_Sample
The most friendliest yolov4 live-detection code

# 介紹
此程式搭配 <a href=''>Jetson Nano 之 Darknet詳解以及最簡單的yolov4即時影像辨識</a> 使用，我們來深入了解一下 `darknet.py`，解析之後對於可以使用的副函式都初步瞭解了再進行改寫，變成一個最簡單的yolov4應用，不同於官方的使用佇列 (queue) 的形式，我們使用較為簡單直覺的 OpenCV 來改寫。

# 使用說明:
首先要先建置darknet的環境，先下載 [darknet](https://github.com/AlexeyAB) 的github：
```
$ git clone https://github.com/AlexeyAB/darknet.git
$ cd darknet
```
修改 Makefile
```
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=0
OPENMP=1
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0

......

USE_CPP=0
DEBUG=0

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]
```

進行build

```
$ make
```

下載 yolov4.weights 權重檔案
```
$ wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -q --show-progress --no-clobber
```

下載我的程式並複製到 darknet 當中
```
$ git clone https://github.com/p513817/Friendiest_YOLOv4_Sample.git
$ cp ./Friendiest_YOLOv4_Sample/yolov4_inference.py ./yolov4_inference.py
```

執行程式
```
python3 yolov4_inference.py
```

運行結果

![運行結果](/figures/01.png)