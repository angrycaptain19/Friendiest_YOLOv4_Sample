# Friendiest_YOLOv4_Sample
最友善的YOLOv4範例程式，如果你認為YOLOv4的程式不友善，那麼請參考看看這個

* [介紹](#intro)
* [DEMO](#demo)
* [使用說明](#how)
* [修改參數](#custom)

## <a id='intro'>介紹</a>
此程式搭配 <a href='https://chiachun0818.medium.com/jetson-nano-%E4%B9%8B-darknet%E8%A9%B3%E8%A7%A3%E4%BB%A5%E5%8F%8A%E6%9C%80%E7%B0%A1%E5%96%AE%E7%9A%84yolov4%E5%8D%B3%E6%99%82%E5%BD%B1%E5%83%8F%E8%BE%A8%E8%AD%98-248e369b93c3'>Jetson Nano 之 Darknet詳解以及最簡單的yolov4即時影像辨識</a> 使用，我們來深入了解一下 `darknet.py`，解析之後對於可以使用的副函式都初步瞭解了再進行改寫，變成一個最簡單的yolov4應用，不同於官方的使用佇列 (queue) 的形式，我們使用較為簡單直覺的 OpenCV 來改寫。如果你是個新手可以先看看這篇 [Jetson Nano使用YOLOv4並透過Tensor RT 進行加速](https://chiachun0818.medium.com/jetson-nano%E4%BD%BF%E7%94%A8yolov4%E4%B8%A6%E9%80%8F%E9%81%8Etensor-rt-%E9%80%B2%E8%A1%8C%E5%8A%A0%E9%80%9F-174f5ad46bb0)

## <a id='demo'>DEMO</a>
![運行結果](/figures/DEMO.gif)

## <a id='how'>使用說明:</a>
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

## <a id='custom'>修改參數</a>

```
# Parameters
win_title = 'YOLOv4 CUSTOM DETECTOR'
cfg_file = 'cfg/yolov4-tiny.cfg'
data_file = 'cfg/coco.data'
weight_file = 'yolov4-tiny.weights'
thre = 0.25
show_coordinates = False
show_size = (800, 800)
```
