# FSFNet: Accelerator-Aware Fast Spatial Feature Network for Real-Time Semantic segmentation
                                                                                                                                         
[![Video Label](http://img.youtube.com/vi/89sccOnl41g/0.jpg)](https://www.youtube.com/watch?v=89sccOnl41g)

## Requirements
   * [Ubuntu 16.04](https://ubuntu.com/)
   * [Python 3.7.4](https://www.python.org/)
   * [NVIDIA 1080Ti GPU](https://www.nvidia.com/ko-kr/)
   * [TensorRT](https://github.com/NVIDIA/TensorRT)
   * [Pytorch 1.1.0](https://pytorch.org/) 
   * [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit) and [CuDNN v7.3](https://developer.nvidia.com/cudnn)
   * [Cityscapes dataset](https://www.cityscapes-dataset.com/submit/)
   * Additional Python packages: [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), [Pillow](https://pypi.org/project/Pillow/), [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) and [visdom](https://anaconda.org/conda-forge/visdom)
   
   
## Installation
   * Clone this files
<pre><code>
cd FSFNet
git clone https://github.com/computervision8/FSFNet.git
</code></pre>   
   * Install dependencies:

<pre><code>
cd FSFNet
pip install requirements.txt
</code></pre>   

   * Install [PyCuda](https://wiki.tiker.net/PyCuda/Installation/) and [TensorRT(v.5.1.5.0)](https://github.com/NVIDIA/TensorRT)

## Train
   * Only Cityscapes dataset
<pre><code>
# encoder-decoder architecture train
cd FSFNet/train
python main.py --savedir FSFNet --datadir /home/user/citysacpes/ --num-epochs 200 --batch-size 6

# only decoder
cd FSFNet/train
python main.py --decoder --savedir FSFNet --datadir /home/user/citysacpes/ --num-epochs 200 --batch-size 6
</code></pre>
   * pretrained using Imagenet
<pre><code>

# pretrained imagenet
cd FSFNet/imagenet
python main.py /home/user/DB/ILSVRC2012

# decoder
cd FSFNet/train
python main.py --decoder --savedir FSFNet --datadir /home/user/citysacpes/ --num-epochs 200 --batch-size 6 --pretrained "../save/FSF_encoder_pretrained(save)/FSFNet_encoder_pretrained.pth.tar"

</code></pre>


## Test
   * Evaluation on Cityscapes test server
<pre><code>
cd FSANet/eval
python eval_cityscapes_server.py

cd FSFNet/eval/save_results
zip test.zip ./*

# go to https://www.cityscapes-dataset.com/
# login cityscapes id 
# go to https://www.cityscapes-dataset.com/login/
# submit FSFNet result zip file


</code></pre>
   * Test the inference speed
     * TensorRT(v5.1.5) does not support bilinear interpolation so we used nearest neighbor interpolation instead of using bilinear interporlation. 
        The gap between nearest neighbor interpolation and bilinear interpolation FPS measurements in Pytorch is only 5.2 FPS. 
<pre><code>
cd FSANet/eval/latency
python eval_forwardTime.py
</code></pre>
   * Accuracy evaluation using intersection-over union (IoU) 
<pre><code>
cd FSFNet/eval
python eval_iou_Cityscapes.py
python eval_iou_Camvid.py.py
</code></pre>
   * Real-world Evaluation 
<pre><code>
# 1. generates result images 
cd FSANet/eval/
python eval_cityscapes_color.py

# 2. save result images
cd FSFNet/eval/save_color/

# 3. move result file to 2_result folder
cd FSFNet/realworld_sample_images/2_result/

# 4. run saveImageResult.py to combine original and result images
python saveImageResult.py

</code></pre>   
   * NVIDIA Jetson TX2 Evaluation
<pre><code>
# 1. install JetPack 3.0 on a NVIDIA Development Kit. JetPack can flash the Jetson TK2

# 2. install python3
sudo apt-get install cmake python3-pip

# 3. install pytorch
git clone -b v1.1.0 https://github.com/pytorch/pytorch
cd pytorch
git submodule update --init --recursive
time python3 setup.py install 
sudo pip3 install -r requirements.txt
sudo python3 setup.py install
gedit tools/setup_helpers/nccl.py
USE_NCCL = False
sudo nvpmodel -m 0
cd /usr/bin/
sudo jetson_clocks

# 4. install numpy, thop, and tqdm
sudo apt-get install python3-numpy
sudo pip3 install thop
sudo apt install python3-tqdm

# 5. install pycuda
sudo pip3 install pycuda
sudo pip3 -vvv install pycuda
export PATH=/usr/local/cuda-7.0/bin:$PATH   =>check cuda version
sudo su -
pip3 install pycuda
reboot
sudo pip3 install pycuda

# 6. download FSFNet 
cd FSFNet/eval
cd FSANet/eval/latency
python eval_forwardTime.py
</code></pre>  


## Result(Cityscapes)
   * Average results
     |Method| IoU Classes|iIoU Classes|IoU Categories|iIoU Categories|
     |---|:---:|:---:|:---:|:---:|
     |FSFNet|68.3798|42.2927|86.4462|72.714|
     |FSFNet(pretrained)|69.1319|43.0262|86.5888|72.554|
     
     
   * Class results(IoU)
     |Method|road|sidewalk|building|wall|fence|pole|traffic light|traffic sign|vegetation|terrain|sky|person|rider|car|truck|bus|train|motorcycle|bicycle|
     |---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
     |FSFNet|97.7996|81.1904|89.7836|40.713|46.297|54.2191|61.0444|65.709|91.8087|69.031|94.0309|77.3874|57.5401|92.8033|47.69|61.4249|56.1338|48.955|65.6548|
     |FSFNet(pretrained)|97.7055|81.1631|90.2109|41.7583|	47.0695|54.1891|61.1365|65.3923|91.8746|	69.4297|94.2097|77.8652|57.8774|92.887|47.3863|64.4488|59.4483|53.1812|66.2731	|
   
## Result(CamVid)
   * Class results(IoU)
     |Method| IoU Classes|
     |---|:---:|
     |FSFNet|63.26|

     
     |Method|Sky|Building|Pole|Road|Pavement|Tree|SignSymbol|Fence|Car|Pedestrian|Bicyclist|
     |---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
     |FSFNet|91.5|79.37|29.79|90.22|70.33|76.14|39.5|40.47|78.68|48.62|51.28|


     

## Result(Mapillary)
   * Class results(IoU)
     |Method| IoU Classes|
     |---|:---:|
     |FSFNet|24.5|
     

     
     
     
     
     


     


## Achknowledgement
  * Segmentation training and evaluation code from [ERFNet](https://github.com/Eromera/erfnet)
  * Performance valuation of latency speed code from [Fasterseg](https://github.com/VITA-Group/FasterSeg)
  * Cityscapes dataset is from [Cityscapes](https://www.cityscapes-dataset.com/submit/)
  
## License
This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License, which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: [http://creativecommons.org/licenses/by-nc/4.0/](http://creativecommons.org/licenses/by-nc/4.0/)
