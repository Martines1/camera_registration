# Camera registration
This is offical repository (only for academic use) of the bachelor thesis "Markerless Registration of Multiple 3D cameras". \
We evaluate camera (point-set) registration methods, which are: 

* Coherent Point Drift ([link](https://github.com/neka-nat/probreg))
* GeoTransformer ([link](https://github.com/qinzheng93/GeoTransformer))
* Teaser++ ([link](https://github.com/MIT-SPARK/TEASER-plusplus))
* GCNet ([link](https://github.com/zhulf0804/GCNet))
* MAC ([link](https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques))
* PointDSC ([link](https://github.com/XuyangBai/PointDSC))
***
For **geometric** features, we use:
* FPFH ([link](https://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf))
***
For **texture** features, we use:
* Superglue ([link](https://github.com/magicleap/SuperGluePretrainedNetwork))
* PDC-Net+ ([link](https://github.com/PruneTruong/DenseMatching))
***
This repository was tested on the following consumer-grade workstation
* OS: UBUNTU 20.04.4
* CPU: AMD Ryzen 7 5700X
* GPU: RTX 3060 12GB
* RAM: 32GB DDR4 3600 Mhz
***
### Install (from one terminal)
* Clone this repository
```
git clone https://github.com/Martines1/camera_registration.git
cd camera_registration
```
* Create conda environment
```
conda env create -f environment.yml
```
Note that we use PyTorch ([link](https://pytorch.org/)), and this repository was tested on the RTX 3060 12GB. Therefore, problems with the CUDA version can occur. If problems occur, please install PyTorch with the compatible version for your GPU.
* Install GCNet environment
```
conda create -n gcnet python=3.8.18
conda activate gcnet
cd models_dir
cd GCNet
pip install -r requirements.txt
cd cpp_wrappers
sh compile_wrappers.sh
conda deactivate
```

* Compile GeoTransformer, GCNet, and Teaser++
```
cd ../..
conda activate cam_registration
cd GeoTransformer
python setup.py build develop
cd ..
sudo apt install cmake libeigen3-dev libboost-all-dev
cd TEASER && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.6 .. && make teaserpp_python
cd python && pip install .
cd ..
```
* Install and prepare PDC-Net+
```
cd ../..
cd pdc_net
conda env create -f environment.yml
conda activate dense_matching_env
bash assets/download_pre_trained_models.sh
```
***
In `registration.py` are all the point cloud registration methods above available. Note, we use `subprocess` for GCNet, and PDC-Net+.
Additionally, for texture features the dataset must contain images of the scene with the resolution width x height = number of data points in point cloud. Also it must be ordered.
***
### Superglue support
To use Superglue, please download their repository and place it into to the main folder in this repository and install the environment according to the instructions on their official repository website. By using Superglue you need to adhere to their license.\
Then run the following commands from the main location of this repository
```
cd SuperGluePretrainedNetwork
./match_pairs.py --input_dir ../images --input_pairs ../superglue_pairs.txt --resize -1 --output_dir ../output_pairs
````



# Metrics
We evaluate the following methods with the metrics obtained from [link](https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf): \
<div style="text-align:center; font-size: 20px; line-height: 2;">
<strong>Translation error</strong> \(e_{t} (t_{gt}, t_{p}) =  \lVert t_{gt} - t_{p} \rVert_{2} \ [cm]\) <br>
<strong>Rotation error</strong> \(e_{R} (R_{gt}, R_{p}) = \arccos{\frac{trace(R_{p}R_{gt}^{-1}) -1}{2}} \ [rad]\) <br>
<strong>Time</strong> of the registration process in the seconds.
</div>
