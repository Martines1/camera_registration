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
**Translation error** $e_{t} (t_{gt}, t_{p}) =  \lVert t_{gt} - t_{p} \rVert_{2} \ [m]$ \
\
**Rotation error** $e_{R} (R_{gt}, R_{p}) = \arccos{\frac{trace(R_{p}R_{gt}^{-1}) -1}{2}} \ [deeg]$ \
\
**Time** of the registration process in the seconds.

# Demo
The demo is prepared in the `registration.py`. Run the file with activated environment `cam_registration`
```
if  __name__ == '__main__':
    #demo test showcase
    t = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    print("GEOMETRIC TEST")
    t.demo_transformation_geometry()
    for geo_method in [t.cpd_register(), t.geotransformer_register(), t.teaser_register("FPFH"), t.gcnet_register(), t.mac_register("FPFH"), t.pointdsc_register("FPFH")]:
        geo_method
    print("SUPERGLUE TEST")
    t1 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t2 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t3 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t1.demo_transformation_texture('Superglue')
    t2.demo_transformation_texture('Superglue')
    t3.demo_transformation_texture('Superglue')
    
    for superglue_method in [t1.teaser_register("Superglue"), t2.mac_register("Superglue"), t3.pointdsc_register("Superglue")]:
        superglue_method
    print("PDC-NET+ TEST")
    t1 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t2 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t3 = Register("demo/0.ply", "demo/1.ply", gt=None, image_source="demo/0.png", image_target="demo/1.png")
    t1.demo_transformation_texture('PDC-Net+')
    t2.demo_transformation_texture('PDC-Net+')
    t3.demo_transformation_texture('PDC-Net+')
    for pdcnet_method in [t1.teaser_register("PDC-Net+"), t2.mac_register("PDC-Net+"), t3.pointdsc_register("PDC-Net+")]:
        pdcnet_method
```
### Expected output (ignoring Teaser++ process outputs)
```
GEOMETRIC TEST
CPD registration done! Total time: 12.63 sec.
Rotation error: 6.44, translation error: 0.26
GeoTransformer registration done! Total time: 1.93 sec.
Rotation error: 1.22, translation error: 0.03
Teaser++ registration done! Total time: 0.01 sec.
Rotation error: 0.43, translation error: 0.03
GCNet registration done! Total time: 2.96 sec.
Rotation error: 0.14, translation error: 0.02
MAC registration done! Total time: 1.22 sec.
Rotation error: 0.13, translation error: 0.02
PointDSC registration done! Total time: 0.47 sec.
Rotation error: 1.32, translation error: 0.03
SUPERGLUE TEST
Teaser++ registration done! Total time: 0.0 sec.
Rotation error: 0.11, translation error: 0.01
MAC registration done! Total time: 98.95 sec.
Rotation error: 0.02, translation error: 0.0
PointDSC registration done! Total time: 0.12 sec.
Rotation error: 0.01, translation error: 0.0
PDC-NET+ TEST
Teaser++ registration done! Total time: 0.02 sec.
Rotation error: 0.04, translation error: 0.0
MAC registration done! Total time: 1.01 sec.
Rotation error: 0.0, translation error: 0.0
PointDSC registration done! Total time: 0.13 sec.
Rotation error: 0.06, translation error: 0.0
```
#### The purpose of the demo is to test if everything was installed correctly.
# Future work
* ##### Add support to Windows OS (if possible)
* ##### Visualize correspondences
* ##### Compare geometric and texture features for one method and chose better.
