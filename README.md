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
In `registration.py` are all methods above available. Note, we use `subprocess` with anaconda environments. Therefore, if you want to use a certain method, you have to prepare the environment (all steps for certain methods are in the provided links).
Additionally, for texture features the dataset must contain images of the scene with the resolution width x height = number of data points in point cloud. Also it must be ordered.\
To use Superglue, please download their repository and place it into to the main folder. By using Superglue you need to adhere to their license.


# Metrics
We evaluate the following methods with the metrics obtained from [link](https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf): \
**Translation error** $e_{t} (t_{gt}, t_{p}) =  \lVert t_{gt} - t_{p} \rVert_{2} \ [cm]$ \
**Rotation error** $e_{R} (R_{gt}, R_{p}) = \arccos{\frac{trace(R_{p}R_{gt}^{-1}) -1}{2}} \ [rad]$ \
**Time** of the registration process in the seconds.
