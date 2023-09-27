
#  Characterization-of-PSD-of-coal-dust

This repository is mainly used to achieve instance segmentation of coal dust particles and perform relevant particle size distribution statistics.

![image](https://github.com/disciplinejzx/Characterization-of-PSD-of-coal-dust/blob/main/README_IMG.jpg))


## Environment construction

The strength segmentation network is based on BlendMask and improves it. Therefore you need to install and build Detectron2 and AdelaiDet. Reference: [AdelaiDet](https://github.com/aim-uofa/AdelaiDet ).

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

Then build AdelaiDet with:

```
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python setup.py build develop
```

## Coal Dust Dataset

Please wait for subsequent public uploads......

## Model improvement part program

Please replace the programs in the provided AdelaiDet and detectron2 folders with the corresponding directories under the installed AdelaiDet and detectron2 folders. The directory structure of the replacement file is as follows:

```
|-- AdelaiDet
	|-- adet
		|-- modeling
			|--backbone
				|--fpn.py
			|--blendmask
				|--basis_module.py
|--detectron2
	|--modeling
		|--backbone
			|--fpn.py
			|--resnet.py
```

## Train

This part needs to be improved after the coal dust data set is made public. Then you can refer to the [AdelaiDet](https://github.com/aim-uofa/AdelaiDet ) training process.  

## Model weight

download: 

[model](https://drive.google.com/file/d/1RAl21pdn3w4uC97RltD7Zxwx8QfVZMes/view?usp=sharing ) or [model](https://drive.google.com/uc?export=download&id=1RAl21pdn3w4uC97RltD7Zxwx8QfVZMes)

## Test

You can refer to [AdelaiDet](https://github.com/aim-uofa/AdelaiDet ).

```
python demo/demo.py \
    --config-file configs/BlendMask/R_101_3x.yaml \
    --input test_img/ \
    --confidence-threshold 0.35 \
    --opts MODEL.WEIGHTS model.pth
```

**Notice:**
The size of the test image is 3072*2048. You need to adjust the "C.INPUT.MIN_SIZE_TEST" and "C.INPUT.MAX_SIZE_TEST" parameters of the "detectron2/detectron2/config/defaults.py" program according to the video memory of the graphics card to run through the model and achieve segmentation of the test image.

There are a large number of particles in the test image, so you need to change the "_C.MODEL.FCOS.POST_NMS_TOPK_TEST" parameter of the "AdelaiDet/adet/config/defaults.py" program to set the upper limit of the number of predicted instances.

## Particle size distribution statistics

Please wait for subsequent public uploads......