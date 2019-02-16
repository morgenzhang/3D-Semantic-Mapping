# 3D-Semantic-Mapping
This is part of my semester project taken in Computer Vision &amp; Geometry Lab, ETH Zurich, Switzerland, aiming to create a pipeline for real-time 3D semantic mapping. The whole system is built on InfiniTAM V3 and new module: semantic fusion has been added. Currently support both CPU and GPU implementation.

InfiniTAM Semantic Mapping *** Yichen Zhang
*******************************************

Running Command
*******************************************
cd /path/to/your/workspace \
mkdir build \
cd build \
cmake /path/to/InfiniTAM \
./Apps/InfiniTAM/InfiniTAM /path/to/camera_calibration.txt /path/to/RGB/%04i.png /path/to/Depth/%04i.png /path/to/Semantic_Prob/%04i.png \
    +- libpng is required \
    +- I store the format of label probability(multiplied by 20000) as png image(uint16) \

Code Description (modification based on InfiniTAM v3) 
*******************************************
Utils/ITMLibSetting.cpp \
    +- Set sceneParams(0.3f, 10, 0.015f, 0.1f, 50.0f, false) \
    +- trackerConfig = "path/to/ground_truth_pose/%04i.txt" \


Utils/ITMMath.h \
    +- Add one more vector type: Vector12f, used for store prob(float) of 12 classes \
    +- Add one more vector type: Vector12us, used for store raw prob(uint16) of 12 classes \


Utils/ITMImageTypes.h
    +- Add one more image type: ITMUShort12Image, used for store input prob image


ITMLib/Scene/ITMVoxelTypes.h
    +- Add one more voxel type: ITMVoxel_s_semantic


ITMLib/Scene/ITMVoxelBlockHash.h
    +- Modify SDF_LOCAL_BLOCK_NUM      0x40000  -> 0x80000
    +- Modify SDF_BUCKET_NUM           0x100000 -> 0x200000
    +- Modify SDF_HASH_MASK            0xfffff  -> 0x1fffff
    +- Modify SDF_EXCESS_LIST_SIZE     0x20000  -> 0x40000
    +- Modify SDF_TRANSFER_BLOCK_NUM   0x1000   -> 0x2000


ITMLib/Scene/ITMRepresentationAccess.h
    +- Add semantic rendering utils (similar to RGB) using voxel information "sem"


ITMLib/Core/ITMMainEngine ITMBasicEngine ITMMultiEngine
    +- Add corresponding ITMUShort12Image IO


ITMLib/Views/ITMView.h
    +- Add corresponding ITMUShort12Image IO


ITMLib/Trackers/Interface/ITMFileBasedTracker
    +- Modify the pose input code


ITMLib/Engines/ViewBuilding/ITMViewBuilder_Shared
    +- Modify convertDisparityToDepth function according to KITTI camera parameters


ITMLib/Engines/Reconstruction/Shared/ITMSceneReconstructionEngine_Shared.h
    +- Add incremental semantic label fusion: computeUpdatedVoxelSemanticInfo
    +- Modify allocation function:buildHashAllocAndVisibleTypePP, discard sky


ITMLib/Engines/Reconstruction/CPU/ITMSceneReconstructionEngine_CPU.tpp
    +- Add CRF optimization (only CPU version)
    +- Add maximal prob decoding table, transfer prob to color


ITMLib/Engines/Visualisation/Shared/ITMVisualisationEngine_Shared.h
    +- Add semantic rendering function based on some utils in ITMRepresentationAccess.h


###### All the modules are modified in both CPU and GPU version except CRF optimization
###### which is currently commented for real-time application
###### Any questions regarding this code source can be sent to zhangyic@student.ethz.ch
