// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#include "ITMSceneReconstructionEngine_CPU.h"

#include "../Shared/ITMSceneReconstructionEngine_Shared.h"
#include "../../../Objects/RenderStates/ITMRenderState_VH.h"
#include "../../../CRF/densecrf.h"

using namespace ITMLib;

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ITMSceneReconstructionEngine_CPU(void) 
{
	int noTotalEntries = ITMVoxelBlockHash::noTotalEntries;
	entriesAllocType = new ORUtils::MemoryBlock<unsigned char>(noTotalEntries, MEMORYDEVICE_CPU);
	blockCoords = new ORUtils::MemoryBlock<Vector4s>(noTotalEntries, MEMORYDEVICE_CPU);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::~ITMSceneReconstructionEngine_CPU(void) 
{
	delete entriesAllocType;
	delete blockCoords;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMVoxelBlockHash>::ResetScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;

	ITMHashEntry tmpEntry;
	memset(&tmpEntry, 0, sizeof(ITMHashEntry));
	tmpEntry.ptr = -2;
	ITMHashEntry *hashEntry_ptr = scene->index.GetEntries();
	for (int i = 0; i < scene->index.noTotalEntries; ++i) hashEntry_ptr[i] = tmpEntry;
	int *excessList_ptr = scene->index.GetExcessAllocationList();
	for (int i = 0; i < SDF_EXCESS_LIST_SIZE; ++i) excessList_ptr[i] = i;

	scene->index.SetLastFreeExcessListId(SDF_EXCESS_LIST_SIZE - 1);
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::IntegrateIntoScene(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
	Vector2i semanticImgSize = view->rgb->noDims; // same as rgb
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb, M_semantic;
	Vector4f projParams_d, projParams_rgb, projParams_semantic;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;
    if (TVoxel::hasSemanticInformation) M_semantic = view->calib.trafo_rgb_to_depth.calib_inv * M_d; // same as rgb

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;
	projParams_semantic = projParams_rgb; // same as rgb

	float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	float *confidence = view->depthConfidence->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
	Vector12us *semantic = view->semantic->GetData(MEMORYDEVICE_CPU);
	TVoxel *localVBA = scene->localVBA.GetVoxelBlocks();
	ITMHashEntry *hashTable = scene->index.GetEntries();

	int *visibleEntryIds = renderState_vh->GetVisibleEntryIDs();
	int noVisibleEntries = renderState_vh->noVisibleEntries;

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int entryId = 0; entryId < noVisibleEntries; entryId++)
	{
		Vector3i globalPos;
		const ITMHashEntry &currentHashEntry = hashTable[visibleEntryIds[entryId]];
		if (currentHashEntry.ptr < 0) continue;

		globalPos.x = currentHashEntry.pos.x;
		globalPos.y = currentHashEntry.pos.y;
		globalPos.z = currentHashEntry.pos.z;
		globalPos *= SDF_BLOCK_SIZE;

		TVoxel *localVoxelBlock = &(localVBA[currentHashEntry.ptr * (SDF_BLOCK_SIZE3)]);

		for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
		{
			Vector4f pt_model; int locId;

			locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			if (stopIntegratingAtMaxW) if (localVoxelBlock[locId].w_depth == maxW) continue;
			//if (approximateIntegration) if (localVoxelBlock[locId].w_depth != 0) continue;

			pt_model.x = (float)(globalPos.x + x) * voxelSize;
			pt_model.y = (float)(globalPos.y + y) * voxelSize;
			pt_model.z = (float)(globalPos.z + z) * voxelSize;
			pt_model.w = 1.0f;

			ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation,TVoxel::hasConfidenceInformation, TVoxel::hasSemanticInformation, TVoxel>::compute(localVoxelBlock[locId], pt_model, M_d,
				projParams_d, M_rgb, projParams_rgb, M_semantic, projParams_semantic, mu, maxW, depth, confidence, depthImgSize, rgb, rgbImgSize, semantic, semanticImgSize);
		}

		// Todo: try CRF update probability inside local block (used for evaluation, not for real-time test yet)
// Comment out for CRF implementation
//		std::vector<int> valid_ids;
//		for (int i = 0; i < SDF_BLOCK_SIZE3; ++i) {
//			if (localVoxelBlock[i].w_semantic > 0) valid_ids.push_back(i);  // collect ids of voxels containing valid information
//		}
//
//		std::vector<float> unary_potentials(valid_ids.size() * 12);  // 12 as number of classes
//		for(int i = 0; i < static_cast<int>(valid_ids.size()); ++i) {
//			for (int j = 0; j < 12; ++j) {
//				int id = valid_ids[i];
//				// search for single voxel inside block
//				unary_potentials[i * 12 + j] = -log(localVoxelBlock[id].prob[j] + 1.0e-12);
//			}
//		}
//
//		float * block_data = new float [valid_ids.size() * 3];
//        for (int i = 0; i < static_cast<int>(valid_ids.size()); ++i) {
//        	int id = valid_ids[i];
        //// position features: still with dragons
//            int z = i / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
//            int y = (i - z * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
//            int x = i - y * SDF_BLOCK_SIZE - z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
//            block_data[i + 0*SDF_BLOCK_SIZE3] = x;
//            block_data[i + 1*SDF_BLOCK_SIZE3] = y;
//            block_data[i + 2*SDF_BLOCK_SIZE3] = z;
        //// color features
//            Vector3f color = TO_FLOAT3(localVoxelBlock[id].clr);
//            block_data[i + 0*valid_ids.size()] = color.r / 255.0f;
//            block_data[i + 1*valid_ids.size()] = color.g / 255.0f;
//            block_data[i + 2*valid_ids.size()] = color.b / 255.0f;
//        }

//		default settings in Kraehenbuehl and Koltun's public implementation
//		spatial_stddev = 0.05
//		colour_stddev = 20
//		normal_stddev = 0.1
//		DenseCRF3D crf(valid_ids.size(), 12, 0.05, 20, 0.1);  // 12 as number of classes
//        crf.setUnaryEnergy(unary_potentials.data());
//        // Add pairwise energies
//        crf.addPairwiseGaussian(0);
//        crf.addPairwiseBilateral(block_data, 50);
//        float* result_probs = crf.runInference(10, 1.0);
//        for (int i = 0; i < static_cast<int>(valid_ids.size()); ++i) {
//            for (int j = 0; j < 12; ++j) {
//            	const int id = valid_ids[i];
//                // Sometimes it returns nan resulting probs... filter these out
//                if (result_probs[i * 12 + j] > 0.0 && result_probs[i * 12 + j] < 1.0) {
//                    localVoxelBlock[id].prob[j] = result_probs[i * 12 + j];
//                }
//            }
//        }
//        delete [] block_data;

        // obtain final class id according to the max prob
        for (int z = 0; z < SDF_BLOCK_SIZE; z++) for (int y = 0; y < SDF_BLOCK_SIZE; y++) for (int x = 0; x < SDF_BLOCK_SIZE; x++)
        {
            int locId = x + y * SDF_BLOCK_SIZE + z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
            float max_prob = 0.0;
            int class_id = 11;
            for (int i = 0; i < 12; ++i) {
                if (localVoxelBlock[locId].prob[i] > max_prob) {
                    max_prob = localVoxelBlock[locId].prob[i];
                    class_id = i;
                }
            }
            Vector3u newC;
            // color render according to the kitti semantic dataset
            switch (class_id) {
                case 0:  newC.r = 128; newC.g = 64;  newC.b = 128; break;  // road
                case 1:  newC.r = 244; newC.g = 35;  newC.b = 232; break;  // sidewalk
                case 2:  newC.r = 70;  newC.g = 70;  newC.b = 70;  break;  // building
                case 3:  newC.r = 190; newC.g = 153; newC.b = 153; break;  // fence
                case 4:  newC.r = 153; newC.g = 153; newC.b = 153; break;  // pole
                case 5:  newC.r = 220; newC.g = 220; newC.b = 0;   break;  // traffic sign
                case 6:  newC.r = 107; newC.g = 142; newC.b = 35;  break;  // vegetation
                // for better visualization, discard part of sky left after allocation filter
                case 7:  newC.r = 70;  newC.g = 130; newC.b = 180;   break;  // sky
//                case 7:  newC.r = 0;  newC.g = 0; newC.b = 0;   break;  // sky
                case 8:  newC.r = 220; newC.g = 20;  newC.b = 60;  break;  // person
                case 9:  newC.r = 255; newC.g = 0;   newC.b = 0;   break;  // rider
                case 10: newC.r = 0;   newC.g = 0;   newC.b = 142; break;  // car
                case 11: newC.r = 0;   newC.g = 0;   newC.b = 0;   break;  // None
                default: newC.r = 0;   newC.g = 0;   newC.b = 0;   break;  // default as None
            }
            localVoxelBlock[locId].sem = newC;
        }
	}
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMVoxelBlockHash>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMVoxelBlockHash> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList, bool resetVisibleList)
{
	Vector2i depthImgSize = view->depth->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, invM_d;
	Vector4f projParams_d, invProjParams_d;

	ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*)renderState;
	if (resetVisibleList) renderState_vh->noVisibleEntries = 0;

	M_d = trackingState->pose_d->GetM(); M_d.inv(invM_d);

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	invProjParams_d = projParams_d;
	invProjParams_d.x = 1.0f / invProjParams_d.x;
	invProjParams_d.y = 1.0f / invProjParams_d.y;

	float mu = scene->sceneParams->mu;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector12us *semantic = view->semantic->GetData(MEMORYDEVICE_CPU);
	int *voxelAllocationList = scene->localVBA.GetAllocationList();
	int *excessAllocationList = scene->index.GetExcessAllocationList();
	ITMHashEntry *hashTable = scene->index.GetEntries();
	ITMHashSwapState *swapStates = scene->globalCache != NULL ? scene->globalCache->GetSwapStates(false) : 0;
	int *visibleEntryIDs = renderState_vh->GetVisibleEntryIDs();
	uchar *entriesVisibleType = renderState_vh->GetEntriesVisibleType();
	uchar *entriesAllocType = this->entriesAllocType->GetData(MEMORYDEVICE_CPU);
	Vector4s *blockCoords = this->blockCoords->GetData(MEMORYDEVICE_CPU);
	int noTotalEntries = scene->index.noTotalEntries;

	bool useSwapping = scene->globalCache != NULL;

	float oneOverVoxelSize = 1.0f / (voxelSize * SDF_BLOCK_SIZE);

	int lastFreeVoxelBlockId = scene->localVBA.lastFreeBlockId;
	int lastFreeExcessListId = scene->index.GetLastFreeExcessListId();

	int noVisibleEntries = 0;

	memset(entriesAllocType, 0, noTotalEntries);

	for (int i = 0; i < renderState_vh->noVisibleEntries; i++)
		entriesVisibleType[visibleEntryIDs[i]] = 3; // visible at previous frame and unstreamed

	//build hashVisibility
#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < depthImgSize.x*depthImgSize.y; locId++)
	{
		int y = locId / depthImgSize.x;
		int x = locId - y * depthImgSize.x;
		buildHashAllocAndVisibleTypePP(entriesAllocType, entriesVisibleType, x, y, blockCoords, depth, semantic, invM_d,
			invProjParams_d, mu, depthImgSize, oneOverVoxelSize, hashTable, scene->sceneParams->viewFrustum_min,
			scene->sceneParams->viewFrustum_max);
	}

	if (onlyUpdateVisibleList) useSwapping = false;
	if (!onlyUpdateVisibleList)
	{
		//allocate
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx, exlIdx;
			unsigned char hashChangeType = entriesAllocType[targetIdx];

			switch (hashChangeType)
			{
			case 1: //needs allocation, fits in the ordered list
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;

				if (vbaIdx >= 0) //there is room in the voxel block array
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;

					hashTable[targetIdx] = hashEntry;
				}
				else
				{
					// Mark entry as not visible since we couldn't allocate it but buildHashAllocAndVisibleTypePP changed its state.
					entriesVisibleType[targetIdx] = 0;

					// Restore previous value to avoid leaks.
					lastFreeVoxelBlockId++;
				}

				break;
			case 2: //needs allocation in the excess list
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;
				exlIdx = lastFreeExcessListId; lastFreeExcessListId--;

				if (vbaIdx >= 0 && exlIdx >= 0) //there is room in the voxel block array and excess list
				{
					Vector4s pt_block_all = blockCoords[targetIdx];

					ITMHashEntry hashEntry;
					hashEntry.pos.x = pt_block_all.x; hashEntry.pos.y = pt_block_all.y; hashEntry.pos.z = pt_block_all.z;
					hashEntry.ptr = voxelAllocationList[vbaIdx];
					hashEntry.offset = 0;

					int exlOffset = excessAllocationList[exlIdx];

					hashTable[targetIdx].offset = exlOffset + 1; //connect to child

					hashTable[SDF_BUCKET_NUM + exlOffset] = hashEntry; //add child to the excess list

					entriesVisibleType[SDF_BUCKET_NUM + exlOffset] = 1; //make child visible and in memory
				}
				else
				{
					// No need to mark the entry as not visible since buildHashAllocAndVisibleTypePP did not mark it.
					// Restore previous value to avoid leaks.
					lastFreeVoxelBlockId++;
					lastFreeExcessListId++;
				}

				break;
			}
		}
	}

	//build visible list
	for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
	{
		unsigned char hashVisibleType = entriesVisibleType[targetIdx];
		const ITMHashEntry &hashEntry = hashTable[targetIdx];
		
		if (hashVisibleType == 3)
		{
			bool isVisibleEnlarged, isVisible;

			if (useSwapping)
			{
				checkBlockVisibility<true>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
				if (!isVisibleEnlarged) hashVisibleType = 0;
			} else {
				checkBlockVisibility<false>(isVisible, isVisibleEnlarged, hashEntry.pos, M_d, projParams_d, voxelSize, depthImgSize);
				if (!isVisible) { hashVisibleType = 0; }
			}
			entriesVisibleType[targetIdx] = hashVisibleType;
		}

		if (useSwapping)
		{
			if (hashVisibleType > 0 && swapStates[targetIdx].state != 2) swapStates[targetIdx].state = 1;
		}

		if (hashVisibleType > 0)
		{	
			visibleEntryIDs[noVisibleEntries] = targetIdx;
			noVisibleEntries++;
		}

#if 0
		// "active list", currently disabled
		if (hashVisibleType == 1)
		{
			activeEntryIDs[noActiveEntries] = targetIdx;
			noActiveEntries++;
		}
#endif
	}

	//reallocate deleted ones from previous swap operation
	if (useSwapping)
	{
		for (int targetIdx = 0; targetIdx < noTotalEntries; targetIdx++)
		{
			int vbaIdx;
			ITMHashEntry hashEntry = hashTable[targetIdx];

			if (entriesVisibleType[targetIdx] > 0 && hashEntry.ptr == -1) 
			{
				vbaIdx = lastFreeVoxelBlockId; lastFreeVoxelBlockId--;
				if (vbaIdx >= 0) hashTable[targetIdx].ptr = voxelAllocationList[vbaIdx];
				else lastFreeVoxelBlockId++; // Avoid leaks
			}
		}
	}

	renderState_vh->noVisibleEntries = noVisibleEntries;

	scene->localVBA.lastFreeBlockId = lastFreeVoxelBlockId;
	scene->index.SetLastFreeExcessListId(lastFreeExcessListId);
}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::~ITMSceneReconstructionEngine_CPU(void) 
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel,ITMPlainVoxelArray>::ResetScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene)
{
	int numBlocks = scene->index.getNumAllocatedVoxelBlocks();
	int blockSize = scene->index.getVoxelBlockSize();

	TVoxel *voxelBlocks_ptr = scene->localVBA.GetVoxelBlocks();
	for (int i = 0; i < numBlocks * blockSize; ++i) voxelBlocks_ptr[i] = TVoxel();
	int *vbaAllocationList_ptr = scene->localVBA.GetAllocationList();
	for (int i = 0; i < numBlocks; ++i) vbaAllocationList_ptr[i] = i;
	scene->localVBA.lastFreeBlockId = numBlocks - 1;
}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::AllocateSceneFromDepth(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState, bool onlyUpdateVisibleList, bool resetVisibleList)
{}

template<class TVoxel>
void ITMSceneReconstructionEngine_CPU<TVoxel, ITMPlainVoxelArray>::IntegrateIntoScene(ITMScene<TVoxel, ITMPlainVoxelArray> *scene, const ITMView *view,
	const ITMTrackingState *trackingState, const ITMRenderState *renderState)
{
	Vector2i rgbImgSize = view->rgb->noDims;
	Vector2i depthImgSize = view->depth->noDims;
    Vector2i semanticImgSize = view->rgb->noDims;
	float voxelSize = scene->sceneParams->voxelSize;

	Matrix4f M_d, M_rgb, M_semantic;
	Vector4f projParams_d, projParams_rgb, projParams_semantic;

	M_d = trackingState->pose_d->GetM();
	if (TVoxel::hasColorInformation) M_rgb = view->calib.trafo_rgb_to_depth.calib_inv * M_d;
    if (TVoxel::hasSemanticInformation) M_semantic = view->calib.trafo_rgb_to_depth.calib_inv * M_d;

	projParams_d = view->calib.intrinsics_d.projectionParamsSimple.all;
	projParams_rgb = view->calib.intrinsics_rgb.projectionParamsSimple.all;
    projParams_semantic = projParams_rgb;

    float mu = scene->sceneParams->mu; int maxW = scene->sceneParams->maxW;

	float *depth = view->depth->GetData(MEMORYDEVICE_CPU);
	Vector4u *rgb = view->rgb->GetData(MEMORYDEVICE_CPU);
    Vector12us *semantic = view->semantic->GetData(MEMORYDEVICE_CPU);
	TVoxel *voxelArray = scene->localVBA.GetVoxelBlocks();

	const ITMPlainVoxelArray::IndexData *arrayInfo = scene->index.getIndexData();

	bool stopIntegratingAtMaxW = scene->sceneParams->stopIntegratingAtMaxW;
	//bool approximateIntegration = !trackingState->requiresFullRendering;

#ifdef WITH_OPENMP
	#pragma omp parallel for
#endif
	for (int locId = 0; locId < scene->index.getVolumeSize().x*scene->index.getVolumeSize().y*scene->index.getVolumeSize().z; ++locId)
	{
		int z = locId / (scene->index.getVolumeSize().x*scene->index.getVolumeSize().y);
		int tmp = locId - z * scene->index.getVolumeSize().x*scene->index.getVolumeSize().y;
		int y = tmp / scene->index.getVolumeSize().x;
		int x = tmp - y * scene->index.getVolumeSize().x;
		Vector4f pt_model;

		if (stopIntegratingAtMaxW) if (voxelArray[locId].w_depth == maxW) continue;
		//if (approximateIntegration) if (voxelArray[locId].w_depth != 0) continue;

		pt_model.x = (float)(x + arrayInfo->offset.x) * voxelSize;
		pt_model.y = (float)(y + arrayInfo->offset.y) * voxelSize;
		pt_model.z = (float)(z + arrayInfo->offset.z) * voxelSize;
		pt_model.w = 1.0f;

		ComputeUpdatedVoxelInfo<TVoxel::hasColorInformation, TVoxel::hasConfidenceInformation, TVoxel::hasSemanticInformation, TVoxel>::compute(voxelArray[locId],
		    pt_model, M_d, projParams_d, M_rgb, projParams_rgb, M_semantic, projParams_semantic, mu, maxW, depth, depthImgSize, rgb, rgbImgSize, semantic, semanticImgSize);
	}
}
