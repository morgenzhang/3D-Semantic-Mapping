// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "../../Utils/ITMMath.h"

/** \brief
    Stores the information of a single voxel in the volume
*/
struct ITMVoxel_f_rgb
{
	_CPU_AND_GPU_CODE_ static float SDF_initialValue() { return 1.0f; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return x; }
	_CPU_AND_GPU_CODE_ static float floatToValue(float x) { return x; }

	static const CONSTPTR(bool) hasColorInformation = true;
	static const CONSTPTR(bool) hasConfidenceInformation = false;
	static const CONSTPTR(bool) hasSemanticInformation = false;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** RGB colour information stored for this voxel. */
	Vector3u clr;
	/** Number of observations that made up @p clr. */
	uchar w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_f_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = Vector3u((uchar)0);
		w_color = 0;
	}
};

/** \brief
    Stores the information of a single voxel in the volume
*/
struct ITMVoxel_s_rgb
{
	_CPU_AND_GPU_CODE_ static short SDF_initialValue() { return 32767; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return (float)(x) / 32767.0f; }
	_CPU_AND_GPU_CODE_ static short floatToValue(float x) { return (short)((x) * 32767.0f); }

	static const CONSTPTR(bool) hasColorInformation = true;
	static const CONSTPTR(bool) hasConfidenceInformation = false;
	static const CONSTPTR(bool) hasSemanticInformation = false;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;
	/** RGB colour information stored for this voxel. */
	Vector3u clr;
	/** Number of observations that made up @p clr. */
	uchar w_color;

	_CPU_AND_GPU_CODE_ ITMVoxel_s_rgb()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		clr = Vector3u((uchar)0);
		w_color = 0;
	}
};

struct ITMVoxel_s
{
	_CPU_AND_GPU_CODE_ static short SDF_initialValue() { return 32767; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return (float)(x) / 32767.0f; }
	_CPU_AND_GPU_CODE_ static short floatToValue(float x) { return (short)((x) * 32767.0f); }

	static const CONSTPTR(bool) hasColorInformation = false;
	static const CONSTPTR(bool) hasConfidenceInformation = false;
	static const CONSTPTR(bool) hasSemanticInformation = false;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;

	_CPU_AND_GPU_CODE_ ITMVoxel_s()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
	}
};

struct ITMVoxel_f
{
	_CPU_AND_GPU_CODE_ static float SDF_initialValue() { return 1.0f; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return x; }
	_CPU_AND_GPU_CODE_ static float floatToValue(float x) { return x; }

	static const CONSTPTR(bool) hasColorInformation = false;
	static const CONSTPTR(bool) hasConfidenceInformation = false;
	static const CONSTPTR(bool) hasSemanticInformation = false;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;

	_CPU_AND_GPU_CODE_ ITMVoxel_f()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
	}
};

struct ITMVoxel_f_conf
{
	_CPU_AND_GPU_CODE_ static float SDF_initialValue() { return 1.0f; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return x; }
	_CPU_AND_GPU_CODE_ static float floatToValue(float x) { return x; }

	static const CONSTPTR(bool) hasColorInformation = false;
	static const CONSTPTR(bool) hasConfidenceInformation = true;
	static const CONSTPTR(bool) hasSemanticInformation = false;

	/** Value of the truncated signed distance transformation. */
	float sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;
	float confidence;

	_CPU_AND_GPU_CODE_ ITMVoxel_f_conf()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
		confidence = 0.0f;
	}
};

// Todo define another voxel type named ITMVoxel_s_semantic
/** step.1: currently input semantic image with assigned label value, imported
 *          just like RGB image
 *  step.2: input semantic image with assigned probability value, imported as
 *          16 channels object. do average over neighboured frames then obtain
 *          the label of corresponding frame                                  */
struct ITMVoxel_s_semantic
{
	_CPU_AND_GPU_CODE_ static short SDF_initialValue() { return 32767; }
	_CPU_AND_GPU_CODE_ static float valueToFloat(float x) { return (float)(x) / 32767.0f; }
	_CPU_AND_GPU_CODE_ static short floatToValue(float x) { return (short)((x) * 32767.0f); }

	static const CONSTPTR(bool) hasColorInformation = true;
	static const CONSTPTR(bool) hasConfidenceInformation = false;
	static const CONSTPTR(bool) hasSemanticInformation = true;

	/** Value of the truncated signed distance transformation. */
	short sdf;
	/** Number of fused observations that make up @p sdf. */
	uchar w_depth;
	/** Padding that may or may not improve performance on certain GPUs */
	//uchar pad;
    /** RGB colour information stored for this voxel. */
    Vector3u clr;
    /** Number of observations that made up @p clr. */
    uchar w_color;
	/** Semantic information (color after decoding) stored for this voxel. */
	Vector3u sem;
	/** Number of observations that made up @p sem. */
	uchar w_semantic;
	/** Semantic information (probabilities) stored for this voxel. */
	Vector12f prob;
	_CPU_AND_GPU_CODE_ ITMVoxel_s_semantic()
	{
		sdf = SDF_initialValue();
		w_depth = 0;
        clr = Vector3u((uchar)0);
        w_color = 0;
		sem = Vector3u((uchar)0);
		w_semantic = 0;
		prob = Vector12f((float)0);  // Todo: the voxel dimension should be custom-defined
	}
};