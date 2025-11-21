add_library(ikDataProcess SHARED
    DataProcess.hpp
    DataProcessGlobal.hpp
    DataProcessTools.hpp
    pch.hpp
    Core/CPluginProcessInterface.hpp
    Core/CProcessRegistration.cpp
    Core/CProcessRegistration.h
    Core/CIkomiaRegistry.cpp
    Core/CIkomiaRegistry.h
    Core/CPluginTools.cpp
    Core/CPluginTools.h
    Core/CRunTaskManager.cpp
    Core/CRunTaskManager.h
    Core/CTaskFactory.hpp
    Core/CTaskParamFactory.hpp
    Core/CWidgetFactory.hpp
    Core/CWorkflow.cpp
    Core/CWorkflow.h
    Core/CHardwareConfig.h
    Core/CHardwareConfig.cpp
    IO/CArrayIO.cpp
    IO/CArrayIO.h
    IO/CBlobMeasureIO.cpp
    IO/CBlobMeasureIO.h
    IO/CConvertIO.cpp
    IO/CConvertIO.h
    IO/CDatasetIO.cpp
    IO/CDatasetIO.h
    IO/CGraphicsInput.cpp
    IO/CGraphicsInput.h
    IO/CGraphicsOutput.cpp
    IO/CGraphicsOutput.h
    IO/CImageIO.cpp
    IO/CImageIO.h
    IO/CInstanceSegIO.cpp
    IO/CInstanceSegIO.h
    IO/CJsonIO.cpp
    IO/CJsonIO.h
    IO/CKeypointsIO.cpp
    IO/CKeypointsIO.h
    IO/CNumericIO.cpp
    IO/CNumericIO.h
    IO/CObjectDetectionIO.cpp
    IO/CObjectDetectionIO.h
    IO/CPathIO.cpp
    IO/CPathIO.h
    IO/CScene3dIO.cpp
    IO/CScene3dIO.h
    IO/CSemanticSegIO.cpp
    IO/CSemanticSegIO.h
    IO/CTaskIORegistration.cpp
    IO/CTaskIORegistration.h
    IO/CTextIO.cpp
    IO/CTextIO.h
    IO/CVideoIO.cpp
    IO/CVideoIO.h
    IO/CWidgetOutput.cpp
    IO/CWidgetOutput.h
    IO/Scene3d/CScene3d.cpp
    IO/Scene3d/CScene3d.h
    IO/Scene3d/CScene3dColor.cpp
    IO/Scene3d/CScene3dColor.h
    IO/Scene3d/CScene3dCoord.cpp
    IO/Scene3d/CScene3dCoord.h
    IO/Scene3d/CScene3dImage2d.cpp
    IO/Scene3d/CScene3dImage2d.h
    IO/Scene3d/CScene3dLayer.cpp
    IO/Scene3d/CScene3dLayer.h
    IO/Scene3d/CScene3dObject.cpp
    IO/Scene3d/CScene3dObject.h
    IO/Scene3d/CScene3dShapeCircle.cpp
    IO/Scene3d/CScene3dShapeCircle.h
    IO/Scene3d/CScene3dShapePoint.cpp
    IO/Scene3d/CScene3dShapePoint.h
    IO/Scene3d/CScene3dShapePoly.cpp
    IO/Scene3d/CScene3dShapePoly.h
    IO/Scene3d/CScene3dVectorField.cpp
    IO/Scene3d/CScene3dVectorField.h
    Process/Core/CBinaryToGraphics.cpp
    Process/Core/CBinaryToGraphics.h
    Process/Core/CBlobMeasure.cpp
    Process/Core/CBlobMeasure.h
    Process/Core/CCut.hpp
    Process/Core/CFillHoles.hpp
    Process/Core/CGraphicsToBinary.cpp
    Process/Core/CGraphicsToBinary.h
    Process/Core/CInstanceSegFilter.cpp
    Process/Core/CInstanceSegFilter.h
    Process/Core/CObjDetectFilter.cpp
    Process/Core/CObjDetectFilter.h
    Process/Core/CPlotMerge.hpp
    Process/Core/CRgbHlsThreshold.cpp
    Process/Core/CRgbHlsThreshold.h
    Process/Core/CSemanticSegFilter.cpp
    Process/Core/CSemanticSegFilter.h
    Process/Core/ProcessCore.hpp
    Process/Gmic/CGmicImageConverter.cpp
    Process/Gmic/CGmicImageConverter.h
    Process/Gmic/CGmicTaskParam.hpp
    Process/Gmic/Colors/CGmicAutoBalance.hpp
    Process/Gmic/Colors/CGmicBoostChroma.hpp
    Process/Gmic/Colors/CGmicBoostFade.hpp
    Process/Gmic/Colors/CGmicColorPresets.hpp
    Process/Gmic/Contours/CGmicDOG.hpp
    Process/Gmic/Contours/CGmicDistanceTransform.hpp
    Process/Gmic/Contours/CGmicSkeleton.hpp
    Process/Gmic/Contours/CGmicSuperPixels.hpp
    Process/Gmic/Details/CGmicConstrainedSharpen.hpp
    Process/Gmic/Details/CGmicDynamicRangeIncrease.hpp
    Process/Gmic/Details/CGmicMagicDetails.hpp
    Process/Gmic/Details/CGmicSharpenDeblur.hpp
    Process/Gmic/Details/CGmicSharpenGradient.hpp
    Process/Gmic/Details/CGmicSharpenLucy.hpp
    Process/Gmic/Details/CGmicSharpenTones.hpp
    Process/Gmic/ProcessGmic.hpp
    Process/Gmic/Repair/CGmicHotPixels.hpp
    Process/Gmic/Repair/CGmicInpaint.hpp
    Process/Gmic/Repair/CGmicRedEye.hpp
    Process/OpenCV/COcvHfsSegment.hpp
    Process/OpenCV/COcvInpaintFuzzy.hpp
    Process/OpenCV/ProcessOcv.hpp
    Process/OpenCV/bgsegm/COcvBckgndSubCnt.hpp
    Process/OpenCV/bgsegm/COcvBckgndSubGmg.hpp
    Process/OpenCV/bgsegm/COcvBckgndSubGsoc.hpp
    Process/OpenCV/bgsegm/COcvBckgndSubLsbp.hpp
    Process/OpenCV/bgsegm/COcvBckgndSubMog.hpp
    Process/OpenCV/bioinspired/COcvRetina.hpp
    Process/OpenCV/bioinspired/COcvRetinaSegmentation.hpp
    Process/OpenCV/bioinspired/COcvRetinaToneMapping.hpp
    Process/OpenCV/core/COcvAbsdiff.hpp
    Process/OpenCV/core/COcvAdd.hpp
    Process/OpenCV/core/COcvAddWeighted.hpp
    Process/OpenCV/core/COcvCompare.hpp
    Process/OpenCV/core/COcvConvertTo.hpp
    Process/OpenCV/core/COcvCopyMakeBorder.hpp
    Process/OpenCV/core/COcvCountNonZero.hpp
    Process/OpenCV/core/COcvCrop.hpp
    Process/OpenCV/core/COcvDft.hpp
    Process/OpenCV/core/COcvDftInv.hpp
    Process/OpenCV/core/COcvDivide.hpp
    Process/OpenCV/core/COcvExp.hpp
    Process/OpenCV/core/COcvExtractChannel.hpp
    Process/OpenCV/core/COcvFlip.hpp
    Process/OpenCV/core/COcvInRange.hpp
    Process/OpenCV/core/COcvInsertChannel.hpp
    Process/OpenCV/core/COcvKMeans.hpp
    Process/OpenCV/core/COcvLog.hpp
    Process/OpenCV/core/COcvLogicalOp.hpp
    Process/OpenCV/core/COcvMagnitude.hpp
    Process/OpenCV/core/COcvMax.hpp
    Process/OpenCV/core/COcvMerge.hpp
    Process/OpenCV/core/COcvMin.hpp
    Process/OpenCV/core/COcvMultiply.hpp
    Process/OpenCV/core/COcvNegative.hpp
    Process/OpenCV/core/COcvNormalize.hpp
    Process/OpenCV/core/COcvPSNR.hpp
    Process/OpenCV/core/COcvRotate.hpp
    Process/OpenCV/core/COcvSplit.hpp
    Process/OpenCV/core/COcvSubtract.hpp
    Process/OpenCV/dnn/COcvDnnClassifier.hpp
    Process/OpenCV/dnn/COcvDnnColorization.hpp
    Process/OpenCV/dnn/COcvDnnDetector.hpp
    Process/OpenCV/dnn/COcvDnnProcess.cpp
    Process/OpenCV/dnn/COcvDnnProcess.h
    Process/OpenCV/dnn/COcvDnnInstanceSeg.cpp
    Process/OpenCV/dnn/COcvDnnInstanceSeg.h
    Process/OpenCV/dnn/COcvDnnSemanticSeg.cpp
    Process/OpenCV/dnn/COcvDnnSemanticSeg.h
    Process/OpenCV/features2d/COcvAGAST.hpp
    Process/OpenCV/features2d/COcvAKAZE.hpp
    Process/OpenCV/features2d/COcvBFMatcher.hpp
    Process/OpenCV/features2d/COcvBRISK.hpp
    Process/OpenCV/features2d/COcvFAST.hpp
    Process/OpenCV/features2d/COcvFlannMatcher.hpp
    Process/OpenCV/features2d/COcvGFTT.hpp
    Process/OpenCV/features2d/COcvKAZE.hpp
    Process/OpenCV/features2d/COcvORB.hpp
    Process/OpenCV/features2d/COcvSIFT.hpp
    Process/OpenCV/features2d/COcvSimpleBlobDetector.hpp
    Process/OpenCV/imgproc/COcvAdaptiveThreshold.hpp
    Process/OpenCV/imgproc/COcvBilateral.hpp
    Process/OpenCV/imgproc/COcvBlur.hpp
    Process/OpenCV/imgproc/COcvBoxFilter.hpp
    Process/OpenCV/imgproc/COcvCLAHE.hpp
    Process/OpenCV/imgproc/COcvCalcHist.hpp
    Process/OpenCV/imgproc/COcvCanny.hpp
    Process/OpenCV/imgproc/COcvColorMap.hpp
    Process/OpenCV/imgproc/COcvCvtColor.hpp
    Process/OpenCV/imgproc/COcvDistanceTransform.hpp
    Process/OpenCV/imgproc/COcvEqualizeHist.hpp
    Process/OpenCV/imgproc/COcvGaussianBlur.hpp
    Process/OpenCV/imgproc/COcvGrabCut.hpp
    Process/OpenCV/imgproc/COcvHoughCircles.hpp
    Process/OpenCV/imgproc/COcvHoughLines.hpp
    Process/OpenCV/imgproc/COcvLaplacian.hpp
    Process/OpenCV/imgproc/COcvMedian.hpp
    Process/OpenCV/imgproc/COcvMorphologyEx.hpp
    Process/OpenCV/imgproc/COcvResize.hpp
    Process/OpenCV/imgproc/COcvRotateEx.hpp
    Process/OpenCV/imgproc/COcvSobel.hpp
    Process/OpenCV/imgproc/COcvStackBlur.hpp
    Process/OpenCV/imgproc/COcvThreshold.hpp
    Process/OpenCV/imgproc/COcvWatershed.hpp
    Process/OpenCV/objdetect/COcvCascadeClassifier.hpp
    Process/OpenCV/objdetect/COcvQRCodeDetector.hpp
    Process/OpenCV/optflow/COcvDeepFlow.hpp
    Process/OpenCV/optflow/COcvDualTVL1OpticalFlow.hpp
    Process/OpenCV/optflow/COcvPCAOpticalFlow.hpp
    Process/OpenCV/optflow/COcvSimpleFlow.hpp
    Process/OpenCV/optflow/COcvSparseToDenseOF.hpp
    Process/OpenCV/photo/COcvColorchange.hpp
    Process/OpenCV/photo/COcvDecolor.hpp
    Process/OpenCV/photo/COcvDenoiseTVL1.hpp
    Process/OpenCV/photo/COcvDetailEnhance.hpp
    Process/OpenCV/photo/COcvEdgePreservingFilter.hpp
    Process/OpenCV/photo/COcvFastNlMeans.hpp
    Process/OpenCV/photo/COcvFastNlMeansMulti.hpp
    Process/OpenCV/photo/COcvIlluminationChange.hpp
    Process/OpenCV/photo/COcvInpaint.hpp
    Process/OpenCV/photo/COcvPencilSketch.hpp
    Process/OpenCV/photo/COcvSeamlessCloning.hpp
    Process/OpenCV/photo/COcvStylization.hpp
    Process/OpenCV/photo/COcvTextureFlattening.hpp
    Process/OpenCV/saliency/COcvMotionSaliencyBinWangApr2014.hpp
    Process/OpenCV/saliency/COcvObjectnessBING.hpp
    Process/OpenCV/saliency/COcvSaliencyFineGrained.hpp
    Process/OpenCV/saliency/COcvSaliencySpectralResidual.hpp
    Process/OpenCV/superres/COcvSuperResBTVL1.cpp
    Process/OpenCV/superres/COcvSuperResBTVL1.h
    Process/OpenCV/superres/CSuperResFrameSrc.cpp
    Process/OpenCV/superres/CSuperResFrameSrc.h
    Process/OpenCV/text/COcvOCRTesseract.hpp
    Process/OpenCV/tracking/COcvTrackerGOTURN.hpp
    Process/OpenCV/tracking/COcvTrackerKCF.hpp
    Process/OpenCV/video/COcvBckgndSubKnn.hpp
    Process/OpenCV/video/COcvBckgndSubMog2.hpp
    Process/OpenCV/video/COcvCamShift.hpp
    Process/OpenCV/video/COcvDISOpticalFlow.hpp
    Process/OpenCV/video/COcvFarnebackOpticalFlow.hpp
    Process/OpenCV/video/COcvMeanShift.hpp
    Process/OpenCV/ximgproc/COcvAdaptiveManifold.hpp
    Process/OpenCV/ximgproc/COcvAnisotropicDiffusion.hpp
    Process/OpenCV/ximgproc/COcvBilateralTextureFilter.hpp
    Process/OpenCV/ximgproc/COcvDTFilter.hpp
    Process/OpenCV/ximgproc/COcvDTFilterEnhance.hpp
    Process/OpenCV/ximgproc/COcvDTFilterStylize.hpp
    Process/OpenCV/ximgproc/COcvFastGlobalSmootherFilter.hpp
    Process/OpenCV/ximgproc/COcvFastLineDetector.hpp
    Process/OpenCV/ximgproc/COcvGradientDeriche.hpp
    Process/OpenCV/ximgproc/COcvGradientPaillou.hpp
    Process/OpenCV/ximgproc/COcvGraphSegmentation.hpp
    Process/OpenCV/ximgproc/COcvGuidedFilter.hpp
    Process/OpenCV/ximgproc/COcvJointBilateralFilter.hpp
    Process/OpenCV/ximgproc/COcvL0Smooth.hpp
    Process/OpenCV/ximgproc/COcvNiblackThreshold.hpp
    Process/OpenCV/ximgproc/COcvPeiLinNormalization.hpp
    Process/OpenCV/ximgproc/COcvRidgeFilter.hpp
    Process/OpenCV/ximgproc/COcvRollingGuidanceFilter.hpp
    Process/OpenCV/ximgproc/COcvSelectiveSearchSegmentation.hpp
    Process/OpenCV/ximgproc/COcvStructuredEdgeDetection.hpp
    Process/OpenCV/ximgproc/COcvSuperpixelLSC.hpp
    Process/OpenCV/ximgproc/COcvSuperpixelSEEDS.hpp
    Process/OpenCV/ximgproc/COcvSuperpixelSLIC.hpp
    Process/OpenCV/ximgproc/COcvThinning.hpp
    Process/OpenCV/xphoto/COcvGrayworldWB.hpp
    Process/OpenCV/xphoto/COcvInpaintX.hpp
    Process/OpenCV/xphoto/COcvLearningBasedWB.hpp
    Process/OpenCV/xphoto/COcvSimpleWB.hpp
    Process/OpenCV/xphoto/COcvTonemapDurand.hpp
    Task/C2dFeatureImageTask.cpp
    Task/C2dFeatureImageTask.h
    Task/C2dFeatureMatcherTask.cpp
    Task/C2dFeatureMatcherTask.h
    Task/C2dImageInteractiveTask.cpp
    Task/C2dImageInteractiveTask.h
    Task/C2dImageTask.cpp
    Task/C2dImageTask.h
    Task/CClassificationTask.cpp
    Task/CClassificationTask.h
    Task/CDnnTrainTask.cpp
    Task/CDnnTrainTask.h
    Task/CInstanceSegTask.cpp
    Task/CInstanceSegTask.h
    Task/CKeypointDetectionTask.cpp
    Task/CKeypointDetectionTask.h
    Task/CMlflowTrainTask.cpp
    Task/CMlflowTrainTask.h
    Task/CObjectDetectionTask.cpp
    Task/CObjectDetectionTask.h
    Task/CSemanticSegTask.cpp
    Task/CSemanticSegTask.h
    Task/CTaskInfo.cpp
    Task/CTaskInfo.h
    Task/CVideoOFTask.cpp
    Task/CVideoOFTask.h
    Task/CVideoTask.cpp
    Task/CVideoTask.h
    Task/CVideoTrackingTask.cpp
    Task/CVideoTrackingTask.h
    Widget/Core/CWidgetApplyOnly.hpp
    Widget/Core/CWidgetBinaryToGraphics.hpp
    Widget/Core/CWidgetBlobMeasure.cpp
    Widget/Core/CWidgetBlobMeasure.h
    Widget/Core/CWidgetCut.hpp
    Widget/Core/CWidgetFillHoles.hpp
    Widget/Core/CWidgetGraphicsToBinary.hpp
    Widget/Core/CWidgetPlotMerge.hpp
    Widget/Core/CWidgetRgbHlsThreshold.cpp
    Widget/Core/CWidgetRgbHlsThreshold.h
    Widget/Core/WidgetCore.hpp
    Widget/Gmic/Colors/CGmicWidgetAutoBalance.hpp
    Widget/Gmic/Colors/CGmicWidgetBoostChroma.hpp
    Widget/Gmic/Colors/CGmicWidgetBoostFade.hpp
    Widget/Gmic/Colors/CGmicWidgetColorPresets.hpp
    Widget/Gmic/Contours/CGmicWidgetDOG.hpp
    Widget/Gmic/Contours/CGmicWidgetDistanceTransform.hpp
    Widget/Gmic/Contours/CGmicWidgetSkeleton.hpp
    Widget/Gmic/Contours/CGmicWidgetSuperPixels.hpp
    Widget/Gmic/Details/CGmicWidgetConstrainedSharpen.hpp
    Widget/Gmic/Details/CGmicWidgetDynamicRangeIncrease.hpp
    Widget/Gmic/Details/CGmicWidgetMagicDetails.hpp
    Widget/Gmic/Details/CGmicWidgetSharpenDeblur.hpp
    Widget/Gmic/Details/CGmicWidgetSharpenGradient.hpp
    Widget/Gmic/Details/CGmicWidgetSharpenLucy.hpp
    Widget/Gmic/Details/CGmicWidgetSharpenTones.hpp
    Widget/Gmic/Repair/CGmicWidgetHotPixels.hpp
    Widget/Gmic/Repair/CGmicWidgetInpaint.hpp
    Widget/Gmic/Repair/CGmicWidgetRedEye.hpp
    Widget/Gmic/WidgetGmic.hpp
    Widget/OpenCV/COcvWidgetHfsSegment.hpp
    Widget/OpenCV/COcvWidgetInpaintFuzzy.hpp
    Widget/OpenCV/WidgetOcv.hpp
    Widget/OpenCV/bgsegm/COcvWidgetBckgndSubCnt.hpp
    Widget/OpenCV/bgsegm/COcvWidgetBckgndSubGmg.hpp
    Widget/OpenCV/bgsegm/COcvWidgetBckgndSubGsoc.hpp
    Widget/OpenCV/bgsegm/COcvWidgetBckgndSubLsbp.hpp
    Widget/OpenCV/bgsegm/COcvWidgetBckgndSubMog.hpp
    Widget/OpenCV/bioinspired/COcvWidgetRetina.hpp
    Widget/OpenCV/bioinspired/COcvWidgetRetinaSegmentation.hpp
    Widget/OpenCV/bioinspired/COcvWidgetRetinaToneMapping.hpp
    Widget/OpenCV/core/COcvWidgetAbsdiff.hpp
    Widget/OpenCV/core/COcvWidgetAdd.hpp
    Widget/OpenCV/core/COcvWidgetAddWeighted.hpp
    Widget/OpenCV/core/COcvWidgetCompare.hpp
    Widget/OpenCV/core/COcvWidgetConvertTo.hpp
    Widget/OpenCV/core/COcvWidgetCopyMakeBorder.hpp
    Widget/OpenCV/core/COcvWidgetCountNonZero.hpp
    Widget/OpenCV/core/COcvWidgetCrop.hpp
    Widget/OpenCV/core/COcvWidgetDft.hpp
    Widget/OpenCV/core/COcvWidgetDftInv.hpp
    Widget/OpenCV/core/COcvWidgetDivide.hpp
    Widget/OpenCV/core/COcvWidgetExp.hpp
    Widget/OpenCV/core/COcvWidgetExtractChannel.hpp
    Widget/OpenCV/core/COcvWidgetFlip.hpp
    Widget/OpenCV/core/COcvWidgetInRange.hpp
    Widget/OpenCV/core/COcvWidgetInsertChannel.hpp
    Widget/OpenCV/core/COcvWidgetKMeans.hpp
    Widget/OpenCV/core/COcvWidgetLog.hpp
    Widget/OpenCV/core/COcvWidgetLogicalOp.hpp
    Widget/OpenCV/core/COcvWidgetMagnitude.hpp
    Widget/OpenCV/core/COcvWidgetMax.hpp
    Widget/OpenCV/core/COcvWidgetMerge.hpp
    Widget/OpenCV/core/COcvWidgetMin.hpp
    Widget/OpenCV/core/COcvWidgetMultiply.hpp
    Widget/OpenCV/core/COcvWidgetNegative.hpp
    Widget/OpenCV/core/COcvWidgetNormalize.hpp
    Widget/OpenCV/core/COcvWidgetPSNR.hpp
    Widget/OpenCV/core/COcvWidgetRotate.hpp
    Widget/OpenCV/core/COcvWidgetSplit.hpp
    Widget/OpenCV/core/COcvWidgetSubtract.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnClassifier.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnColorization.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnCore.cpp
    Widget/OpenCV/dnn/COcvWidgetDnnCore.h
    Widget/OpenCV/dnn/COcvWidgetDnnDetector.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnProcess.cpp
    Widget/OpenCV/dnn/COcvWidgetDnnProcess.h
    Widget/OpenCV/dnn/COcvWidgetDnnInstanceSeg.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnSemanticSeg.hpp
    Widget/OpenCV/features2d/COcvWidgetAGAST.hpp
    Widget/OpenCV/features2d/COcvWidgetAKAZE.hpp
    Widget/OpenCV/features2d/COcvWidgetBFMatcher.hpp
    Widget/OpenCV/features2d/COcvWidgetBRISK.hpp
    Widget/OpenCV/features2d/COcvWidgetFAST.hpp
    Widget/OpenCV/features2d/COcvWidgetFlannMatcher.hpp
    Widget/OpenCV/features2d/COcvWidgetGFTT.hpp
    Widget/OpenCV/features2d/COcvWidgetKAZE.hpp
    Widget/OpenCV/features2d/COcvWidgetORB.hpp
    Widget/OpenCV/features2d/COcvWidgetSIFT.hpp
    Widget/OpenCV/features2d/COcvWidgetSimpleBlobDetector.hpp
    Widget/OpenCV/imgproc/COcvWidgetAdaptiveThreshold.hpp
    Widget/OpenCV/imgproc/COcvWidgetBilateral.hpp
    Widget/OpenCV/imgproc/COcvWidgetBlur.hpp
    Widget/OpenCV/imgproc/COcvWidgetBoxFilter.hpp
    Widget/OpenCV/imgproc/COcvWidgetCLAHE.hpp
    Widget/OpenCV/imgproc/COcvWidgetCalcHist.hpp
    Widget/OpenCV/imgproc/COcvWidgetCanny.hpp
    Widget/OpenCV/imgproc/COcvWidgetColorMap.hpp
    Widget/OpenCV/imgproc/COcvWidgetCvtColor.hpp
    Widget/OpenCV/imgproc/COcvWidgetDistanceTransform.hpp
    Widget/OpenCV/imgproc/COcvWidgetEqualizeHist.hpp
    Widget/OpenCV/imgproc/COcvWidgetGaussianBlur.hpp
    Widget/OpenCV/imgproc/COcvWidgetGrabCut.hpp
    Widget/OpenCV/imgproc/COcvWidgetHoughCircles.hpp
    Widget/OpenCV/imgproc/COcvWidgetHoughLines.hpp
    Widget/OpenCV/imgproc/COcvWidgetLaplacian.hpp
    Widget/OpenCV/imgproc/COcvWidgetMedian.hpp
    Widget/OpenCV/imgproc/COcvWidgetMorphologyEx.hpp
    Widget/OpenCV/imgproc/COcvWidgetResize.hpp
    Widget/OpenCV/imgproc/COcvWidgetRotateEx.hpp
    Widget/OpenCV/imgproc/COcvWidgetSobel.hpp
    Widget/OpenCV/imgproc/COcvWidgetStackBlur.hpp
    Widget/OpenCV/imgproc/COcvWidgetThreshold.hpp
    Widget/OpenCV/imgproc/COcvWidgetWatershed.hpp
    Widget/OpenCV/objdetect/COcvWidgetCascadeClassifier.hpp
    Widget/OpenCV/objdetect/COcvWidgetQRCodeDetector.hpp
    Widget/OpenCV/optflow/COcvWidgetDeepFlow.hpp
    Widget/OpenCV/optflow/COcvWidgetDualTVL1OpticalFlow.hpp
    Widget/OpenCV/optflow/COcvWidgetPCAOpticalFlow.hpp
    Widget/OpenCV/optflow/COcvWidgetSimpleFlow.hpp
    Widget/OpenCV/optflow/COcvWidgetSparseToDenseOF.hpp
    Widget/OpenCV/photo/COcvWidgetColorchange.hpp
    Widget/OpenCV/photo/COcvWidgetDecolor.hpp
    Widget/OpenCV/photo/COcvWidgetDenoiseTVL1.hpp
    Widget/OpenCV/photo/COcvWidgetDetailEnhance.hpp
    Widget/OpenCV/photo/COcvWidgetEdgePreservingFilter.hpp
    Widget/OpenCV/photo/COcvWidgetFastNlMeans.hpp
    Widget/OpenCV/photo/COcvWidgetFastNlMeansMulti.hpp
    Widget/OpenCV/photo/COcvWidgetIlluminationChange.hpp
    Widget/OpenCV/photo/COcvWidgetInpaint.hpp
    Widget/OpenCV/photo/COcvWidgetPencilSketch.hpp
    Widget/OpenCV/photo/COcvWidgetSeamlessCloning.hpp
    Widget/OpenCV/photo/COcvWidgetStylization.hpp
    Widget/OpenCV/photo/COcvWidgetTextureFlattening.hpp
    Widget/OpenCV/saliency/COcvWidgetMotionSaliencyBinWangApr2014.hpp
    Widget/OpenCV/saliency/COcvWidgetObjectnessBING.hpp
    Widget/OpenCV/saliency/COcvWidgetSaliencyFineGrained.hpp
    Widget/OpenCV/saliency/COcvWidgetSaliencySpectralResidual.hpp
    Widget/OpenCV/superres/COcvWidgetSuperResBTVL1.hpp
    Widget/OpenCV/text/COcvWidgetOCRTesseract.hpp
    Widget/OpenCV/tracking/COcvWidgetTrackerGOTURN.hpp
    Widget/OpenCV/tracking/COcvWidgetTrackerKCF.hpp
    Widget/OpenCV/video/COcvWidgetBckgndSubKnn.hpp
    Widget/OpenCV/video/COcvWidgetBckgndSubMog2.hpp
    Widget/OpenCV/video/COcvWidgetCamShift.hpp
    Widget/OpenCV/video/COcvWidgetDISOpticalFlow.hpp
    Widget/OpenCV/video/COcvWidgetFarnebackOpticalFlow.hpp
    Widget/OpenCV/video/COcvWidgetMeanShift.hpp
    Widget/OpenCV/ximgproc/COcvWidgetAdaptiveManifold.hpp
    Widget/OpenCV/ximgproc/COcvWidgetAnisotropicDiffusion.hpp
    Widget/OpenCV/ximgproc/COcvWidgetBilateralTextureFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetDTFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetDTFilterEnhance.hpp
    Widget/OpenCV/ximgproc/COcvWidgetDTFilterStylize.hpp
    Widget/OpenCV/ximgproc/COcvWidgetFastGlobalSmootherFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetFastLineDetector.hpp
    Widget/OpenCV/ximgproc/COcvWidgetGradientDeriche.hpp
    Widget/OpenCV/ximgproc/COcvWidgetGradientPaillou.hpp
    Widget/OpenCV/ximgproc/COcvWidgetGraphSegmentation.hpp
    Widget/OpenCV/ximgproc/COcvWidgetGuidedFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetJointBilateralFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetL0Smooth.hpp
    Widget/OpenCV/ximgproc/COcvWidgetNiblackThreshold.hpp
    Widget/OpenCV/ximgproc/COcvWidgetPeiLinNormalization.hpp
    Widget/OpenCV/ximgproc/COcvWidgetRidgeFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetRollingGuidanceFilter.hpp
    Widget/OpenCV/ximgproc/COcvWidgetSelectiveSearchSegmentation.hpp
    Widget/OpenCV/ximgproc/COcvWidgetStructuredEdgeDetection.hpp
    Widget/OpenCV/ximgproc/COcvWidgetSuperpixelLSC.hpp
    Widget/OpenCV/ximgproc/COcvWidgetSuperpixelSEEDS.hpp
    Widget/OpenCV/ximgproc/COcvWidgetSuperpixelSLIC.hpp
    Widget/OpenCV/ximgproc/COcvWidgetThinning.hpp
    Widget/OpenCV/xphoto/COcvWidgetGrayworldWB.hpp
    Widget/OpenCV/xphoto/COcvWidgetInpaintX.hpp
    Widget/OpenCV/xphoto/COcvWidgetLearningBasedWB.hpp
    Widget/OpenCV/xphoto/COcvWidgetSimpleWB.hpp
    Widget/OpenCV/xphoto/COcvWidgetTonemapDurand.hpp
    IO/CTextStreamIO.h IO/CTextStreamIO.cpp
)

set_target_properties(ikDataProcess PROPERTIES VERSION ${IKOMIA_VERSION} SOVERSION ${IKOMIA_VERSION})

set(PUBLIC_HEADERS
    DataProcessGlobal.hpp
)

set(PUBLIC_HEADERS_CORE
    Core/CPluginProcessInterface.hpp
    Core/CTaskFactory.hpp
    Core/CTaskParamFactory.hpp
    Core/CWorkflow.h
    Core/CWidgetFactory.hpp
    Core/CRunTaskManager.h
)

set(PUBLIC_HEADERS_IO
    IO/CImageIO.h
    IO/CBlobMeasureIO.h
    IO/CGraphicsInput.h
    IO/CGraphicsOutput.h
    IO/CTaskIORegistration.h
    IO/CNumericIO.h
    IO/CVideoIO.h
    IO/CDatasetIO.h
    IO/CObjectDetectionIO.h
    IO/CInstanceSegIO.h
    IO/CJsonIO.h
    IO/CScene3dIO.h
    IO/CSemanticSegIO.h
    IO/CTextIO.h
)

set(PUBLIC_HEADERS_IO_SCENE_3D
    IO/Scene3d/CScene3d.h
    IO/Scene3d/CScene3dColor.h
    IO/Scene3d/CScene3dCoord.h
    IO/Scene3d/CScene3dImage2d.h
    IO/Scene3d/CScene3dLayer.h
    IO/Scene3d/CScene3dObject.h
    IO/Scene3d/CScene3dShapeCircle.h
    IO/Scene3d/CScene3dShapePoint.h
    IO/Scene3d/CScene3dShapePoly.h
    IO/Scene3d/CScene3dVectorField.cpp
    IO/Scene3d/CScene3dVectorField.h
)

set(PUBLIC_HEADERS_PROCESS
    Process/Core/CBlobMeasure.h
)

set(PUBLIC_HEADERS_TASK
    Task/C2dImageInteractiveTask.h
    Task/C2dImageTask.h
    Task/CVideoTask.h
    Task/CVideoOFTask.h
    Task/CVideoTrackingTask.h
    Task/C2dFeatureImageTask.h
    Task/C2dFeatureMatcherTask.h
    Task/CInstanceSegTask.h
    Task/CTaskInfo.h
    Task/CDnnTrainTask.h
    Task/CMlflowTrainTask.h
    Task/CClassificationTask.h
    Task/CObjectDetectionTask.h
    Task/CSemanticSegTask.h
)

set(PUBLIC_HEADERS_PROCESS_DNN
    Process/OpenCV/dnn/COcvDnnProcess.h
)

set(PUBLIC_HEADERS_WIDGET_DNN
    Widget/OpenCV/dnn/COcvWidgetDnnCore.h
)

target_precompile_headers(ikDataProcess PRIVATE
    pch.hpp
)

target_compile_definitions(ikDataProcess PRIVATE
    DATAPROCESS_LIBRARY
)

target_compile_features(ikDataProcess PRIVATE cxx_std_14)

target_include_directories(ikDataProcess PRIVATE
    ../Core
    ../DataIO
    ../Utils
    # GMIC
    ${CMAKE_CURRENT_LIST_DIR}/../../gmic/src
)

target_link_libraries(ikDataProcess PRIVATE
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Python3::Python
    Boost::system
    Boost::filesystem
    ${BOOST_PYTHON_TARGET}
    OpenMP::OpenMP_CXX
    opencv_bgsegm
    opencv_bioinspired
    opencv_core
    opencv_dnn
    opencv_face
    opencv_features2d
    opencv_fuzzy
    opencv_hfs
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_objdetect
    opencv_optflow
    opencv_photo
    opencv_saliency
    opencv_superres
    opencv_text
    opencv_tracking
    opencv_video
    opencv_ximgproc
    opencv_xphoto
    gmic
    ikUtils
    ikCore
    ikDataIO
)

if(WIN32)
    target_link_directories(ikDataProcess PRIVATE
        # GMIC
        $ENV{ProgramW6432}/gmic/bin
        $ENV{ProgramW6432}/gmic/lib
    )

    target_link_libraries(ikDataProcess PRIVATE
        shell32
    )
endif()

if(UNIX)
    target_link_directories(ikDataProcess PRIVATE
        # GMIC
        ${CMAKE_CURRENT_LIST_DIR}/../../gmic/build
    )
endif()

if(OPENCV_GPU)
    target_link_libraries(ikDataProcess PRIVATE
        opencv_cudawarping
    )
endif()

if(WIN32)
    target_compile_options(ikDataProcess PRIVATE
        /bigobj
    )
endif()

# Translations
qt5_create_translation(QM_FILE ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR}/Translations/dataprocess_fr.ts)

install(TARGETS ikDataProcess
    LIBRARY DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
    RUNTIME DESTINATION ${CMAKE_INSTALL_IKOMIA_LIB}
)

install(FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess
)

install(FILES ${PUBLIC_HEADERS_CORE}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/Core
)

install(FILES ${PUBLIC_HEADERS_IO}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/IO
)

install(FILES ${PUBLIC_HEADERS_IO_SCENE_3D}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/IO/Scene3d
)

install(FILES ${PUBLIC_HEADERS_PROCESS}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/Process/Core
)

install(FILES ${PUBLIC_HEADERS_PROCESS_DNN}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/Process/OpenCV/dnn
)

install(FILES ${PUBLIC_HEADERS_TASK}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/Task
)

install(FILES ${PUBLIC_HEADERS_WIDGET_DNN}
    DESTINATION ${CMAKE_INSTALL_IKOMIA_INCLUDE}/DataProcess/Widget/OpenCV/dnn
)
