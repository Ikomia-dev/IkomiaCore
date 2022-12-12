add_library(ikDataProcess SHARED
    CPluginProcessInterface.hpp
    CProcessRegistration.cpp CProcessRegistration.h
    Core/C2dFeatureImageTask.cpp Core/C2dFeatureImageTask.h
    Core/C2dFeatureMatcherTask.cpp Core/C2dFeatureMatcherTask.h
    Core/C2dImageInteractiveTask.cpp Core/C2dImageInteractiveTask.h
    Core/C2dImageTask.cpp Core/C2dImageTask.h
    Core/CDnnTrainTask.cpp Core/CDnnTrainTask.h
    Core/CIkomiaRegistry.cpp Core/CIkomiaRegistry.h
    Core/CMlflowTrainTask.cpp Core/CMlflowTrainTask.h
    Core/CRunTaskManager.cpp Core/CRunTaskManager.h
    Core/CTaskFactory.hpp
    Core/CTaskInfo.cpp Core/CTaskInfo.h
    Core/CVideoOFTask.cpp Core/CVideoOFTask.h
    Core/CVideoTask.cpp Core/CVideoTask.h
    Core/CVideoTrackingTask.cpp Core/CVideoTrackingTask.h
    Core/CWidgetFactory.hpp
    Core/CWorkflow.cpp Core/CWorkflow.h
    DataProcess.hpp
    DataProcessGlobal.hpp
    DataProcessTools.hpp
    IO/CArrayIO.cpp IO/CArrayIO.h
    IO/CBlobMeasureIO.cpp IO/CBlobMeasureIO.h
    IO/CConvertIO.cpp IO/CConvertIO.h
    IO/CDatasetIO.cpp IO/CDatasetIO.h
    IO/CGraphicsInput.cpp IO/CGraphicsInput.h
    IO/CGraphicsOutput.cpp IO/CGraphicsOutput.h
    IO/CImageIO.cpp IO/CImageIO.h
    IO/CInstanceSegIO.cpp IO/CInstanceSegIO.h
    IO/CNumericIO.cpp IO/CNumericIO.h
    IO/CObjectDetectionIO.cpp IO/CObjectDetectionIO.h
    IO/CPathIO.cpp IO/CPathIO.h
    IO/CSemanticSegIO.cpp IO/CSemanticSegIO.h
    IO/CTaskIORegistration.cpp IO/CTaskIORegistration.h
    IO/CVideoIO.cpp IO/CVideoIO.h
    IO/CWidgetOutput.cpp IO/CWidgetOutput.h
    Process/Core/CBinaryToGraphics.cpp Process/Core/CBinaryToGraphics.h
    Process/Core/CBlobMeasure.cpp Process/Core/CBlobMeasure.h
    Process/Core/CCut.hpp
    Process/Core/CFillHoles.hpp
    Process/Core/CGraphicsToBinary.cpp Process/Core/CGraphicsToBinary.h
    Process/Core/CInstanceSegFilter.cpp Process/Core/CInstanceSegFilter.h
    Process/Core/CObjDetectFilter.cpp Process/Core/CObjDetectFilter.h
    Process/Core/CPlotMerge.hpp
    Process/Core/CRgbHlsThreshold.cpp Process/Core/CRgbHlsThreshold.h
    Process/Core/CSemanticSegFilter.cpp Process/Core/CSemanticSegFilter.h
    Process/Core/ProcessCore.hpp
    Process/Gmic/CGmicImageConverter.cpp Process/Gmic/CGmicImageConverter.h
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
    Process/OpenCV/dnn/COcvDnnProcess.cpp Process/OpenCV/dnn/COcvDnnProcess.h
    Process/OpenCV/dnn/COcvDnnSegmentation.cpp Process/OpenCV/dnn/COcvDnnSegmentation.h
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
    Process/OpenCV/superres/COcvSuperResBTVL1.cpp Process/OpenCV/superres/COcvSuperResBTVL1.h
    Process/OpenCV/superres/CSuperResFrameSrc.cpp Process/OpenCV/superres/CSuperResFrameSrc.h
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
    Widget/Core/CWidgetApplyOnly.hpp
    Widget/Core/CWidgetBinaryToGraphics.hpp
    Widget/Core/CWidgetBlobMeasure.cpp Widget/Core/CWidgetBlobMeasure.h
    Widget/Core/CWidgetCut.hpp
    Widget/Core/CWidgetFillHoles.hpp
    Widget/Core/CWidgetGraphicsToBinary.hpp
    Widget/Core/CWidgetPlotMerge.hpp
    Widget/Core/CWidgetRgbHlsThreshold.cpp Widget/Core/CWidgetRgbHlsThreshold.h
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
    Widget/OpenCV/dnn/COcvWidgetDnnCore.cpp Widget/OpenCV/dnn/COcvWidgetDnnCore.h
    Widget/OpenCV/dnn/COcvWidgetDnnDetector.hpp
    Widget/OpenCV/dnn/COcvWidgetDnnProcess.cpp Widget/OpenCV/dnn/COcvWidgetDnnProcess.h
    Widget/OpenCV/dnn/COcvWidgetDnnSegmentation.hpp
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
    pch.hpp
)

target_compile_definitions(ikDataProcess PUBLIC
    DATAPROCESS_LIBRARY
)

target_include_directories(ikDataProcess PUBLIC
    ../Core
    ../DataIO
    ../Utils
    # GMIC
    ${CMAKE_CURRENT_LIST_DIR}/../../gmic/src
)

target_link_libraries(ikDataProcess PUBLIC
    Qt::Concurrent
    Qt::Core
    Qt::Gui
    Qt::Sql
    Qt::Widgets
    Boost::system
    Boost::filesystem
    Boost::python${PYTHON_VERSION_NO_DOT}
    gmic
)

if(WIN32)
    target_link_directories(ikDataProcess PUBLIC
        # GMIC
        $ENV{ProgramW6432}/gmic/bin
        $ENV{ProgramW6432}/gmic/lib
    )

    target_link_libraries(ikUtils PUBLIC
        $<$<CONFIG:RELEASE>:opencv_bgsegm${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_bgsegm${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_bioinspired${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_bioinspired${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_core${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_core${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_dnn${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_dnn${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_face${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_face${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_features2d${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_features2d${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_fuzzy${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_fuzzy${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_hfs${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_hfs${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_highgui${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_highgui${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_imgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_imgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_objdetect${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_objdetect${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_optflow${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_optflow${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_photo${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_photo${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_saliency${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_saliency${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_superres${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_superres${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_text${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_text${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_tracking${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_tracking${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_video${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_video${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_ximgproc${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_ximgproc${OPENCV_VERSION}d>
        $<$<CONFIG:RELEASE>:opencv_xphoto${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_xphoto${OPENCV_VERSION}d>
    )
endif()

if(UNIX)
    target_link_directories(ikDataProcess PUBLIC
        # GMIC
        ${CMAKE_CURRENT_LIST_DIR}/../../gmic/build
    )

    target_link_libraries(ikUtils PUBLIC
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
    )
endif()

if(WIN32 AND NOT IK_CPU)
    target_link_libraries(ikDataProcess PUBLIC
        $<$<CONFIG:RELEASE>:opencv_cudawarping${OPENCV_VERSION}>
        $<$<CONFIG:DEBUG>:opencv_cudawarping${OPENCV_VERSION}d>
    )
endif()

if(UNIX AND NOT IK_CPU)
    target_link_libraries(ikDataProcess PUBLIC
        opencv_cudawarping
    )
endif()

if(WIN32)
    target_link_libraries(ikDataProcess PUBLIC
        # Remove: L$ENV{ProgramW6432}/Boost/lib
        # Remove: L$ENV{ProgramW6432}/GDCM/bin
        # Remove: L$ENV{ProgramW6432}/OpenSSL-Win64/bin
        # Remove: L$ENV{ProgramW6432}/Python38/
        # Remove: L$ENV{ProgramW6432}/Python38/libs
        # Remove: L$ENV{ProgramW6432}/VTK/bin
        # Remove: L$ENV{ProgramW6432}/VTK/lib
        # Remove: L$ENV{ProgramW6432}/gmic/bin/
        # Remove: L$ENV{ProgramW6432}/gmic/lib/
        # Remove: L$ENV{ProgramW6432}/intel_tbb/bin/intel64/vc14
        # Remove: L$ENV{ProgramW6432}/leptonica/bin
        # Remove: L$ENV{ProgramW6432}/tesseract/bin
        # Remove: L$ENV{ProgramW6432}/zlib/bin
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../quazip/build/lib
        # Remove: L${CMAKE_CURRENT_SOURCE_DIR}/../qwt/build/lib
        # Remove: L'C:\Program
#        (
#        )
#        Files
#        Kits\10\Lib\10.0.19041.0\um\x64'
#        \Windows
#        shell32
#        x86
    )

    target_compile_options(ikDataProcess
        -bigobj
    )
endif()

if(UNIX AND NOT MACOS)
    target_link_libraries(ikDataProcess PUBLIC
        # Remove: L/usr/lib64/nvidia/
        # Remove: L/usr/local/cuda/lib64/
        # Remove: L/usr/local/lib/
        # Remove: L/usr/local/lib64/
    )
endif()

if(UNIX AND centos7 AND NOT MACOS)
    target_link_libraries(ikDataProcess PUBLIC
        # Remove: L/work/shared/local/lib/
        # Remove: L/work/shared/local/lib64/
    )
endif()

if(WIN32 AND ik_cpu)
    target_link_libraries(ikDataProcess PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cpu/x64/vc/lib
    )
endif()

if(WIN32 AND NOT ik_cpu)
    target_link_libraries(ikDataProcess PUBLIC
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/bin
        # Remove: L$ENV{ProgramW6432}/OpenCV/cuda/x64/vc/lib
        # Remove: L'$ENV{ProgramW6432}/NVIDIA
#        Computing
#        GPU
#        Toolkit/CUDA/v/bin'
#        Toolkit/CUDA/v/lib/x64'
    )
endif()

install(TARGETS ikDataProcess
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
