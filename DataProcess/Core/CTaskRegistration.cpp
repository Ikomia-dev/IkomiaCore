// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CTaskRegistration.h"
#include "Process/Core/ProcessCore.hpp"
#include "Process/OpenCV/ProcessOcv.hpp"
#include "Process/Gmic/ProcessGmic.hpp"
#include "Widget/Core/WidgetCore.hpp"
#include "Widget/OpenCV/WidgetOcv.hpp"
#include "Widget/Gmic/WidgetGmic.hpp"

CTaskRegistration::CTaskRegistration()
{
    registerCore();
    registerOpenCV();
    registerGmic();
}

CTaskRegistration::~CTaskRegistration()
{
    // Explicit destruction in case of multi-threading context with Python
    CPyEnsureGIL gil;
    m_taskFactory.clear();
    m_widgetFactory.clear();
    m_paramFactory.clear();
}

void CTaskRegistration::registerTask(const std::shared_ptr<CTaskFactory>& pTaskFactory,
                                           const std::shared_ptr<CWidgetFactory>& pWidgetFactory,
                                           const std::shared_ptr<CTaskParamFactory>& pTaskParamFactory)
{
    assert(pTaskFactory);

    // Register process factory
    std::string name = pTaskFactory->getInfo().getName();
    if(m_taskFactory.isCreatorExists(name) == true)
    {
        m_taskFactory.unregisterCreator(name);
        m_taskFactory.remove(name);
    }

    m_taskFactory.getList().push_back(pTaskFactory);
    auto pProcessFunc = [pTaskFactory](const WorkflowTaskParamPtr& param)
    {
        //Passage par lambda -> pFactory par valeur pour assurer la portée du pointeur
        return pTaskFactory->create(param);
    };
    m_taskFactory.registerCreator(pTaskFactory->getInfo().m_name, pProcessFunc);

    // Register widget factory
    if(pWidgetFactory)
    {
        name = pWidgetFactory->getName();
        if(m_widgetFactory.isCreatorExists(name) == true)
        {
            m_widgetFactory.unregisterCreator(name);
            m_widgetFactory.remove(name);
        }

        m_widgetFactory.getList().push_back(pWidgetFactory);
        auto pWidgetFunc = [pWidgetFactory](const WorkflowTaskParamPtr& param)
        {
            return pWidgetFactory->create(param);
        };
        m_widgetFactory.registerCreator(name, pWidgetFunc);
    }

    // Register param factory
    if (pTaskParamFactory)
    {
        name = pTaskParamFactory->getName();
        if(m_paramFactory.isCreatorExists(name) == true)
        {
            m_paramFactory.unregisterCreator(name);
            m_paramFactory.remove(name);
        }

        m_paramFactory.getList().push_back(pTaskParamFactory);
        auto pParamFunc = [pTaskParamFactory]()
        {
            return pTaskParamFactory->create();
        };
        m_paramFactory.registerCreator(name, pParamFunc);
    }

    //Pour mémoire
    //Passage par std::bind -> cast nécessaire car 2 méthodes create() existent
    //auto pFunc = static_cast<std::shared_ptr<CWorkflowTask> (CTaskFactory::*)(const std::shared_ptr<CWorkflowTaskParam>&)>(&CTaskFactory::create);
    //m_factory.registerCreator(pFactory->name(), std::bind(pFunc, pFactory, std::placeholders::_1));
}

void CTaskRegistration::unregisterTask(const std::string &name)
{
    m_taskFactory.remove(name);
    m_widgetFactory.remove(name);
    m_paramFactory.remove(name);
}

WorkflowTaskPtr CTaskRegistration::createTaskObject(const std::string &name, const WorkflowTaskParamPtr &paramPtr)
{
    WorkflowTaskPtr taskPtr = nullptr;
    try
    {
        taskPtr = m_taskFactory.createObject(name, std::move(paramPtr));
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException());
    }
    return taskPtr;
}

WorkflowTaskWidgetPtr CTaskRegistration::createWidgetObject(const std::string &name, const WorkflowTaskParamPtr &paramPtr)
{
    WorkflowTaskWidgetPtr widgetPtr = nullptr;
    try
    {
        widgetPtr = m_widgetFactory.createObject(name, std::move(paramPtr));
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException());
    }
    return widgetPtr;
}

WorkflowTaskParamPtr CTaskRegistration::createParamObject(const std::string &name)
{
    WorkflowTaskParamPtr paramPtr = nullptr;
    try
    {
        paramPtr = m_paramFactory.createObject(name);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException());
    }
    return paramPtr;
}

void CTaskRegistration::reset()
{
    m_taskFactory.getList().clear();
    m_widgetFactory.getList().clear();
    m_paramFactory.getList().clear();
    registerCore();
    registerOpenCV();
    registerGmic();
}

void CTaskRegistration::registerCore()
{
    registerTask(std::make_shared<CBinaryToGraphicsFactory>(), std::make_shared<CWidgetBinaryToGraphicsFactory>());
    registerTask(std::make_shared<CBlobMeasureFactory>(), std::make_shared<CWidgetBlobMeasureFactory>());
    registerTask(std::make_shared<CCutFactory>(), std::make_shared<CWidgetCutFactory>());
    registerTask(std::make_shared<CFillHolesFactory>(), std::make_shared<CWidgetFillHolesFactory>());
    registerTask(std::make_shared<CGraphicsToBinaryFactory>(), std::make_shared<CWidgetGraphicsToBinaryFactory>());
    registerTask(std::make_shared<CObjDetectFilterFactory>(), std::make_shared<CWidgetObjDetectFilterFactory>());
    registerTask(std::make_shared<CInstanceSegFilterFactory>(), std::make_shared<CWidgetInstanceSegFilterFactory>());
    registerTask(std::make_shared<CSemanticSegFilterFactory>(), std::make_shared<CWidgetSemanticSegFilterFactory>());
    registerTask(std::make_shared<CPlotMergeFactory>(), std::make_shared<CWidgetPlotMergeFactory>());
    registerTask(std::make_shared<CRgbHlsThresholdFactory>(), std::make_shared<CWidgetRgbHlsThresholdFactory>());
}

void CTaskRegistration::registerOpenCV()
{
    registerCvCore();
    registerCvDnn();
    registerCvFeatures2d();
    registerCvImgproc();
    registerCvPhoto();
    registerCvTracking();
    registerCvVideo();
    registerCvBgsegm();
    registerCvXimgproc();
    registerCvXphoto();
    registerCvOptflow();
    registerCvBioinspired();
    registerCvSaliency();
    registerCvSuperres();
    registerCvObjdetect();
    registerCvText();

    registerTask(std::make_shared<COcvInpaintFuzzyFactory>(), std::make_shared<COcvWidgetInpaintFuzzyFactory>());
    registerTask(std::make_shared<COcvHfsSegmentFactory>(), std::make_shared<COcvWidgetHfsSegmentFactory>());
}

void CTaskRegistration::registerGmic()
{
    //Colors
    registerTask(std::make_shared<CGmicAutoBalanceFactory>(), std::make_shared<CGmicWidgetAutoBalanceFactory>());
    registerTask(std::make_shared<CGmicBoostChromaFactory>(), std::make_shared<CGmicWidgetBoostChromaFactory>());
    registerTask(std::make_shared<CGmicBoostFadeFactory>(), std::make_shared<CGmicWidgetBoostFadeFactory>());
    registerTask(std::make_shared<CGmicColorPresetsFactory>(), std::make_shared<CGmicWidgetColorPresetsFactory>());

    //Contours
    registerTask(std::make_shared<CGmicDoGFactory>(), std::make_shared<CGmicWidgetDoGFactory>());
    registerTask(std::make_shared<CGmicDistanceTransformFactory>(), std::make_shared<CGmicWidgetDistanceTransformFactory>());
    registerTask(std::make_shared<CGmicSkeletonFactory>(), std::make_shared<CGmicWidgetSkeletonFactory>());
    registerTask(std::make_shared<CGmicSuperPixelsFactory>(), std::make_shared<CGmicWidgetSuperPixelsFactory>());

    //Details
    registerTask(std::make_shared<CGmicConstrainedSharpenFactory>(), std::make_shared<CGmicWidgetConstrainedSharpenFactory>());
    registerTask(std::make_shared<CGmicDynamicRangeIncreaseFactory>(), std::make_shared<CGmicWidgetDynamicRangeIncreaseFactory>());
    registerTask(std::make_shared<CGmicMagicDetailsFactory>(), std::make_shared<CGmicWidgetMagicDetailsFactory>());
    registerTask(std::make_shared<CGmicSharpenDeblurFactory>(), std::make_shared<CGmicWidgetSharpenDeblurFactory>());
    registerTask(std::make_shared<CGmicSharpenGradientFactory>(), std::make_shared<CGmicWidgetSharpenGradientFactory>());
    registerTask(std::make_shared<CGmicSharpenLucyFactory>(), std::make_shared<CGmicWidgetSharpenLucyFactory>());
    registerTask(std::make_shared<CGmicSharpenTonesFactory>(), std::make_shared<CGmicWidgetSharpenTonesFactory>());

    //Repair
    registerTask(std::make_shared<CGmicHotPixelsFactory>(), std::make_shared<CGmicWidgetHotPixelsFactory>());
    registerTask(std::make_shared<CGmicInpaintFactory>(), std::make_shared<CGmicWidgetInpaintFactory>());
    registerTask(std::make_shared<CGmicRedEyeFactory>(), std::make_shared<CGmicWidgetRedEyeFactory>());
}

void CTaskRegistration::registerCvCore()
{
    registerTask(std::make_shared<COcvAbsdiffFactory>(), std::make_shared<COcvWidgetAbsdiffFactory>());
    registerTask(std::make_shared<COcvAddFactory>(), std::make_shared<COcvWidgetAddFactory>());
    registerTask(std::make_shared<COcvAddWeightedFactory>(), std::make_shared<COcvWidgetAddWeightedFactory>());
    registerTask(std::make_shared<COcvCompareFactory>(), std::make_shared<COcvWidgetCompareFactory>());
    registerTask(std::make_shared<COcvConvertToFactory>(), std::make_shared<COcvWidgetConvertToFactory>());
    registerTask(std::make_shared<COcvCopyMakeBorderFactory>(), std::make_shared<COcvWidgetCopyMakeBorderFactory>());
    registerTask(std::make_shared<COcvCountNonZeroFactory>(), std::make_shared<COcvWidgetCountNonZeroFactory>());
    registerTask(std::make_shared<COcvCropFactory>(), std::make_shared<COcvWidgetCropFactory>());
    registerTask(std::make_shared<COcvDftFactory>(), std::make_shared<COcvWidgetDftFactory>());
    registerTask(std::make_shared<COcvDftInvFactory>(), std::make_shared<COcvWidgetDftInvFactory>());
    registerTask(std::make_shared<COcvDivideFactory>(), std::make_shared<COcvWidgetDivideFactory>());
    registerTask(std::make_shared<COcvExpFactory>(), std::make_shared<COcvWidgetExpFactory>());
    registerTask(std::make_shared<COcvExtractChannelFactory>(), std::make_shared<COcvWidgetExtractChannelFactory>());
    registerTask(std::make_shared<COcvFlipFactory>(), std::make_shared<COcvWidgetFlipFactory>());
    registerTask(std::make_shared<COcvInRangeFactory>(), std::make_shared<COcvWidgetInRangeFactory>());
    registerTask(std::make_shared<COcvInsertChannelFactory>(), std::make_shared<COcvWidgetInsertChannelFactory>());
    registerTask(std::make_shared<COcvLogFactory>(), std::make_shared<COcvWidgetLogFactory>());
    registerTask(std::make_shared<COcvLogicalOpFactory>(), std::make_shared<COcvWidgetLogicalOpFactory>());
    registerTask(std::make_shared<COcvMagnitudeFactory>(), std::make_shared<COcvWidgetMagnitudeFactory>());    
    registerTask(std::make_shared<COcvMaxFactory>(), std::make_shared<COcvWidgetMaxFactory>());
    registerTask(std::make_shared<COcvMergeFactory>(), std::make_shared<COcvWidgetMergeFactory>());
    registerTask(std::make_shared<COcvMinFactory>(), std::make_shared<COcvWidgetMinFactory>());
    registerTask(std::make_shared<COcvMultiplyFactory>(), std::make_shared<COcvWidgetMultiplyFactory>());
    registerTask(std::make_shared<COcvNegativeFactory>(), std::make_shared<COcvWidgetNegativeFactory>());
    registerTask(std::make_shared<COcvNormalizeFactory>(), std::make_shared<COcvWidgetNormalizeFactory>());
    registerTask(std::make_shared<COcvPSNRFactory>(), std::make_shared<COcvWidgetPSNRFactory>());
    registerTask(std::make_shared<COcvRotateFactory>(), std::make_shared<COcvWidgetRotateFactory>());
    registerTask(std::make_shared<COcvSplitFactory>(), std::make_shared<COcvWidgetSplitFactory>());
    registerTask(std::make_shared<COcvSubtractFactory>(), std::make_shared<COcvWidgetSubtractFactory>());

    registerTask(std::make_shared<COcvKMeansFactory>(), std::make_shared<COcvWidgetKMeansFactory>());
}

void CTaskRegistration::registerCvDnn()
{
    registerTask(std::make_shared<COcvDnnClassifierFactory>(), std::make_shared<COcvWidgetDnnClassifierFactory>());
    registerTask(std::make_shared<COcvDnnColorizationFactory>(), std::make_shared<COcvWidgetDnnColorizationFactory>());
    registerTask(std::make_shared<COcvDnnDetectorFactory>(), std::make_shared<COcvWidgetDnnDetectorFactory>());
    registerTask(std::make_shared<COcvDnnSegmentationFactory>(), std::make_shared<COcvWidgetDnnSegmentationFactory>());
}

void CTaskRegistration::registerCvImgproc()
{
    registerTask(std::make_shared<COcvAdaptiveThresholdFactory>(), std::make_shared<COcvWidgetAdaptiveThresholdFactory>());
    registerTask(std::make_shared<COcvBilateralFactory>(), std::make_shared<COcvWidgetBilateralFactory>());
    registerTask(std::make_shared<COcvBlurFactory>(), std::make_shared<COcvWidgetBlurFactory>(), std::make_shared<COcvBlurParamFactory>());
    registerTask(std::make_shared<COcvBoxFilterFactory>(), std::make_shared<COcvWidgetBoxFilterFactory>());
    registerTask(std::make_shared<COcvCalcHistFactory>(), std::make_shared<COcvWidgetCalcHistFactory>());
    registerTask(std::make_shared<COcvCannyFactory>(), std::make_shared<COcvWidgetCannyFactory>());
    registerTask(std::make_shared<COcvCascadeClassifierFactory>(), std::make_shared<COcvWidgetCascadeClassifierFactory>());
    registerTask(std::make_shared<COcvCLAHEFactory>(), std::make_shared<COcvWidgetCLAHEFactory>());
    registerTask(std::make_shared<COcvColorMapFactory>(), std::make_shared<COcvWidgetColorMapFactory>());
    registerTask(std::make_shared<COcvCvtColorFactory>(), std::make_shared<COcvWidgetCvtColorFactory>());
    registerTask(std::make_shared<COcvDistanceTransformFactory>(), std::make_shared<COcvWidgetDistanceTransformFactory>());
    registerTask(std::make_shared<COcvEqualizeHistFactory>(), std::make_shared<COcvWidgetEqualizeHistFactory>());
    registerTask(std::make_shared<COcvGaussianBlurFactory>(), std::make_shared<COcvWidgetGaussianBlurFactory>());
    registerTask(std::make_shared<COcvGrabCutFactory>(), std::make_shared<COcvWidgetGrabCutFactory>());
    registerTask(std::make_shared<COcvHoughCirclesFactory>(), std::make_shared<COcvWidgetHoughCirclesFactory>());
    registerTask(std::make_shared<COcvHoughLinesFactory>(), std::make_shared<COcvWidgetHoughLinesFactory>());
    registerTask(std::make_shared<COcvLaplacianFactory>(), std::make_shared<COcvWidgetLaplacianFactory>());
    registerTask(std::make_shared<COcvMedianFactory>(), std::make_shared<COcvWidgetMedianFactory>());
    registerTask(std::make_shared<COcvMorphologyExFactory>(), std::make_shared<COcvWidgetMorphologyExFactory>());
    registerTask(std::make_shared<COcvResizeFactory>(), std::make_shared<COcvWidgetResizeFactory>());
    registerTask(std::make_shared<COcvRotateExFactory>(), std::make_shared<COcvWidgetRotateExFactory>());
    registerTask(std::make_shared<COcvSobelFactory>(), std::make_shared<COcvWidgetSobelFactory>());
    registerTask(std::make_shared<COcvStackBlurFactory>(), std::make_shared<COcvWidgetStackBlurFactory>());
    registerTask(std::make_shared<COcvThresholdFactory>(), std::make_shared<COcvWidgetThresholdFactory>());
    registerTask(std::make_shared<COcvWatershedFactory>(), std::make_shared<COcvWidgetWatershedFactory>());
}

void CTaskRegistration::registerCvFeatures2d()
{
    registerTask(std::make_shared<COcvAGASTFactory>(), std::make_shared<COcvWidgetAGASTFactory>());
    registerTask(std::make_shared<COcvAKAZEFactory>(), std::make_shared<COcvWidgetAKAZEFactory>());
    registerTask(std::make_shared<COcvBRISKFactory>(), std::make_shared<COcvWidgetBRISKFactory>());
    registerTask(std::make_shared<COcvFASTFactory>(), std::make_shared<COcvWidgetFASTFactory>());
    registerTask(std::make_shared<COcvGFTTFactory>(), std::make_shared<COcvWidgetGFTTFactory>());
    registerTask(std::make_shared<COcvKAZEFactory>(), std::make_shared<COcvWidgetKAZEFactory>());
    registerTask(std::make_shared<COcvORBFactory>(), std::make_shared<COcvWidgetORBFactory>());
    registerTask(std::make_shared<COcvSIFTFactory>(), std::make_shared<COcvWidgetSIFTFactory>());
    registerTask(std::make_shared<COcvSimpleBlobDetectorFactory>(), std::make_shared<COcvWidgetSimpleBlobDetectorFactory>());

    registerTask(std::make_shared<COcvBFMatcherFactory>(), std::make_shared<COcvWidgetBFMatcherFactory>());
    registerTask(std::make_shared<COcvFlannMatcherFactory>(), std::make_shared<COcvWidgetFlannMatcherFactory>());
}

void CTaskRegistration::registerCvPhoto()
{
    registerTask(std::make_shared<COcvDecolorFactory>(), std::make_shared<COcvWidgetDecolorFactory>());
    registerTask(std::make_shared<COcvColorchangeFactory>(), std::make_shared<COcvWidgetColorchangeFactory>());
    registerTask(std::make_shared<COcvDenoiseTVL1Factory>(), std::make_shared<COcvWidgetDenoiseTVL1Factory>());
    registerTask(std::make_shared<COcvDetailEnhanceFactory>(), std::make_shared<COcvWidgetDetailEnhanceFactory>());
    registerTask(std::make_shared<COcvEdgePreservingFilterFactory>(), std::make_shared<COcvWidgetEdgePreservingFilterFactory>());
    registerTask(std::make_shared<COcvFastNlMeansFactory>(), std::make_shared<COcvWidgetFastNlMeansFactory>());
    registerTask(std::make_shared<COcvFastNlMeansMultiFactory>(), std::make_shared<COcvWidgetFastNlMeansMultiFactory>());
    registerTask(std::make_shared<COcvIlluminationChangeFactory>(), std::make_shared<COcvWidgetIlluminationChangeFactory>());
    registerTask(std::make_shared<COcvInpaintFactory>(), std::make_shared<COcvWidgetInpaintFactory>());
    registerTask(std::make_shared<COcvPencilSketchFactory>(), std::make_shared<COcvWidgetPencilSketchFactory>());
    registerTask(std::make_shared<COcvSeamlessCloningFactory>(), std::make_shared<COcvWidgetSeamlessCloningFactory>());
    registerTask(std::make_shared<COcvStylizationFactory>(), std::make_shared<COcvWidgetStylizationFactory>());
    registerTask(std::make_shared<COcvTextureFlatteningFactory>(), std::make_shared<COcvWidgetTextureFlatteningFactory>());
}

void CTaskRegistration::registerCvTracking()
{
    registerTask(std::make_shared<COcvTrackerGOTURNFactory>(), std::make_shared<COcvWidgetTrackerGOTURNFactory>());
    registerTask(std::make_shared<COcvTrackerKCFFactory>(), std::make_shared<COcvWidgetTrackerKCFFactory>());
}

void CTaskRegistration::registerCvVideo()
{
    registerTask(std::make_shared<COcvBckgndSubKnnFactory>(), std::make_shared<COcvWidgetBckgndSubKnnFactory>());
    registerTask(std::make_shared<COcvBckgndSubMog2Factory>(), std::make_shared<COcvWidgetBckgndSubMog2Factory>());

    registerTask(std::make_shared<COcvCamShiftFactory>(), std::make_shared<COcvWidgetCamShiftFactory>());
    registerTask(std::make_shared<COcvDISOFFactory>(), std::make_shared<COcvWidgetDISOFFactory>());
    registerTask(std::make_shared<COcvFarnebackOFFactory>(), std::make_shared<COcvWidgetFarnebackOFFactory>());
    registerTask(std::make_shared<COcvMeanShiftFactory>(), std::make_shared<COcvWidgetMeanShiftFactory>());
}

void CTaskRegistration::registerCvBgsegm()
{
    registerTask(std::make_shared<COcvBckgndSubCntFactory>(), std::make_shared<COcvWidgetBckgndSubCntFactory>());
    registerTask(std::make_shared<COcvBckgndSubGmgFactory>(), std::make_shared<COcvWidgetBckgndSubGmgFactory>());
    registerTask(std::make_shared<COcvBckgndSubGsocFactory>(), std::make_shared<COcvWidgetBckgndSubGsocFactory>());
    registerTask(std::make_shared<COcvBckgndSubLsbpFactory>(), std::make_shared<COcvWidgetBckgndSubLsbpFactory>());
    registerTask(std::make_shared<COcvBckgndSubMogFactory>(), std::make_shared<COcvWidgetBckgndSubMogFactory>());
}

void CTaskRegistration::registerCvXimgproc()
{
    // Filters
    registerTask(std::make_shared<COcvAdaptiveManifoldFactory>(), std::make_shared<COcvWidgetAdaptiveManifoldFactory>());
    registerTask(std::make_shared<COcvBilateralTextureFilterFactory>(), std::make_shared<COcvWidgetBilateralTextureFilterFactory>());
    registerTask(std::make_shared<COcvGradientDericheFactory>(), std::make_shared<COcvWidgetGradientDericheFactory>());
    registerTask(std::make_shared<COcvDTFilterFactory>(), std::make_shared<COcvWidgetDTFilterFactory>());
    registerTask(std::make_shared<COcvDTFilterEnhanceFactory>(), std::make_shared<COcvWidgetDTFilterEnhanceFactory>());
    registerTask(std::make_shared<COcvDTFilterStylizeFactory>(), std::make_shared<COcvWidgetDTFilterStylizeFactory>());
    registerTask(std::make_shared<COcvFastGlobalSmootherFilterFactory>(), std::make_shared<COcvWidgetFastGlobalSmootherFilterFactory>());
    registerTask(std::make_shared<COcvGuidedFilterFactory>(), std::make_shared<COcvWidgetGuidedFilterFactory>());
    registerTask(std::make_shared<COcvGradientPaillouFactory>(), std::make_shared<COcvWidgetGradientPaillouFactory>());
    registerTask(std::make_shared<COcvJointBilateralFilterFactory>(), std::make_shared<COcvWidgetJointBilateralFilterFactory>());
    registerTask(std::make_shared<COcvL0SmoothFactory>(), std::make_shared<COcvWidgetL0SmoothFactory>());
    registerTask(std::make_shared<COcvRidgeFilterFactory>(), std::make_shared<COcvWidgetRidgeFilterFactory>());
    registerTask(std::make_shared<COcvRollingGuidanceFilterFactory>(), std::make_shared<COcvWidgetRollingGuidanceFilterFactory>());

    // Line detector
    registerTask(std::make_shared<COcvFastLineDetectorFactory>(), std::make_shared<COcvWidgetFastLineDetectorFactory>());

    // Segmentation
    registerTask(std::make_shared<COcvGraphSegmentationFactory>(), std::make_shared<COcvWidgetGraphSegmentationFactory>());
    registerTask(std::make_shared<COcvSelectiveSearchSegmentationFactory>(), std::make_shared<COcvWidgetSelectiveSearchSegmentationFactory>());

    //Superpixels
    registerTask(std::make_shared<COcvSuperpixelLSCFactory>(), std::make_shared<COcvWidgetSuperpixelLSCFactory>());
    registerTask(std::make_shared<COcvSuperpixelSEEDSFactory>(), std::make_shared<COcvWidgetSuperpixelSEEDSFactory>());
    registerTask(std::make_shared<COcvSuperpixelSLICFactory>(), std::make_shared<COcvWidgetSuperpixelSLICFactory>());

    // Structure forests
    registerTask(std::make_shared<COcvStructuredEdgeDetectionFactory>(), std::make_shared<COcvWidgetStructuredEdgeDetectionFactory>());

    // Others
    registerTask(std::make_shared<COcvAnisotropicDiffusionFactory>(), std::make_shared<COcvWidgetAnisotropicDiffusionFactory>());
    registerTask(std::make_shared<COcvNiblackThresholdFactory>(), std::make_shared<COcvWidgetNiblackThresholdFactory>());
    registerTask(std::make_shared<COcvPeiLinNormalizationFactory>(), std::make_shared<COcvWidgetPeiLinNormalizationFactory>());
    registerTask(std::make_shared<COcvThinningFactory>(), std::make_shared<COcvWidgetThinningFactory>());
}

void CTaskRegistration::registerCvXphoto()
{
    registerTask(std::make_shared<COcvGrayworldWBFactory>(), std::make_shared<COcvWidgetGrayworldWBFactory>());
    registerTask(std::make_shared<COcvLearningBasedWBFactory>(), std::make_shared<COcvWidgetLearningBasedWBFactory>());
    registerTask(std::make_shared<COcvSimpleWBFactory>(), std::make_shared<COcvWidgetSimpleWBFactory>());
    registerTask(std::make_shared<COcvInpaintXFactory>(), std::make_shared<COcvWidgetInpaintXFactory>());
    // Non free
    //registerProcess(std::make_shared<COcvTonemapDurandFactory>(), std::make_shared<COcvWidgetTonemapDurandFactory>());
}

void CTaskRegistration::registerCvOptflow()
{
    registerTask(std::make_shared<COcvDeepFlowFactory>(), std::make_shared<COcvWidgetDeepFlowFactory>());
    registerTask(std::make_shared<COcvDualTVL1OFFactory>(), std::make_shared<COcvWidgetDualTVL1OFFactory>());
    registerTask(std::make_shared<COcvPCAOFFactory>(), std::make_shared<COcvWidgetPCAOFFactory>());
    registerTask(std::make_shared<COcvSimpleFlowFactory>(), std::make_shared<COcvWidgetSimpleFlowFactory>());
    registerTask(std::make_shared<COcvSparseToDenseOFFactory>(), std::make_shared<COcvWidgetSparseToDenseOFFactory>());
}

void CTaskRegistration::registerCvBioinspired()
{
    registerTask(std::make_shared<COcvRetinaFactory>(), std::make_shared<COcvWidgetRetinaFactory>());
    // Non free
    //registerProcess(std::make_shared<COcvRetinaToneMappingFactory>(), std::make_shared<COcvWidgetRetinaToneMappingFactory>());
    registerTask(std::make_shared<COcvRetinaSegmentationFactory>(), std::make_shared<COcvWidgetRetinaSegmentationFactory>());
}

void CTaskRegistration::registerCvSaliency()
{
    registerTask(std::make_shared<COcvMotionSaliencyBinWangApr2014Factory>(), std::make_shared<COcvWidgetMotionSaliencyBinWangApr2014Factory>());
    registerTask(std::make_shared<COcvObjectnessBINGFactory>(), std::make_shared<COcvWidgetObjectnessBINGFactory>());
    registerTask(std::make_shared<COcvSaliencyFineGrainedFactory>(), std::make_shared<COcvWidgetSaliencyFineGrainedFactory>());
    registerTask(std::make_shared<COcvSaliencySpectralResidualFactory>(), std::make_shared<COcvWidgetSaliencySpectralResidualFactory>());
}

void CTaskRegistration::registerCvSuperres()
{
    registerTask(std::make_shared<COcvSuperResBTVL1Factory>(), std::make_shared<COcvWidgetSuperResBTVL1Factory>());
}

void CTaskRegistration::registerCvObjdetect()
{
    registerTask(std::make_shared<COcvQRCodeDetectorFactory>(), std::make_shared<COcvWidgetQRCodeDetectorFactory>());
}

void CTaskRegistration::registerCvText()
{
    registerTask(std::make_shared<COcvOCRTesseractFactory>(), std::make_shared<COcvWidgetOCRTesseractFactory>());
}

const CTaskAbstractFactory& CTaskRegistration::getTaskFactory() const
{
    CPyEnsureGIL gil;
    return m_taskFactory;
}

TaskFactoryPtr CTaskRegistration::getTaskFactory(const std::string &name) const
{
    CPyEnsureGIL gil;
    return m_taskFactory.getFactory(name);
}

const CWidgetAbstractFactory& CTaskRegistration::getWidgetFactory() const
{
    CPyEnsureGIL gil;
    return m_widgetFactory;
}

WidgetFactoryPtr CTaskRegistration::getWidgetFactory(const std::string &name) const
{
    CPyEnsureGIL gil;
    return m_widgetFactory.getFactory(name);
}

const CTaskParamAbstractFactory &CTaskRegistration::getTaskParamFactory() const
{
    CPyEnsureGIL gil;
    return m_paramFactory;
}

TaskParamFactoryPtr CTaskRegistration::getTaskParamFactory(const std::string &name) const
{
    CPyEnsureGIL gil;
    return m_paramFactory.getFactory(name);
}

CTaskInfo CTaskRegistration::getTaskInfo(const std::string &name) const
{
    auto factory = m_taskFactory.getFactory(name);
    if(!factory)
    {
        std::string msg = "Information for task " + name + " can't be retrieved";
        throw CException(CoreExCode::INVALID_PARAMETER, msg, __func__, __FILE__, __LINE__);
    }
    return factory->getInfo();
}
