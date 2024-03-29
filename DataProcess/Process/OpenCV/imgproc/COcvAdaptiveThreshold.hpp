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

#ifndef COCVADAPTIVETHRESHOLD_HPP
#define COCVADAPTIVETHRESHOLD_HPP

#include "DataProcessTools.hpp"
#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"

//--------------------------------------//
//----- COcvAdaptiveThresholdParam -----//
//--------------------------------------//
class COcvAdaptiveThresholdParam: public CWorkflowTaskParam
{
    public:

        COcvAdaptiveThresholdParam() : CWorkflowTaskParam()
        {
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            m_adaptiveMethod = std::stoi(paramMap.at("adaptiveMethod"));
            m_thresholdType = std::stoi(paramMap.at("thresholdType"));
            m_blockSize = std::stoi(paramMap.at("blockSize"));
            m_offset = std::stod(paramMap.at("offset"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("adaptiveMethod", std::to_string(m_adaptiveMethod)));
            map.insert(std::make_pair("thresholdType", std::to_string(m_thresholdType)));
            map.insert(std::make_pair("blockSize", std::to_string(m_blockSize)));
            map.insert(std::make_pair("offset", std::to_string(m_offset)));
            return map;
        }

    public:

        int     m_adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C;
        int     m_thresholdType = cv::THRESH_BINARY;
        int     m_blockSize = 33;
        double  m_offset = 10.0;
};

//--------------------------------------//
//----- COcvAdaptiveThresholdParam -----//
//--------------------------------------//
class COcvAdaptiveThreshold : public C2dImageTask
{
    public:

        COcvAdaptiveThreshold() : C2dImageTask()
        {
            getOutput(0)->setDataType(IODataType::IMAGE_BINARY);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
        }
        COcvAdaptiveThreshold(const std::string name, const std::shared_ptr<COcvAdaptiveThresholdParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<COcvAdaptiveThresholdParam>(*pParam);
            getOutput(0)->setDataType(IODataType::IMAGE_BINARY);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
        }

        size_t     getProgressSteps() override
        {
            return 3;
        }

        void    updateStaticOutputs() override
        {
            C2dImageTask::updateStaticOutputs();
            auto pImgOutput =  std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            assert(pImgOutput);
            pImgOutput->setChannelCount(1);
        }

        void    run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
            auto pParam = std::dynamic_pointer_cast<COcvAdaptiveThresholdParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgSrcTmp, imgDst;
            CMat imgSrc = pInput->getImage();
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                //Require 8 bits monochrome source image
                imgSrcTmp = conformInput(imgSrc);
                cv::adaptiveThreshold(imgSrcTmp, imgDst, 255, pParam->m_adaptiveMethod, pParam->m_thresholdType, pParam->m_blockSize, pParam->m_offset);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }

            forwardInputImage(0, 1);
            applyGraphicsMaskToBinary(imgDst, imgDst, 0);
            emit m_signalHandler->doProgress();

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            endTaskRun();
            emit m_signalHandler->doProgress();
        }

        CMat    conformInput(CMat source)
        {
            CMat conformImg;

            if(source.depth() != CV_8S && source.depth() != CV_8U)
            {
                source.convertTo(conformImg, CV_8U);
                if(source.channels() != 1)
                    cv::cvtColor(source, conformImg, cv::COLOR_RGB2GRAY, 0);
            }
            else if(source.channels() != 1)
                cv::cvtColor(source, conformImg, cv::COLOR_RGB2GRAY, 0);
            else
                conformImg = source;

            return conformImg;
        }
};

class COcvAdaptiveThresholdFactory : public CTaskFactory
{
    public:

        COcvAdaptiveThresholdFactory()
        {
            m_info.m_name = "ocv_adaptive_threshold";
            m_info.m_shortDescription = QObject::tr("Adaptive thresholding according to local background estimation").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Image processing/Miscellaneous image transformations").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "segmentation,adaptive,threshold,mean,gaussian";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pDerivedParam = std::dynamic_pointer_cast<COcvAdaptiveThresholdParam>(pParam);
            if(pDerivedParam != nullptr)
                return std::make_shared<COcvAdaptiveThreshold>(m_info.m_name, pDerivedParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pDerivedParam = std::make_shared<COcvAdaptiveThresholdParam>();
            assert(pDerivedParam != nullptr);
            return std::make_shared<COcvAdaptiveThreshold>(m_info.m_name, pDerivedParam);
        }
};

#endif // COCVADAPTIVETHRESHOLD_HPP
