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

#ifndef COCVBLUR_H
#define COCVBLUR_H

#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"

//-------------------------//
//----- COcvBlurParam -----//
//-------------------------//
class COcvBlurParam: public CWorkflowTaskParam
{
    public:

        COcvBlurParam() : CWorkflowTaskParam(){}

        void        setParamMap(const UMapString& paramMap) override
        {
            m_ksize.width = std::stoi(paramMap.at("kSizeWidth"));
            m_ksize.height = std::stoi(paramMap.at("kSizeHeight"));
            m_anchor.x = std::stoi(paramMap.at("anchorX"));
            m_anchor.y = std::stoi(paramMap.at("anchorY"));
            m_borderType = std::stoi(paramMap.at("borderType"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("kSizeWidth", std::to_string(m_ksize.width)));
            map.insert(std::make_pair("kSizeHeight", std::to_string(m_ksize.height)));
            map.insert(std::make_pair("anchorX", std::to_string(m_anchor.x)));
            map.insert(std::make_pair("anchorY", std::to_string(m_anchor.y)));
            map.insert(std::make_pair("borderType", std::to_string(m_borderType)));
            return map;
        }

    public:

        int         m_ddepth = -1;
        cv::Size    m_ksize = cv::Size(5, 5);
        cv::Point   m_anchor = cv::Point(-1, -1);
        int         m_borderType = cv::BORDER_DEFAULT;
};

class COcvBlurParamFactory: public CTaskParamFactory
{
    public:

        COcvBlurParamFactory()
        {
            m_name = "ocv_blur";
        }

        WorkflowTaskParamPtr create()
        {
            return std::make_shared<COcvBlurParam>();
        }
};

//--------------------//
//----- COcvBlur -----//
//--------------------//
class COcvBlur : public C2dImageTask
{
    public:

        COcvBlur() : C2dImageTask()
        {
        }
        COcvBlur(const std::string name, const std::shared_ptr<COcvBlurParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<COcvBlurParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
            auto pParam = std::dynamic_pointer_cast<COcvBlurParam>(m_pParam);

            if (pInput == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

            if (pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgDst;
            CMat imgSrc = pInput->getImage();
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                cv::blur(imgSrc, imgDst, pParam->m_ksize, pParam->m_anchor, pParam->m_borderType);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }

            endTaskRun();
            applyGraphicsMask(imgSrc, imgDst, 0);
            emit m_signalHandler->doProgress();

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            emit m_signalHandler->doProgress();
        }
};

class COcvBlurFactory : public CTaskFactory
{
    public:

        COcvBlurFactory()
        {
            m_info.m_name = "ocv_blur";
            m_info.m_shortDescription = QObject::tr("This process smoothes your image with a box filter.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Image processing/Image filtering").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "Smooth,Blur,Isotropic,Filter,Gaussian";
            m_info.m_docLink = "https://docs.opencv.org/4.7.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pBoxFilterParam = std::dynamic_pointer_cast<COcvBlurParam>(pParam);
            if(pBoxFilterParam != nullptr)
                return std::make_shared<COcvBlur>(m_info.m_name, pBoxFilterParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pBoxFilterParam = std::make_shared<COcvBlurParam>();
            assert(pBoxFilterParam != nullptr);
            return std::make_shared<COcvBlur>(m_info.m_name, pBoxFilterParam);
        }
};

#endif // COCVBLUR_H
