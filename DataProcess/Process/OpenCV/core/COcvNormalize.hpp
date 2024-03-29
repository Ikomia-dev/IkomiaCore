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

#ifndef COCVNORMALIZE_HPP
#define COCVNORMALIZE_HPP

#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"

//-------------------------//
//----- COcvNormalizeParam -----//
//-------------------------//
class COcvNormalizeParam: public CWorkflowTaskParam
{
    public:

        COcvNormalizeParam() : CWorkflowTaskParam(){}

        void        setParamMap(const UMapString& paramMap) override
        {
            m_alpha = std::stod(paramMap.at("alpha"));
            m_beta = std::stod(paramMap.at("beta"));
            m_norm_type = std::stoi(paramMap.at("norm_type"));
            m_dtype = std::stoi(paramMap.at("dtype"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("alpha", std::to_string(m_alpha)));
            map.insert(std::make_pair("beta", std::to_string(m_beta)));
            map.insert(std::make_pair("norm_type", std::to_string(m_norm_type)));
            map.insert(std::make_pair("dtype", std::to_string(m_dtype)));
            return map;
        }

    public:

        double  	m_alpha = 1;
        double  	m_beta = 0;
        int         m_norm_type = cv::NORM_L2;
        int         m_dtype = -1;
};

//--------------------//
//----- COcvNormalize -----//
//--------------------//
class COcvNormalize : public C2dImageTask
{
    public:

        COcvNormalize() : C2dImageTask()
        {
        }
        COcvNormalize(const std::string name, const std::shared_ptr<COcvNormalizeParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<COcvNormalizeParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pParam = std::dynamic_pointer_cast<COcvNormalizeParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgDst;
            CMat imgSrc = pInput->getImage();
            emit m_signalHandler->doProgress();

            try
            {
                cv::normalize(imgSrc, imgDst, pParam->m_alpha, pParam->m_beta, pParam->m_norm_type, pParam->m_dtype);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }
            emit m_signalHandler->doProgress();

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            applyInputGraphicsMask(1, 0, 0, MaskMode::MERGE_SOURCE);
            endTaskRun();
            emit m_signalHandler->doProgress();
        }
};

class COcvNormalizeFactory : public CTaskFactory
{
    public:

        COcvNormalizeFactory()
        {
            m_info.m_name = "ocv_normalize";
            m_info.m_shortDescription = QObject::tr("This process Normalize your image around vertical, horizontal or both axes.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Core functionality/Operations on arrays").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "Normalize";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pNormalizeParam = std::dynamic_pointer_cast<COcvNormalizeParam>(pParam);
            if(pNormalizeParam != nullptr)
                return std::make_shared<COcvNormalize>(m_info.m_name, pNormalizeParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pNormalizeParam = std::make_shared<COcvNormalizeParam>();
            assert(pNormalizeParam != nullptr);
            return std::make_shared<COcvNormalize>(m_info.m_name, pNormalizeParam);
        }
};

#endif // COCVNORMALIZE_HPP
