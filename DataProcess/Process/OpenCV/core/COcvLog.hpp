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

#ifndef COCVLOG_HPP
#define COCVLOG_HPP


#include "Task/C2dImageTask.h"

//-------------------//
//----- COcvLog -----//
//-------------------//
class COcvLog : public C2dImageTask
{
    public:

        COcvLog() : C2dImageTask()
        {
        }
        COcvLog(const std::string name, const std::shared_ptr<CWorkflowTaskParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<CWorkflowTaskParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void run() override
        {
            beginTaskRun();

            if(getInputCount() < 2)
                throw CException(CoreExCode::INVALID_PARAMETER, "Not enough inputs", __func__, __FILE__, __LINE__);

            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            if(pInput == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty images", __func__, __FILE__, __LINE__);

            emit m_signalHandler->doProgress();

            CMat imgDst;
            CMat imgSrc = pInput->getImage();

            try
            {
                CMat imgTmp;
                if(imgSrc.depth() != CV_32F || imgSrc.depth() != CV_64F)
                    imgSrc.convertTo(imgTmp, CV_32F);
                else
                    imgTmp = imgSrc;

                cv::log(imgTmp, imgDst);
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

class COcvLogFactory : public CTaskFactory
{
    public:

        COcvLogFactory()
        {
            m_info.m_name = "ocv_log";
            m_info.m_description = QObject::tr("Calculates the natural logarithm of every element of the input image").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Core functionality/Operations on arrays").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "log, logarithm";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/d2/de8/group__core__array.html#ga937ecdce4679a77168730830a955bea7";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            if(pParam != nullptr)
                return std::make_shared<COcvLog>(m_info.m_name, pParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pParam = std::make_shared<CWorkflowTaskParam>();
            assert(pParam != nullptr);
            return std::make_shared<COcvLog>(m_info.m_name, pParam);
        }
};

#endif // COCVLOG_HPP
