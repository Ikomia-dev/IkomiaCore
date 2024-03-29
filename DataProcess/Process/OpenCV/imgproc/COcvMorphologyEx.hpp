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

#ifndef COCVMORPHOLOGYEX_HPP
#define COCVMORPHOLOGYEX_HPP

#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"

//---------------------------------//
//----- COcvMorphologyExParam -----//
//---------------------------------//
class COcvMorphologyExParam: public CWorkflowTaskParam
{
    public:

        COcvMorphologyExParam() : CWorkflowTaskParam(){}

        void setParamMap(const UMapString& paramMap) override
        {
            m_op = std::stoi(paramMap.at("op"));
            m_kernel = std::stoi(paramMap.at("kernel"));
            m_kernelSize = cv::Size(std::stoi(paramMap.at("kernelWidth")), std::stoi(paramMap.at("kernelHeight")));
            m_anchor = cv::Point(std::stoi(paramMap.at("anchorX")), std::stoi(paramMap.at("anchorY")));
            m_iterations = std::stoi(paramMap.at("iterations"));
            m_borderType = std::stoi(paramMap.at("borderType"));
            m_borderValue = std::stod(paramMap.at("borderValue"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("op", std::to_string(m_op)));
            map.insert(std::make_pair("kernel", std::to_string(m_kernel)));
            map.insert(std::make_pair("kernelWidth", std::to_string(m_kernelSize.width)));
            map.insert(std::make_pair("kernelHeight", std::to_string(m_kernelSize.height)));
            map.insert(std::make_pair("anchorX", std::to_string(m_anchor.x)));
            map.insert(std::make_pair("anchorY", std::to_string(m_anchor.y)));
            map.insert(std::make_pair("iterations", std::to_string(m_iterations)));
            map.insert(std::make_pair("borderType", std::to_string(m_borderType)));
            map.insert(std::make_pair("borderValue", std::to_string(m_borderValue)));

            return map;
        }

    public:

        int         m_op = cv::MorphTypes::MORPH_ERODE;
        int         m_kernel = cv::MorphShapes::MORPH_RECT;
        cv::Size    m_kernelSize = cv::Size(3, 3);
        cv::Point   m_anchor = cv::Point(-1, -1);
        int         m_iterations = 1;
        int         m_borderType = cv::BORDER_CONSTANT;
        double      m_borderValue = 0;
};

//-------------------------//
//----- COcvMorphologyEx -----//
//-------------------------//
class COcvMorphologyEx : public C2dImageTask
{
    public:

        COcvMorphologyEx() : C2dImageTask()
        {
        }
        COcvMorphologyEx(const std::string name, const std::shared_ptr<COcvMorphologyExParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<COcvMorphologyExParam>(*pParam);
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
            auto pParam = std::dynamic_pointer_cast<COcvMorphologyExParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgDst;
            CMat imgSrc = pInput->getImage();
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                auto kernel = cv::getStructuringElement(pParam->m_kernel, pParam->m_kernelSize, pParam->m_anchor);
                cv::morphologyEx(imgSrc, imgDst, pParam->m_op, kernel, pParam->m_anchor, pParam->m_iterations, pParam->m_borderType, pParam->m_borderValue);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }

            endTaskRun();
            applyGraphicsMask(imgSrc, imgDst, 0);
            emit m_signalHandler->doProgress();

            if(pInput->getDataType() == IODataType::IMAGE_BINARY)
                getOutput(0)->setDataType(IODataType::IMAGE_BINARY);

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            emit m_signalHandler->doProgress();
        }
};

class COcvMorphologyExFactory : public CTaskFactory
{
    public:

        COcvMorphologyExFactory()
        {
            m_info.m_name = "ocv_morphology_ex";
            m_info.m_shortDescription = QObject::tr("Performs advanced morphological transformations. \
                                            The function cv::morphologyEx can perform advanced \
                                            morphological transformations using an erosion and dilation \
                                            as basic operations. Any of the operations can be done in-place. \
                                            In case of multi-channel images, each channel is processed \
                                            independently.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Image processing/Image filtering").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "Morphology, dilate, erode, tophat";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pMorphologyExParam = std::dynamic_pointer_cast<COcvMorphologyExParam>(pParam);
            if(pMorphologyExParam != nullptr)
                return std::make_shared<COcvMorphologyEx>(m_info.m_name, pMorphologyExParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pMorphologyExParam = std::make_shared<COcvMorphologyExParam>();
            assert(pMorphologyExParam != nullptr);
            return std::make_shared<COcvMorphologyEx>(m_info.m_name, pMorphologyExParam);
        }
};

#endif // COCVMORPHOLOGYEX_HPP
