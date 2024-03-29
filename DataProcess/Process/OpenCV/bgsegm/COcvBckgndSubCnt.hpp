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

#ifndef COCVBCKGNDSUBCNT_HPP
#define COCVBCKGNDSUBCNT_HPP

#include "opencv2/bgsegm.hpp"
#include "Task/CVideoTask.h"

//---------------------------------//
//----- COcvBckgndSubCntParam -----//
//---------------------------------//
class COcvBckgndSubCntParam: public CWorkflowTaskParam
{
    public:

        COcvBckgndSubCntParam()
        {
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            m_minPixelStability = std::stoi(paramMap.at("minPixelStability"));
            m_maxPixelStability = std::stoi(paramMap.at("maxPixelStability"));
            m_bUseHistory = std::stoi(paramMap.at("isUseHistory"));
            m_bIsParallel = std::stoi(paramMap.at("isParallel"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("minPixelStability", std::to_string(m_minPixelStability)));
            map.insert(std::make_pair("maxPixelStability", std::to_string(m_maxPixelStability)));
            map.insert(std::make_pair("isUseHistory", std::to_string(m_bUseHistory)));
            map.insert(std::make_pair("isParallel", std::to_string(m_bIsParallel)));
            return map;
        }

    public:

        int     m_minPixelStability = 15;
        int     m_maxPixelStability = 15 * 60;
        bool  	m_bUseHistory = true;
        bool  	m_bIsParallel = true;
};

//----------------------------//
//----- COcvBckgndSubCnt -----//
//----------------------------//
class COcvBckgndSubCnt : public CVideoTask
{
    public:

        COcvBckgndSubCnt() : CVideoTask()
        {
            setOutputDataType(IODataType::IMAGE_BINARY, 0);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
        }
        COcvBckgndSubCnt(const std::string name, const std::shared_ptr<COcvBckgndSubCntParam>& pParam) : CVideoTask(name)
        {
            setOutputDataType(IODataType::IMAGE_BINARY, 0);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
            m_pParam = std::make_shared<COcvBckgndSubCntParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void    notifyVideoStart(int frameCount) override
        {
            Q_UNUSED(frameCount);

            auto pParam = std::dynamic_pointer_cast<COcvBckgndSubCntParam>(m_pParam);
            if(pParam == nullptr)
                return;

            m_pBgSubtractor = cv::bgsegm::createBackgroundSubtractorCNT(pParam->m_minPixelStability, pParam->m_bUseHistory, pParam->m_maxPixelStability, pParam->m_bIsParallel);
        }

        void    run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pParam = std::dynamic_pointer_cast<COcvBckgndSubCntParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgDst;
            CMat imgSrc = pInput->getImage();
            emit m_signalHandler->doProgress();

            try
            {
                if(m_pBgSubtractor.empty())
                    m_pBgSubtractor = cv::bgsegm::createBackgroundSubtractorCNT(pParam->m_minPixelStability, pParam->m_bUseHistory, pParam->m_maxPixelStability, pParam->m_bIsParallel);

                m_pBgSubtractor->apply(imgSrc, imgDst);
            }
            catch(std::exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
            }

            emit m_signalHandler->doProgress();

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            applyInputGraphicsMask(1, 0, 0, MaskMode::MASK_ONLY);
            forwardInputImage(0, 1);
            endTaskRun();
            emit m_signalHandler->doProgress();
        }

    private:

        cv::Ptr<cv::bgsegm::BackgroundSubtractorCNT>    m_pBgSubtractor;
};

//-----------------------------------//
//----- COcvBckgndSubCntFactory -----//
//-----------------------------------//
class COcvBckgndSubCntFactory : public CTaskFactory
{
    public:

        COcvBckgndSubCntFactory()
        {
            m_info.m_name = "ocv_bck_substractor_cnt";
            m_info.m_shortDescription = QObject::tr("This process removes background on video sequence based on counting algorithm.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Extra modules/Improved Background-Foreground Segmentation Methods").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "video,background,substractor,segmentation,binary,counting";
            m_info.m_authors = "Sagi Zeevi";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pDerivedParam = std::dynamic_pointer_cast<COcvBckgndSubCntParam>(pParam);
            if(pDerivedParam != nullptr)
                return std::make_shared<COcvBckgndSubCnt>(m_info.m_name, pDerivedParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pDerivedParam = std::make_shared<COcvBckgndSubCntParam>();
            assert(pDerivedParam != nullptr);
            return std::make_shared<COcvBckgndSubCnt>(m_info.m_name, pDerivedParam);
        }
};

#endif // COCVBCKGNDSUBCNT_HPP
