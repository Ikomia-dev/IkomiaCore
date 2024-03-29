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

#ifndef COCVBCKGNDSUBKNN_HPP
#define COCVBCKGNDSUBKNN_HPP

#include "opencv2/video/background_segm.hpp"
#include "Task/CVideoTask.h"

//---------------------------------//
//----- COcvBckgndSubKnnParam -----//
//---------------------------------//
class COcvBckgndSubKnnParam: public CWorkflowTaskParam
{
    public:

        COcvBckgndSubKnnParam()
        {
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            m_history = std::stoi(paramMap.at("history"));
            m_distanceThreshold = std::stod(paramMap.at("distanceThreshold"));
            m_bDetectShadow = std::stoi(paramMap.at("isShadowDetect"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("history", std::to_string(m_history)));
            map.insert(std::make_pair("distanceThreshold", std::to_string(m_distanceThreshold)));
            map.insert(std::make_pair("isShadowDetect", std::to_string(m_bDetectShadow)));
            return map;
        }

    public:

        int     m_history = 500;
        double  m_distanceThreshold = 400.0;
        bool    m_bDetectShadow = true;
};

//----------------------------//
//----- COcvBckgndSubKnn -----//
//----------------------------//
class COcvBckgndSubKnn : public CVideoTask
{
    public:

        COcvBckgndSubKnn() : CVideoTask()
        {
            setOutputDataType(IODataType::IMAGE_BINARY, 0);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
        }
        COcvBckgndSubKnn(const std::string name, const std::shared_ptr<COcvBckgndSubKnnParam>& pParam) : CVideoTask(name)
        {
            setOutputDataType(IODataType::IMAGE_BINARY, 0);
            addOutput(std::make_shared<CImageIO>());
            setOutputColorMap(1, 0, {{255, 0, 0}});
            m_pParam = std::make_shared<COcvBckgndSubKnnParam>(*pParam);
        }

        size_t     getProgressSteps() override
        {
            return 3;
        }

        void    notifyVideoStart(int frameCount) override
        {
            Q_UNUSED(frameCount);

            auto pParam = std::dynamic_pointer_cast<COcvBckgndSubKnnParam>(m_pParam);
            if(pParam == nullptr)
                return;

            m_pBgSubtractor = cv::createBackgroundSubtractorKNN(pParam->m_history, pParam->m_distanceThreshold, pParam->m_bDetectShadow);
        }

        void    run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pParam = std::dynamic_pointer_cast<COcvBckgndSubKnnParam>(m_pParam);

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
                    m_pBgSubtractor = cv::createBackgroundSubtractorKNN(pParam->m_history, pParam->m_distanceThreshold, pParam->m_bDetectShadow);

                m_pBgSubtractor->apply(imgSrc, imgDst);
            }
            catch(std::exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
            }

            emit m_signalHandler->doProgress();
            forwardInputImage(0, 1);

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            endTaskRun();
            emit m_signalHandler->doProgress();
        }

    private:

        cv::Ptr<cv::BackgroundSubtractorKNN>    m_pBgSubtractor;
};

//-----------------------------------//
//----- COcvBckgndSubKnnFactory -----//
//-----------------------------------//
class COcvBckgndSubKnnFactory : public CTaskFactory
{
    public:

        COcvBckgndSubKnnFactory()
        {
            m_info.m_name = "ocv_bck_substractor_knn";
            m_info.m_shortDescription = QObject::tr("This process removes background on video sequence with K-Nearest Neighbors algorithm.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Video analysis/Motion analysis").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "video,background,substractor,nearest,neighbor,segmentation,binary";
            m_info.m_authors = "Zoran Zivkovic and Ferdinand Van Der Heijden";
            m_info.m_article = "Efficient adaptive density estimation per image pixel for the task of background subtraction";
            m_info.m_journal = "Pattern recognition letters, 27(7):773–780";
            m_info.m_year = 2006;
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/db/d88/classcv_1_1BackgroundSubtractorKNN.html";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pDerivedParam = std::dynamic_pointer_cast<COcvBckgndSubKnnParam>(pParam);
            if(pDerivedParam != nullptr)
                return std::make_shared<COcvBckgndSubKnn>(m_info.m_name, pDerivedParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pDerivedParam = std::make_shared<COcvBckgndSubKnnParam>();
            assert(pDerivedParam != nullptr);
            return std::make_shared<COcvBckgndSubKnn>(m_info.m_name, pDerivedParam);
        }
};

#endif // COCVBCKGNDSUBKNN_HPP
