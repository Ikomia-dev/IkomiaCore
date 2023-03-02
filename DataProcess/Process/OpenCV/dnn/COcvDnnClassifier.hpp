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

#ifndef COCVDNNCLASSIFIER_HPP
#define COCVDNNCLASSIFIER_HPP

#include "COcvDnnProcess.h"
#include "Graphics/CGraphicsLayer.h"
#include "Task/CClassificationTask.h"

//----------------------------------//
//----- COcvDnnClassifierParam -----//
//----------------------------------//
class COcvDnnClassifierParam: public COcvDnnProcessParam
{
    public:

        enum NetworkType {ALEXNET, GOOGLENET, INCEPTION, VGG, RESNET};

        COcvDnnClassifierParam() : COcvDnnProcessParam()
        {
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
            m_netType = std::stoi(paramMap.at("networkType"));
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            paramMap.insert(std::make_pair("networkType", std::to_string(m_netType)));
            return paramMap;
        }

    public:

        int m_netType = NetworkType::VGG;
};

//-----------------------------//
//----- COcvDnnClassifier -----//
//-----------------------------//
class COcvDnnClassifier: public COcvDnnProcess, public CClassificationTask
{
    public:

        COcvDnnClassifier() : COcvDnnProcess(), CClassificationTask()
        {
        }
        COcvDnnClassifier(const std::string& name, const std::shared_ptr<COcvDnnClassifierParam>& pParam)
            : COcvDnnProcess(), CClassificationTask(name)
        {
            m_pParam = std::make_shared<COcvDnnClassifierParam>(*pParam);
        }

        size_t      getProgressSteps() override
        {
            return 3;
        }
        int         getNetworkInputSize() const override
        {
            int size = 224;
            auto pParam = std::dynamic_pointer_cast<COcvDnnClassifierParam>(m_pParam);

            switch(pParam->m_netType)
            {
                case COcvDnnClassifierParam::ALEXNET: size = 224; break;
                case COcvDnnClassifierParam::GOOGLENET: size = 224; break;
                case COcvDnnClassifierParam::INCEPTION: size = 224; break;
                case COcvDnnClassifierParam::VGG: size = 300; break;
                case COcvDnnClassifierParam::RESNET: size = 224; break;
            }
            return size;
        }
        std::vector<std::string> getOutputsNames() const override
        {
            // Return empty list as we only want the last layer
            return std::vector<std::string>();
        }

        void        globalInputChanged(bool bNewSequence) override
        {
            setNewInputState(bNewSequence);
        }

        void        run() override
        {
            beginTaskRun();

            auto imgInputPtr = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            if(imgInputPtr == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid input image", __func__, __FILE__, __LINE__);

            auto paramPtr = std::dynamic_pointer_cast<COcvDnnClassifierParam>(m_pParam);
            if (paramPtr == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if (imgInputPtr->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Source image is empty", __func__, __FILE__, __LINE__);

            CMat imgOrigin = imgInputPtr->getImage();
            std::vector<cv::Mat> dnnOutputs;
            CMat imgSrc;

            //Need color image as input
            if(imgOrigin.channels() < 3)
                cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
            else
                imgSrc = imgOrigin;

            emit m_signalHandler->doProgress();

            try
            {
                if(m_net.empty() || paramPtr->m_bUpdate)
                {
                    m_net = readDnn(paramPtr);
                    if(m_net.empty())
                        throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

                    paramPtr->m_bUpdate = false;
                    readClassNames(paramPtr->m_labelsFile);
                }

                double inferTime = 0.0;
                if (isWholeImageClassification())
                {
                    inferTime = forward(imgSrc, dnnOutputs, paramPtr);
                    manageWholeImageOutput(dnnOutputs[0]);
                }
                else
                {
                    auto objects = getInputObjects();
                    for (size_t i=0; i<objects.size(); ++i)
                    {
                        auto subImage = getObjectSubImage(objects[i]);
                        inferTime += forward(subImage, dnnOutputs, paramPtr);
                        manageObjectOutput(dnnOutputs[0], objects[i]);
                    }
                }
                emit m_signalHandler->doProgress();

                m_customInfo.clear();
                m_customInfo.push_back(std::make_pair("Inference time (ms)", std::to_string(inferTime)));
                endTaskRun();
                emit m_signalHandler->doProgress();
            }
            catch(std::exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
            }
        }

        void        manageWholeImageOutput(cv::Mat &dnnOutput)
        {
            //Sort the 1 x n matrix of probabilities
            cv::Mat sortedIdx;
            cv::sortIdx(dnnOutput, sortedIdx, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
            std::vector<std::string> classes;
            std::vector<std::string> confidences;

            for(int i=0; i<sortedIdx.cols; ++i)
            {
                int classId = sortedIdx.at<int>(i);
                std::string className = classId < (int)m_classNames.size() ? m_classNames[classId] : "unknown " + std::to_string(classId);
                classes.push_back(className);
                confidences.push_back(std::to_string(dnnOutput.at<float>(classId)));
            }
            setWholeImageResults(classes, confidences);
        }

        void        manageObjectOutput(cv::Mat &dnnOutput, const ProxyGraphicsItemPtr &objectPtr)
        {
            //Sort the 1 x n matrix of probabilities
            cv::Mat sortedIdx;
            cv::sortIdx(dnnOutput, sortedIdx, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
            int classId = sortedIdx.at<int>(0, 0);
            double confidence = dnnOutput.at<float>(classId);
            addObject(objectPtr, classId, confidence);
        }
};

//------------------------------------//
//----- COcvDnnClassifierFactory -----//
//------------------------------------//
class COcvDnnClassifierFactory : public CTaskFactory
{
    public:

        COcvDnnClassifierFactory()
        {
            m_info.m_name = "ocv_dnn_classification";
            m_info.m_description = QObject::tr("This process gives the possibility to launch inference from already trained networks for classification purpose (CAFFE, TENSORFLOW, DARKNET and TORCH)).").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Deep neural network").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "cnn,dnn,deep,neural,network,classification";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/d6/d0f/group__dnn.html";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pDerivedParam = std::dynamic_pointer_cast<COcvDnnClassifierParam>(pParam);
            if(pDerivedParam != nullptr)
                return std::make_shared<COcvDnnClassifier>(m_info.m_name, pDerivedParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pDerivedParam = std::make_shared<COcvDnnClassifierParam>();
            assert(pDerivedParam != nullptr);
            return std::make_shared<COcvDnnClassifier>(m_info.m_name, pDerivedParam);
        }
};

#endif // COCVDNNCLASSIFIER_HPP
