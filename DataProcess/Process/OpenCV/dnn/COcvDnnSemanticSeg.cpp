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

#include "COcvDnnSemanticSeg.h"
#include "Graphics/CGraphicsLayer.h"

//-----------------------------------//
//----- COcvDnnSemanticSegParam -----//
//-----------------------------------//
COcvDnnSemanticSegParam::COcvDnnSemanticSegParam() : COcvDnnProcessParam()
{
}

void COcvDnnSemanticSegParam::setParamMap(const UMapString &paramMap)
{
    COcvDnnProcessParam::setParamMap(paramMap);
    m_netType = std::stoi(paramMap.at("networkType"));
    m_confidence = std::stod(paramMap.at("confidence"));
    m_maskThreshold = std::stod(paramMap.at("maskThreshold"));
}

UMapString COcvDnnSemanticSegParam::getParamMap() const
{
    auto paramMap = COcvDnnProcessParam::getParamMap();
    paramMap.insert(std::make_pair("networkType", std::to_string(m_netType)));
    paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    paramMap.insert(std::make_pair("maskThreshold", std::to_string(m_maskThreshold)));
    return paramMap;
}

//------------------------------//
//----- COcvDnnSemanticSeg -----//
//------------------------------//
COcvDnnSemanticSeg::COcvDnnSemanticSeg(): COcvDnnProcess(), CSemanticSegTask()
{
}

COcvDnnSemanticSeg::COcvDnnSemanticSeg(const std::string name, const std::shared_ptr<COcvDnnSemanticSegParam> &pParam)
    : COcvDnnProcess(), CSemanticSegTask(name)
{
    m_pParam = std::make_shared<COcvDnnSemanticSegParam>(*pParam);
}

size_t COcvDnnSemanticSeg::getProgressSteps()
{
    return 3;
}

int COcvDnnSemanticSeg::getNetworkInputSize() const
{
    int size = 299;
    auto pParam = std::dynamic_pointer_cast<COcvDnnSemanticSegParam>(m_pParam);

    switch(pParam->m_netType)
    {
        case COcvDnnSemanticSegParam::ENET: size = 512; break;
        case COcvDnnSemanticSegParam::FCN: size = 500; break;
        case COcvDnnSemanticSegParam::UNET: size = 572; break;
    }
    return size;
}

double COcvDnnSemanticSeg::getNetworkInputScaleFactor() const
{
    double factor = 1.0;
    auto pParam = std::dynamic_pointer_cast<COcvDnnSemanticSegParam>(m_pParam);

    switch(pParam->m_netType)
    {
        case COcvDnnSemanticSegParam::ENET: factor = 1.0/255.0; break;
        case COcvDnnSemanticSegParam::FCN: factor = 1.0; break;
        case COcvDnnSemanticSegParam::UNET: factor = 1.0; break;
    }
    return factor;
}

cv::Scalar COcvDnnSemanticSeg::getNetworkInputMean() const
{
    return cv::Scalar();
}

std::vector<cv::String> COcvDnnSemanticSeg::getOutputsNames() const
{
    auto outNames = COcvDnnProcess::getOutputsNames();
    return outNames;
}

void COcvDnnSemanticSeg::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<COcvDnnSemanticSegParam>(m_pParam);

    if (pInput == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

    if (pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if (pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    CMat imgSrc = pInput->getImage();
    std::vector<cv::Mat> netOutputs;
    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn(pParam);
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            readClassNames(pParam->m_labelsFile);
            pParam->m_bUpdate = false;
        }
        forward(imgSrc, netOutputs, pParam);
    }
    catch(std::exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    //readClassNames();
    manageOutput(netOutputs[0]);
    emit m_signalHandler->doProgress();
    endTaskRun();
    emit m_signalHandler->doProgress();
}

void COcvDnnSemanticSeg::manageOutput(const cv::Mat &netOutput)
{
    const int classes = netOutput.size[1];
    const int rows = netOutput.size[2];
    const int cols = netOutput.size[3];

    cv::Mat labelImg = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1, netOutput.data);

    for(int c=1; c<classes; ++c)
    {
        for(int i=0; i<rows; ++i)
        {
            const float* pScore = netOutput.ptr<float>(0, c, i);
            uint8_t* pMaxClass = labelImg.ptr<uint8_t>(i);
            float* pMaxVal = maxVal.ptr<float>(i);

            for(int j=0; j<cols; ++j)
            {
                if(pScore[j] > pMaxVal[j])
                {
                    pMaxVal[j] = pScore[j];
                    pMaxClass[j] = (uint8_t)c;
                }
            }
        }
    }

    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    cv::resize(labelImg, labelImg, pInput->getImage().size(), 0, 0, cv::INTER_LINEAR);
    setMask(labelImg);
}
