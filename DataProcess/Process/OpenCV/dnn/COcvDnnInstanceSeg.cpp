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

#include "COcvDnnInstanceSeg.h"
#include "Graphics/CGraphicsLayer.h"

//------------------------------------//
//----- COcvDnnSegmentationParam -----//
//------------------------------------//
COcvDnnInstanceSegParam::COcvDnnInstanceSegParam() : COcvDnnProcessParam()
{
}

void COcvDnnInstanceSegParam::setParamMap(const UMapString &paramMap)
{
    COcvDnnProcessParam::setParamMap(paramMap);
    m_netType = std::stoi(paramMap.at("networkType"));
    m_confidence = std::stod(paramMap.at("confidence"));
    m_maskThreshold = std::stod(paramMap.at("maskThreshold"));
}

UMapString COcvDnnInstanceSegParam::getParamMap() const
{
    auto paramMap = COcvDnnProcessParam::getParamMap();
    paramMap.insert(std::make_pair("networkType", std::to_string(m_netType)));
    paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    paramMap.insert(std::make_pair("maskThreshold", std::to_string(m_maskThreshold)));
    return paramMap;
}

//-------------------------------//
//----- COcvDnnSegmentation -----//
//-------------------------------//
COcvDnnInstanceSeg::COcvDnnInstanceSeg() : COcvDnnProcess(), CInstanceSegTask()
{
}

COcvDnnInstanceSeg::COcvDnnInstanceSeg(const std::string name, const std::shared_ptr<COcvDnnInstanceSegParam> &pParam)
    : COcvDnnProcess(), CInstanceSegTask(name)
{
    m_pParam = std::make_shared<COcvDnnInstanceSegParam>(*pParam);
}

size_t COcvDnnInstanceSeg::getProgressSteps()
{
    return 3;
}

int COcvDnnInstanceSeg::getNetworkInputSize() const
{
    int size = 299;
    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);

    switch(pParam->m_netType)
    {
        case COcvDnnInstanceSegParam::MASK_RCNN: size = 800; break;
    }
    return size;
}

double COcvDnnInstanceSeg::getNetworkInputScaleFactor() const
{
    double factor = 1.0;
    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);

    switch(pParam->m_netType)
    {
        case COcvDnnInstanceSegParam::MASK_RCNN: factor = 1.0; break;
    }
    return factor;
}

cv::Scalar COcvDnnInstanceSeg::getNetworkInputMean() const
{
    return cv::Scalar();
}

std::vector<cv::String> COcvDnnInstanceSeg::getOutputsNames() const
{
    auto outNames = COcvDnnProcess::getOutputsNames();

    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
    if(pParam->m_netType == COcvDnnInstanceSegParam::MASK_RCNN)
        outNames.push_back("detection_out_final");

    return outNames;
}

void COcvDnnInstanceSeg::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);

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
    manageOutput(netOutputs);
    emit m_signalHandler->doProgress();
    endTaskRun();
    emit m_signalHandler->doProgress();
}

void COcvDnnInstanceSeg::manageOutput(std::vector<cv::Mat> &netOutputs)
{
    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
    if(pParam->m_netType == COcvDnnInstanceSegParam::MASK_RCNN)
        manageMaskRCNNOutput(netOutputs);
}

void COcvDnnInstanceSeg::manageMaskRCNNOutput(std::vector<cv::Mat> &netOutputs)
{
    auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    int nbDetections = netOutputs[1].size[2];
    for(int n=0; n<nbDetections; ++n)
    {
        //Detected class
        int classIndex[4] = { 0, 0, n, 1 };
        size_t classId = (size_t)netOutputs[1].at<float>(classIndex);
        //Confidence
        int confidenceIndex[4] = { 0, 0, n, 2 };
        float confidence = netOutputs[1].at<float>(confidenceIndex);

        if(confidence > pParam->m_confidence)
        {
            //Bounding box
            int leftIndex[4] = { 0, 0, n, 3 };
            int topIndex[4] = { 0, 0, n, 4 };
            int rightIndex[4] = { 0, 0, n, 5 };
            int bottomIndex[4] = { 0, 0, n, 6 };
            float left = netOutputs[1].at<float>(leftIndex) * imgSrc.cols;
            float top = netOutputs[1].at<float>(topIndex) * imgSrc.rows;
            float right = netOutputs[1].at<float>(rightIndex) * imgSrc.cols;
            float bottom = netOutputs[1].at<float>(bottomIndex) * imgSrc.rows;
            float width = right - left + 1;
            float height = bottom - top + 1;

            //Extract mask
            cv::Mat objMask(netOutputs[0].size[2], netOutputs[0].size[3], CV_32F, netOutputs[0].ptr<float>(n, classId));
            //Resize to the size of the box
            cv::resize(objMask, objMask, cv::Size(width, height), cv::INTER_LINEAR);
            //Apply thresholding to get the pixel wise mask
            cv::Mat objMaskBinary = (objMask > pParam->m_maskThreshold);
            objMaskBinary.convertTo(objMaskBinary, CV_8U);
            cv::Mat mask(imgSrc.rows, imgSrc.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat roi(mask, cv::Rect(left, top, width, height));
            objMaskBinary.copyTo(roi);

            addInstance(n, CInstanceSegmentation::ObjectType::THING, classId, confidence, left, top, width, height, mask);
        }
    }
}
