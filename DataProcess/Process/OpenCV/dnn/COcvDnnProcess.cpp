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

#include "COcvDnnProcess.h"

//----------------------------//
//----- COcvDnnCoreParam -----//
//----------------------------//
COcvDnnCoreParam::COcvDnnCoreParam() : CWorkflowTaskParam()
{
}

void COcvDnnCoreParam::setParamMap(const UMapString& paramMap)
{
    m_backend = static_cast<cv::dnn::Backend>(std::stoi(paramMap.at("backend")));
    m_target = static_cast<cv::dnn::Target>(std::stoi(paramMap.at("target")));
}

UMapString COcvDnnCoreParam::getParamMap() const
{
    UMapString map;
    map.insert(std::make_pair("backend", std::to_string(m_backend)));
    map.insert(std::make_pair("target", std::to_string(m_target)));
    return map;
}

//-------------------------------//
//----- COcvDnnProcessParam -----//
//-------------------------------//
COcvDnnProcessParam::COcvDnnProcessParam() : COcvDnnCoreParam()
{
}

void COcvDnnProcessParam::setParamMap(const UMapString &paramMap)
{
    COcvDnnCoreParam::setParamMap(paramMap);
    m_framework = std::stoi(paramMap.at("framework"));
    m_inputSize = std::stoi(paramMap.at("inputSize"));
    m_modelName = paramMap.at("modelName");
    m_datasetName = paramMap.at("datasetName");
    m_modelFile = paramMap.at("modelFile");
    m_structureFile = paramMap.at("structureFile");
    m_labelsFile = paramMap.at("labelsFile");
}

UMapString COcvDnnProcessParam::getParamMap() const
{
    UMapString map = COcvDnnCoreParam::getParamMap();
    map.insert(std::make_pair("framework", std::to_string(m_framework)));
    map.insert(std::make_pair("inputSize", std::to_string(m_inputSize)));
    map.insert(std::make_pair("modelName", m_modelName));
    map.insert(std::make_pair("datasetName", m_datasetName));
    map.insert(std::make_pair("modelFile", m_modelFile));
    map.insert(std::make_pair("structureFile", m_structureFile));
    map.insert(std::make_pair("labelsFile", m_labelsFile));
    return map;
}

//--------------------------//
//----- COcvDnnProcess -----//
//--------------------------//
COcvDnnProcess::COcvDnnProcess()
{
}

int COcvDnnProcess::getNetworkInputSize() const
{
    return 224;
}

double COcvDnnProcess::getNetworkInputScaleFactor() const
{
    return 1.0;
}

cv::Scalar COcvDnnProcess::getNetworkInputMean() const
{
    return cv::Scalar();
}

std::vector<cv::String> COcvDnnProcess::getOutputsNames() const
{
    std::vector<cv::String> names;
    std::vector<int> outLayers = m_net.getUnconnectedOutLayers();
    std::vector<cv::String> layersNames = m_net.getLayerNames();
    names.resize(outLayers.size());

    for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];

    return names;
}

cv::dnn::Net COcvDnnProcess::readDnn(const std::shared_ptr<COcvDnnProcessParam>& paramPtr)
{
    cv::dnn::Net net;

    if (paramPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid DNN parameter object", __func__, __FILE__, __LINE__);

    switch(paramPtr->m_framework)
    {
        case COcvDnnProcessParam::TENSORFLOW:
            net = cv::dnn::readNetFromTensorflow(paramPtr->m_modelFile, paramPtr->m_structureFile);
            break;
        case COcvDnnProcessParam::CAFFE:
            net = cv::dnn::readNetFromCaffe(paramPtr->m_structureFile, paramPtr->m_modelFile);
            break;
        case COcvDnnProcessParam::DARKNET:
            net = cv::dnn::readNetFromDarknet(paramPtr->m_structureFile, paramPtr->m_modelFile);
            break;
        case COcvDnnProcessParam::TORCH:
            net = cv::dnn::readNetFromTorch(paramPtr->m_modelFile, true);
            break;
        case COcvDnnProcessParam::ONNX:
            net = cv::dnn::readNetFromONNX(paramPtr->m_modelFile);
    }
    net.setPreferableBackend(paramPtr->m_backend);
    net.setPreferableTarget(paramPtr->m_target);
    //displayLayers(net);
    return net;
}

double COcvDnnProcess::forward(const CMat& imgSrc, std::vector<cv::Mat> &outputs, const std::shared_ptr<COcvDnnProcessParam> &paramPtr)
{
    int size = getNetworkInputSize();
    double scaleFactor = getNetworkInputScaleFactor();
    cv::Scalar mean = getNetworkInputMean();
    auto inputBlob = cv::dnn::blobFromImage(imgSrc, scaleFactor, cv::Size(size,size), mean, false, false);
    m_net.setInput(inputBlob);
    auto netOutNames = getOutputsNames();

    Utils::CTimer inferenceTime;
    inferenceTime.start();

    if (netOutNames.size() == 0)
        outputs.push_back(m_net.forward());
    else
        m_net.forward(outputs, netOutNames);

    auto t = inferenceTime.get_elapsed_ms();

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566    
    if(paramPtr->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
    {
        m_sign *= -1;
        m_bNewInput = false;
    }
    return t;
}

void COcvDnnProcess::displayLayers(const cv::dnn::Net& net)
{
    auto layerNames = net.getLayerNames();
    for(size_t i=0; i<layerNames.size(); ++i)
        Utils::print(layerNames[i], QtDebugMsg);
}

void COcvDnnProcess::setNewInputState(bool bNewSequence)
{
    m_bNewInput = bNewSequence;
}
