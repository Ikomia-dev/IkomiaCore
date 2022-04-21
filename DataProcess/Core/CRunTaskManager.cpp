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

#include "CRunTaskManager.h"
#include "IO/CImageIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CVideoIO.h"
#include "Main/CoreTools.hpp"

CRunTaskManager::CRunTaskManager()
{
}

void CRunTaskManager::setCfg(std::map<std::string, std::string> *pCfg)
{
    m_pCfg = pCfg;
}

void CRunTaskManager::run(const WorkflowTaskPtr &pTask, const std::string inputName)
{
    m_bStop = false;

    if(pTask == nullptr)
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Invalid task").toStdString(), __func__, __FILE__, __LINE__);

    switch(pTask->getType())
    {
        case CWorkflowTask::Type::GENERIC:
        case CWorkflowTask::Type::DNN_TRAIN:
            pTask->run();
            break;
        case CWorkflowTask::Type::IMAGE_PROCESS_2D:
            runImageProcess2D(pTask);
            break;
        case CWorkflowTask::Type::IMAGE_PROCESS_3D:
            pTask->run();
            break;
        case CWorkflowTask::Type::VIDEO:
            runVideoProcess(pTask);
            break;
    }
    manageOutputs(pTask, inputName);
}

void CRunTaskManager::stop(const WorkflowTaskPtr &taskPtr)
{
    m_bStop = true;
    if(taskPtr)
        taskPtr->stop();
}

void CRunTaskManager::runImageProcess2D(const WorkflowTaskPtr &taskPtr)
{
    //Thread safety -> scoped lock for all inputs/outputs
    //Access through CObjectLocker<CWorkflowTaskIO> ioLock(*ioPtr);
    //auto inputLocks = taskPtr->createInputScopedLocks();
    //auto outputLocks = taskPtr->createOutputScopedLocks();
    const std::set<IODataType> volumeTypes = {IODataType::VOLUME, IODataType::VOLUME_LABEL, IODataType::VOLUME_BINARY};
    bool batchMode = std::stoi(m_pCfg->at("BatchMode"));

    if(taskPtr->hasInput(volumeTypes) &&
        ((taskPtr->isActionFlagEnable(CWorkflowTask::ActionFlag::APPLY_VOLUME)) || batchMode == true))
    {
        //Get volume inputs
        auto imageInputs = taskPtr->getInputs(volumeTypes);
        if(imageInputs.size() == 0)
            return;

        //Check volume size
        CMat firstVolume = std::static_pointer_cast<CImageIO>(imageInputs[0])->getData();
        for(size_t i=1; i<imageInputs.size(); ++i)
        {
            if(imageInputs[i]->isDataAvailable())
            {
                CMat volume = std::static_pointer_cast<CImageIO>(imageInputs[i])->getData();
                if(firstVolume.size != volume.size)
                    throw CException(CoreExCode::INVALID_SIZE, QObject::tr("Different volume dimensions").toStdString(), __func__, __FILE__, __LINE__);
            }
        }

        //Run process on each 2D images
        std::vector<CMat> volumes;
        for(size_t i=0; i<firstVolume.getNbStacks() && m_bStop==false; ++i)
        {
            //Set the current image from the 3D input to process
            for(size_t j=0; j<imageInputs.size(); ++j)
                std::static_pointer_cast<CImageIO>(imageInputs[j])->setCurrentImage(i);

            //Run process
            taskPtr->run();

            if(i == 0)
            {
                //Get volume outputs and allocate 3D CMat
                auto volumeOutputs = taskPtr->getOutputs(volumeTypes);
                for(size_t j=0; j<volumeOutputs.size(); ++j)
                {
                    auto ouputPtr = std::static_pointer_cast<CImageIO>(volumeOutputs[j]);
                    volumes.emplace_back(CMat((int)firstVolume.getNbRows(), (int)firstVolume.getNbCols(), (int)firstVolume.getNbStacks(), ouputPtr->getImage().type()));
                }
            }

            //Insert 2D image into output 3D CMat
            for(size_t j=0; j<volumes.size(); ++j)
            {
                auto pOutput = std::static_pointer_cast<CImageIO>(taskPtr->getOutput(j));
                volumes[j].setPlane(i, pOutput->getImage());
            }
        }

        //Set final output of task
        for(size_t i=0; i<volumes.size(); ++i)
        {
            auto pOutput = std::static_pointer_cast<CImageIO>(taskPtr->getOutput(i));
            pOutput->setImage(volumes[i]);
        }
    }
    else
    {
        taskPtr->run();
    }
}

void CRunTaskManager::runVideoProcess(const WorkflowTaskPtr& taskPtr)
{
    const std::set<IODataType> videoTypes = {
        IODataType::VIDEO, IODataType::VIDEO_LABEL, IODataType::VIDEO_BINARY,
        IODataType::LIVE_STREAM, IODataType::LIVE_STREAM_LABEL, IODataType::LIVE_STREAM_BINARY
    };

    //Get video inputs
    auto videoInputs = taskPtr->getInputs(videoTypes);
    if(videoInputs.size() == 0)
        return;

    //Run process
    taskPtr->run();
}

void CRunTaskManager::manageOutputs(const WorkflowTaskPtr &taskPtr, const std::string& inputName)
{
    // Auto-save outputs if mode is enabled
    if (taskPtr->isAutoSave())
    {
        if (std::stoi(m_pCfg->at("WholeVideo")))
            saveVideoOutputs(taskPtr, inputName);
        else
            taskPtr->saveOutputs(inputName);
    }
}

void CRunTaskManager::saveVideoOutputs(const WorkflowTaskPtr &taskPtr, const std::string& inputName)
{
    assert(m_pCfg);
    bool bImageSequence = false;
    bool bEmbedGraphics = std::stoi(m_pCfg->at("GraphicsEmbedded"));
    const std::set<IODataType> videoTypes = {IODataType::VIDEO, IODataType::VIDEO_LABEL, IODataType::VIDEO_BINARY};

    //Get video inputs
    auto videoInputs = taskPtr->getInputs(videoTypes);
    if(videoInputs.size() == 0)
        return;

    //Get video outputs
    auto videoOutputs = taskPtr->getOutputs(videoTypes);
    if(videoOutputs.size() == 0)
        return;

    for(size_t i=0; i<videoInputs.size(); ++i)
    {
        auto inputPtr = std::static_pointer_cast<CVideoIO>(videoInputs[i]);

        //Check source type
        auto infoPtr = std::static_pointer_cast<CDataVideoInfo>(inputPtr->getDataInfo());
        if(infoPtr && infoPtr->m_sourceType == CDataVideoBuffer::IMAGE_SEQUENCE)
            bImageSequence = true;
    }

    for(size_t i=0; i<videoOutputs.size(); ++i)
    {
        // Set save path
        std::string outPath;
        auto outputPtr = std::static_pointer_cast<CVideoIO>(videoOutputs[i]);
        Utils::File::createDirectory(taskPtr->getOutputFolder());

        if(bImageSequence)
            outPath = taskPtr->getOutputFolder() + inputName + "_" + std::to_string(i+1) + "_%04d.png";
        else
            outPath = taskPtr->getOutputFolder() + inputName + "_" + std::to_string(i+1) + ".avi";

        outputPtr->setVideoPath(outPath);
    }

    auto infoPtr = std::static_pointer_cast<CDataVideoInfo>(videoInputs[0]->getDataInfo());

    InputOutputVect graphicsOutputs;
    if(bEmbedGraphics)
        graphicsOutputs = taskPtr->getOutputs({IODataType::OUTPUT_GRAPHICS});

    for(size_t i=0; i<videoOutputs.size(); ++i)
    {
        auto outputPtr = std::static_pointer_cast<CVideoIO>(videoOutputs[i]);
        CMat img = outputPtr->getImage();

        // Start video write if needed
        if(outputPtr->isWriteStarted() == false)
            outputPtr->startVideoWrite(img.getNbCols(), img.getNbRows(), infoPtr->m_frameCount, infoPtr->m_fps, infoPtr->m_fourcc, m_timeout);

        if(bEmbedGraphics && graphicsOutputs.size() > 0)
        {
            for(size_t j=0; j<graphicsOutputs.size(); ++j)
            {
                auto graphicsOutPtr = std::static_pointer_cast<CGraphicsOutput>(graphicsOutputs[j]);
                if(graphicsOutPtr->getImageIndex() == (int)i)
                {
                    CGraphicsConversion graphicsConv((int)img.getNbCols(), (int)img.getNbRows());
                    for(auto it : graphicsOutPtr->getItems())
                        it->insertToImage(img, graphicsConv, false, false);
                }
            }
        }
        outputPtr->writeImage(img);
    }
}

