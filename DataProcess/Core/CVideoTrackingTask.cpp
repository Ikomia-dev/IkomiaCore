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

#include "CVideoTrackingTask.h"
#include "Graphics/CGraphicsLayer.h"
#include "Graphics/CGraphicsRectangle.h"

CVideoTrackingTask::CVideoTrackingTask() : CVideoTask()
{
    setOutputDataType(IODataType::IMAGE_BINARY, 0);
    addOutput(std::make_shared<CImageIO>());
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

CVideoTrackingTask::CVideoTrackingTask(const std::string& name) : CVideoTask(name)
{
    setOutputDataType(IODataType::IMAGE_BINARY, 0);
    addOutput(std::make_shared<CImageIO>());
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CBlobMeasureIO>());
}

void CVideoTrackingTask::setRoiToTrack()
{
    auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
    if(pGraphicsInput == nullptr)
        return;

    auto items = pGraphicsInput->getItems();
    if(items.size() == 0)
    {
        Utils::print(QObject::tr("Tracker requires rectangular object").toStdString(), QtCriticalMsg);
        return;
    }

    if(items.size() > 1)
        Utils::print(QObject::tr("Tracker can only track one rectangular object").toStdString(), QtWarningMsg);

    auto pItem = std::dynamic_pointer_cast<CProxyGraphicsRect>(items.back());
    if(pItem == nullptr)
    {
        Utils::print(QObject::tr("Tracker can only track rectangular object").toStdString(), QtCriticalMsg);
        return;
    }

    m_trackedRect = cv::Rect2d(pItem->m_x, pItem->m_y, pItem->m_width, pItem->m_height);
    m_bInitRoi = true;
}

void CVideoTrackingTask::manageOutputs()
{
    forwardInputImage(0, 1);

    //Generate binary mask
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));

    if(pOutput)
    {
        CMat imgSrc = pInput->getImage();
        CMat imgDst(imgSrc.rows, imgSrc.cols, CV_8UC1, cv::Scalar(0));
        CMat roi(imgDst, m_trackedRect);
        roi.setTo(cv::Scalar(255));
        pOutput->setImage(imgDst);
    }

    //Graphics output
    auto pGraphicsOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(2));
    pGraphicsOutput->setNewLayer("Trackerlayer");
    pGraphicsOutput->setImageIndex(1);

    //Create rectangle graphics of bbox
    auto graphicsObj = pGraphicsOutput->addRectangle(m_trackedRect.x, m_trackedRect.y, m_trackedRect.width, m_trackedRect.height);

    //Tracked rectangle coordinates
    auto pMeasureOutput = std::dynamic_pointer_cast<CBlobMeasureIO>(getOutput(3));
    if(pMeasureOutput)
    {
        CMeasure bboxMeasure(CMeasure::BBOX, QObject::tr("Tracked ROI").toStdString());
        pMeasureOutput->setObjectMeasure(0, CObjectMeasure(bboxMeasure, {(double)m_trackedRect.x, (double)m_trackedRect.y, (double)m_trackedRect.width, (double)m_trackedRect.height}, graphicsObj->getId(), ""));
    }
}

void CVideoTrackingTask::notifyVideoStart(int frameCount)
{
    Q_UNUSED(frameCount);
    m_bVideoStarted = true;
    m_bInitRoi = false;
}

void CVideoTrackingTask::notifyVideoEnd()
{
    m_bVideoStarted = false;
}
