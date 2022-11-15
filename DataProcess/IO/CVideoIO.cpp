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

#include "CVideoIO.h"
#include "Data/CDataVideoInfo.h"
#include "CDataVideoIO.h"
#include "UtilsTools.hpp"
#include "Data/CDataConversion.h"

CVideoIO::CVideoIO() : CImageIO(IODataType::VIDEO, "CVideoIO")
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
}

CVideoIO::CVideoIO(IODataType data) : CImageIO(data, "CVideoIO")
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
}

CVideoIO::CVideoIO(IODataType data, const CMat &image) : CImageIO(data, image, "CVideoIO")
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
}

CVideoIO::CVideoIO(IODataType data, const CMat& image, const std::string& name) : CImageIO(data, image, name)
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
}

CVideoIO::CVideoIO(IODataType data, const std::string &name): CImageIO(data, name)
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
}

CVideoIO::CVideoIO(IODataType data, const std::string &name, const std::string &path): CImageIO(data, name)
{
    m_description = QObject::tr("Video with read/write capabilities.").toStdString();
    m_saveFormat = DataFileFormat::AVI;
    setVideoPath(path);
}

CVideoIO::CVideoIO(const CVideoIO &io) : CImageIO(io)
{
    m_frameIndex = io.m_frameIndex;
    m_frameIndexRead = io.m_frameIndexRead;
}

CVideoIO::CVideoIO(const CVideoIO&& io) : CImageIO(io)
{
    m_frameIndex = std::move(io.m_frameIndex);
    m_frameIndexRead = std::move(io.m_frameIndexRead);
}

CVideoIO &CVideoIO::operator=(const CVideoIO& io)
{
    CImageIO::operator=(io);
    m_frameIndex = io.m_frameIndex;
    m_frameIndexRead = io.m_frameIndexRead;
    return *this;
}

CVideoIO &CVideoIO::operator=(const CVideoIO&& io)
{
    CImageIO::operator=(io);
    m_frameIndex = std::move(io.m_frameIndex);
    m_frameIndexRead = std::move(io.m_frameIndexRead);
    return *this;
}

void CVideoIO::setVideoPath(const std::string& path)
{
    if (m_pVideoBuffer && m_pVideoBuffer->getCurrentPath() == path)
        return;

    std::string extension = Utils::File::extension(path);
    if(CDataVideoIO::isVideoFormat(extension, true))
    {
        // Videos
        m_pVideoBuffer = std::make_unique<CDataVideoBuffer>(path);
    }
    else
    {
        // Image sequence
        auto ret = CDataVideoIO::getImageSequenceInfo(path);
        m_pVideoBuffer = std::make_unique<CDataVideoBuffer>(ret.first, ret.second);
    }
}

void CVideoIO::setVideoPos(size_t pos)
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->setPosition(pos);        
    else
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);
}

void CVideoIO::setFrameToRead(size_t index)
{
    m_frameIndex = index;
}

void CVideoIO::setDataInfo(const CDataInfoPtr &infoPtr)
{
    m_infoPtr = infoPtr;
}

void CVideoIO::startVideo(int timeout)
{
    if(m_pVideoBuffer)
    {
        m_pVideoBuffer->startRead(timeout);
        m_frameIndexRead = -1;
    }
    else
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);
}

void CVideoIO::stopVideo()
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->stopRead();
    else
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);
}

void CVideoIO::startVideoWrite(int width, int height, int frames, int fps, int fourcc, int timeout)
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->startWrite(width, height, frames, fps, fourcc, timeout);
    else
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);
}

void CVideoIO::stopVideoWrite(int timeout)
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->stopWrite(timeout);
    else
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);
}

void CVideoIO::addVideoImage(const CMat& image)
{
    m_videoImgList.push_back(image);
}

void CVideoIO::writeImage(CMat image)
{
    if(m_pVideoBuffer == nullptr)
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);

    CMat tmp;

    // 8 bits unsigned only for OpenCV video writter
    int depth = image.depth();
    if (depth != CV_8U)
        CDataConversion::to8Bits(image, tmp);
    else
        tmp = image;

    if (image.channels() == 1)
        cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
    else
        cv::cvtColor(tmp, tmp, cv::COLOR_RGB2BGR);

    m_pVideoBuffer->write(tmp);
}

bool CVideoIO::hasVideo()
{
    return m_pVideoBuffer != nullptr;
}

CMat CVideoIO::getImage()
{
    if(m_pVideoBuffer && m_pVideoBuffer->hasReadImage())
    {
        // Do not read image from source multiple times if we get it already
        if(m_frameIndex != m_frameIndexRead)
        {
            m_image = m_pVideoBuffer->read();
            m_frameIndexRead = m_frameIndex;
        }
    }
    return m_image;
}

size_t CVideoIO::getVideoFrameCount() const
{
    if (m_pVideoBuffer)
        return m_pVideoBuffer->getFrameCount();
    else if (m_infoPtr)
    {
        auto videoInfoPtr = std::dynamic_pointer_cast<CDataVideoInfo>(m_infoPtr);
        if (videoInfoPtr)
            return videoInfoPtr->m_frameCount;
    }
    return 0;
}

std::vector<CMat> CVideoIO::getVideoImages() const
{
    return m_videoImgList;
}

std::string CVideoIO::getVideoPath() const
{
    if(!m_pVideoBuffer)
        return "";
    else
        return m_pVideoBuffer->getCurrentPath();
}

CMat CVideoIO::getSnapshot(size_t pos)
{
    if(m_pVideoBuffer == nullptr)
        throw CException(CoreExCode::NULL_POINTER, QObject::tr("Video buffer pointer is null.").toStdString(), __func__, __FILE__, __LINE__);

    return m_pVideoBuffer->grab(pos);
}

size_t CVideoIO::getCurrentPos() const
{
    if (m_pVideoBuffer)
        return m_pVideoBuffer->getCurrentPos();
    else if (m_infoPtr)
    {
        auto videoInfoPtr = std::dynamic_pointer_cast<CDataVideoInfo>(m_infoPtr);
        if (videoInfoPtr)
            return videoInfoPtr->m_currentPos;
    }
    return 0;
}

CMat CVideoIO::getStaticImage() const
{
    return m_image;
}

size_t CVideoIO::getUnitElementCount() const
{
    if(m_pVideoBuffer)
        return m_pVideoBuffer->getFrameCount();
    else
        return 1;
}

CDataInfoPtr CVideoIO::getDataInfo()
{
    if(m_pVideoBuffer)
    {
        m_infoPtr = std::make_shared<CDataVideoInfo>(m_pVideoBuffer->getFPS(),
                                                     m_pVideoBuffer->getFrameCount(),
                                                     m_pVideoBuffer->getCurrentPos(),
                                                     m_pVideoBuffer->getCodec());
        auto infoPtr = std::static_pointer_cast<CDataVideoInfo>(m_infoPtr);
        infoPtr->setFileName(m_pVideoBuffer->getCurrentPath());
        infoPtr->m_width = m_pVideoBuffer->getWidth();
        infoPtr->m_height = m_pVideoBuffer->getHeight();
        infoPtr->m_sourceType = m_pVideoBuffer->getSourceType();
    }
    return m_infoPtr;
}

std::vector<DataFileFormat> CVideoIO::getPossibleSaveFormats() const
{
    std::vector<DataFileFormat> formats = { DataFileFormat::AVI, DataFileFormat::MPEG };
    return formats;
}

bool CVideoIO::isDataAvailable() const
{
    bool bRet =  m_image.data != nullptr || m_videoImgList.size() > 0 ;
    if(m_pVideoBuffer)
        bRet = bRet || m_pVideoBuffer->isReadOpened();

    return bRet;
}

bool CVideoIO::isReadStarted() const
{
    if(m_pVideoBuffer)
        return m_pVideoBuffer->isReadStarted();

    return false;
}

bool CVideoIO::isWriteStarted() const
{
    if(m_pVideoBuffer)
        return m_pVideoBuffer->isWriteStarted();

    return false;
}

void CVideoIO::clearData()
{
    m_image.release();
}

void CVideoIO::copyStaticData(const WorkflowTaskIOPtr &ioPtr)
{
    CImageIO::copyStaticData(ioPtr);

    auto videoIoPtr = std::dynamic_pointer_cast<CVideoIO>(ioPtr);
    if (videoIoPtr)
    {
        auto infoPtr = videoIoPtr->getDataInfo();
        if (infoPtr)
            setDataInfo(infoPtr);
    }
}

void CVideoIO::waitWriteFinished(int timeout)
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->waitWriteFinished(timeout);
}

void CVideoIO::waitReadImage(int timeout) const
{
    if(m_pVideoBuffer)
        m_pVideoBuffer->waitReadFinished(timeout);
}

std::shared_ptr<CVideoIO> CVideoIO::clone() const
{
    return std::static_pointer_cast<CVideoIO>(cloneImp());
}

void CVideoIO::save()
{
    m_tempFiles.clear();
}

std::shared_ptr<CWorkflowTaskIO> CVideoIO::cloneImp() const
{
    return std::shared_ptr<CVideoIO>(new CVideoIO(*this));
}
