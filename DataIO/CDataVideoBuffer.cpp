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

#include <future>
#include <thread>
#include "CDataVideoBuffer.h"
#include "UtilsTools.hpp"
#include "CTimer.hpp"

CDataVideoBuffer::CDataVideoBuffer()
{
}

CDataVideoBuffer::CDataVideoBuffer(const std::string &path)
{
    setVideoPath(path);
}

CDataVideoBuffer::CDataVideoBuffer(const std::string &path, int frameCount)
{
    setVideoPath(path);
    m_nbFrames = frameCount;
}

CDataVideoBuffer::~CDataVideoBuffer()
{
    close();
}

void CDataVideoBuffer::close()
{
    stopRead();
    stopWrite();
    m_reader.release();
}

void CDataVideoBuffer::openVideo()
{
    switch(m_type)
    {
        case NONE:
            return;
        case OPENNI_STREAM:
            if(!m_reader.open(cv::CAP_OPENNI))
            {
                close();
                throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to open video file or camera", __func__, __FILE__, __LINE__);
            }
            break;

        case ID_STREAM:
            openStreamFromId(std::stoi(m_path));
            break;

        case IP_STREAM:
            openStreamFromIP(m_path);
            break;

        case VIDEO:
        case PATH_STREAM:
        case IMAGE_SEQUENCE:
        case GSTREAMER_PIPE:
            openStreamFromPath(m_path);
            break;
    }

    if(m_type != OPENNI_STREAM)
    {
        if(!m_reader.isOpened())
        {
            close();
            throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to open video file or camera", __func__, __FILE__, __LINE__);
        }

        // Parameter not available for image sequence
        m_nbFrames = m_reader.get(cv::CAP_PROP_FRAME_COUNT);
        m_fps = m_reader.get(cv::CAP_PROP_FPS);

        if(m_fps <= 0)
            m_fps = 25;

        m_width = m_reader.get(cv::CAP_PROP_FRAME_WIDTH);
        m_height = m_reader.get(cv::CAP_PROP_FRAME_HEIGHT);
        // Default video compress mode format
        //m_fourcc = m_reader.get(CV_CAP_PROP_FOURCC);
    }
}

void CDataVideoBuffer::startRead(int timeout)
{
    if(m_bStopRead)
        m_bStopRead = false;
    else
        return;

    // If not already open, try open video
    if(!m_reader.isOpened())
        openVideo();

    m_bError = false;
    m_timeout = timeout;
    m_queueRead.activate();
    m_readFuture = Utils::async([this]{ updateRead(); });
}

void CDataVideoBuffer::stopRead()
{
    waitReadFinished(m_timeout);

    if(m_type != OPENNI_STREAM)
        m_reader.set(cv::CAP_PROP_POS_FRAMES, 0);

    m_currentPos = 0;
}

void CDataVideoBuffer::pauseRead()
{
    m_bStopRead = true;
    if (isReadStarted())
        m_readFuture.wait();
}

void CDataVideoBuffer::clearRead()
{
    m_queueRead.clear();
}

void CDataVideoBuffer::startWrite(int width, int height, int nbFrames, int fps, int fourcc, int timeout)
{
    if(m_bStopWrite)
        m_bStopWrite = false;

    m_bError = false;
    m_width = width;
    m_height = height;
    m_nbFrames = nbFrames;
    m_fps = fps;
    m_fourcc = fourcc;
    m_timeout = timeout;
    isWritable(); // may throw

    if (m_timeout != -1)
        m_queueWrite.setTimeout(m_timeout);

    m_queueWrite.activate();
    m_writeFuture = Utils::async([this]{ updateWrite(); });
}

void CDataVideoBuffer::startStreamWrite(int width, int height, int fps, int fourcc, int timeout)
{
    if(m_bStopWrite)
        m_bStopWrite = false;

    m_bError = false;
    m_width = width;
    m_height = height;
    m_fps = fps;
    m_fourcc = fourcc;
    m_timeout = timeout;
    isWritable(); // may throw

    if (m_timeout != -1)
        m_queueWrite.setTimeout(m_timeout);

    m_queueWrite.activate();
    m_writeFuture = Utils::async([this]{ updateStreamWrite(); });
}

void CDataVideoBuffer::stopWrite()
{
    waitWriteFinished(m_timeout);
}

CMat CDataVideoBuffer::read()
{
    if(m_bError)
    {
        close();
        throw CException(CoreExCode::PROCESS_CANCELLED, m_lastErrorMsg, __func__, __FILE__, __LINE__);
    }

    CMat img;
    try
    {
        if(m_currentPos < m_nbFrames)
        {
            m_queueRead.pop(img);
            m_currentPos++;
        }
        else if(isReadStarted())
        {
            m_queueRead.pop(img);
            m_currentPos++;
        }
    }
    catch(std::exception& e)
    {
        close();
        throw CException(CoreExCode::TIMEOUT_REACHED, e.what() + tr("No more images to read.").toStdString(), __func__, __FILE__, __LINE__);
    }
    return img;
}

void CDataVideoBuffer::write(CMat image)
{
    if(m_bError)
    {
        close();
        throw CException(CoreExCode::PROCESS_CANCELLED, m_lastErrorMsg, __func__, __FILE__, __LINE__);
    }

    if(image.empty() == false)
    {
        // VideoWriter can only write 3-channels images
        assert(image.channels() == 3);

        while(m_queueWrite.size() >= m_queueSize)
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

        m_queueWrite.push(image.clone());
    }
}

CMat CDataVideoBuffer::snapshot(int pos)
{
    if(!isReadOpened())
        openVideo();

    pauseRead();

    if(m_type != OPENNI_STREAM)
    {
        if(pos != -1 && m_currentPos != pos)
        {
            m_currentPos = pos;
            m_reader.set(cv::CAP_PROP_POS_FRAMES, pos);
        }
        else if(pos > m_nbFrames)
        {
            m_currentPos = 0;
            m_reader.set(cv::CAP_PROP_POS_FRAMES, m_currentPos);
        }
        else if(m_currentPos >= m_nbFrames)
        {
            m_currentPos = 0;
            m_reader.set(cv::CAP_PROP_POS_FRAMES, m_currentPos);
        }
    }

    CMat image;
    bool bRet = m_reader.grab();
    if(bRet == false)
    {
        close();
        throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to snap image", __func__, __FILE__, __LINE__);
    }

    bRet = m_reader.retrieve(image, m_mode);
    if(bRet == false)
    {
        close();
        throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to snap image", __func__, __FILE__, __LINE__);
    }

    m_currentPos++;
    return image;
}

void CDataVideoBuffer::waitWriteFinished(int timeout)
{
    if (m_writeFuture.valid())
    {
        if (timeout == -1)
            m_writeFuture.wait();
        else
            m_writeFuture.wait_for(std::chrono::milliseconds(timeout));
    }
    m_bStopWrite = true;
    m_queueWrite.cancel();
    m_queueWrite.clear();
}

void CDataVideoBuffer::waitReadFinished(int timeout)
{
    if (m_readFuture.valid())
    {
        if (timeout == -1)
            m_readFuture.wait();
        else
            m_readFuture.wait_for(std::chrono::milliseconds(timeout));
    }
    m_bStopRead = true;
    m_queueRead.cancel();
    m_queueRead.clear();
}

bool CDataVideoBuffer::hasReadImage() const
{
    if (isReadStarted())
    {
        Utils::CTimer timer;
        timer.start();
        while(m_queueRead.size() == 0 && timer.get_total_elapsed_ms() <= m_timeout);
        return m_queueRead.size() > 0;
    }
    else
        return m_queueRead.size() > 0;
}

bool CDataVideoBuffer::isReadStarted() const
{
    using namespace std::chrono_literals;
    bool isValid = m_readFuture.valid();
    if (!isValid)
        return false;
    else
    {
        auto status = m_readFuture.wait_for(0ms);
        return status != std::future_status::ready;
    }
}

bool CDataVideoBuffer::isReadOpened() const
{
    return m_reader.isOpened();
}

bool CDataVideoBuffer::isReadMode() const
{
    if(isStreamSource())
        return true;
    else if(m_type == IMAGE_SEQUENCE)
        return Utils::File::isFileSequenceExist(m_path);
    else if(m_type == VIDEO)
        return Utils::File::isFileExist(m_path);
    else
        return false;
}

bool CDataVideoBuffer::hasWriteImage() const
{
    return m_queueWrite.size() > 0;
}

bool CDataVideoBuffer::isWriteStarted() const
{
    using namespace std::chrono_literals;
    bool isValid = m_writeFuture.valid();
    if (!isValid)
        return false;
    else
    {
        auto status = m_writeFuture.wait_for(0ms);
        return status != std::future_status::ready;
    }
}

void CDataVideoBuffer::setVideoPath(const std::string& path)
{
    m_path = path;
    init();

    if(isReadMode())
    {
        // Query video information by opening video and then closing reader (necessary for windows in order to avoid sharing violation)
        openVideo();
        m_reader.release();
    }
}

void CDataVideoBuffer::setQueueSize(size_t queueSize)
{
    m_queueSize = queueSize;
}

void CDataVideoBuffer::setPosition(int pos)
{
    if(m_type != IMAGE_SEQUENCE && pos >= m_nbFrames && pos != 0)
    {
        close();
        throw CException(DataIOExCode::VIDEO_WRONG_IMG_NUMBERS, "Invalid frame number", __func__, __FILE__, __LINE__);
    }
    m_mutex.lock();
    m_currentPos = pos;
    m_reader.set(cv::CAP_PROP_POS_FRAMES, pos);
    clearRead();
    m_mutex.unlock();
}

void CDataVideoBuffer::setFPS(int fps)
{
    m_fps = fps;
}

void CDataVideoBuffer::setSize(int width, int height)
{
    m_width = width;
    m_height = height;
}

void CDataVideoBuffer::setFrameCount(int nb)
{
    m_nbFrames = nb;
}

void CDataVideoBuffer::setMode(int mode)
{
    m_mode = mode;
}

void CDataVideoBuffer::setFourCC(int code)
{
    m_fourcc = code;
}

std::string CDataVideoBuffer::getCurrentPath() const
{
    return m_path;
}

std::string CDataVideoBuffer::getSourceName() const
{
    std::string name;
    switch(m_type)
    {
        case NONE:
            name = "NoSource";
            break;
        case OPENNI_STREAM:
            name = "OpenNI_stream";
            break;
        case ID_STREAM:
        case PATH_STREAM:
            name = "Camera";
            break;
        case IP_STREAM:
            name = "IP_camera";
            break;
        case GSTREAMER_PIPE:
            name = "GStreamer";
            break;
        case VIDEO:
        case IMAGE_SEQUENCE:
            name = Utils::File::getFileNameWithoutExtension(m_path);
            break;
    }
    return name;
}

int CDataVideoBuffer::getFrameCount() const
{
    return m_nbFrames;
}

int CDataVideoBuffer::getCurrentPos() const
{
    return m_currentPos;
}

int CDataVideoBuffer::getFPS() const
{
    return m_fps;
}

int CDataVideoBuffer::getWidth() const
{
    return m_width;
}

int CDataVideoBuffer::getHeight() const
{
    return m_height;
}

int CDataVideoBuffer::getCodec() const
{
    return m_fourcc;
}

CDataVideoBuffer::Type CDataVideoBuffer::getSourceType() const
{
    return m_type;
}

void CDataVideoBuffer::updateRead()
{
    int grabCount = 0;
    Utils::CTimer timer;
    timer.start();

    while(m_bStopRead == false && timer.get_total_elapsed_ms() <= m_timeout)
    {
        if(m_nbFrames != -1 && grabCount == m_nbFrames)
        {
            m_bStopRead = true;
            break;
        }

        if(m_queueRead.size() >= m_queueSize)
        {
            // Queue is full, we have to wait
            timer.start();
            continue;
        }

        try
        {
            if(!m_reader.grab())
            {
                m_lastErrorMsg = "OpenCV VideoCapture::grab() function failed.";
                Utils::print(m_lastErrorMsg, QtDebugMsg);
                // Sleep to avoid failing too many times
                std::this_thread::sleep_for(std::chrono::milliseconds(m_timeout/5));
                continue;
            }

            CMat frame;
            if(!m_reader.retrieve(frame, m_mode))
            {
                m_lastErrorMsg = "OpenCV VideoCapture::retrieve() function failed.";
                Utils::print(m_lastErrorMsg, QtDebugMsg);
                // Sleep to avoid failing too many times
                std::this_thread::sleep_for(std::chrono::milliseconds(m_timeout/5));
                continue;
            }

            CMat dst;
            grabCount++;

            if(frame.channels() > 1)
                cv::cvtColor(frame, dst, cv::COLOR_BGR2RGB);
            else
                dst = frame.clone();

            if(isStreamSource())
            {
                //For stream, we replace the current frame
                if(m_queueRead.size() > 0)
                    m_queueRead.pop();

                m_queueRead.push(dst);
            }
            else
            {
                // For video files, we buffer frames in the queue
                m_queueRead.push(dst);
            }
            //Reset timeout when success
            timer.start();
        }
        catch(std::exception& e)
        {
            m_lastErrorMsg = e.what();
            Utils::print(m_lastErrorMsg, QtDebugMsg);
        }
    }
    m_bError = timer.get_total_elapsed_ms() > m_timeout;
}

void CDataVideoBuffer::updateWrite()
{
    if(m_type == CDataVideoBuffer::IMAGE_SEQUENCE)
        writeImageSequenceThread();
    else
        writeVideoThread();
}

void CDataVideoBuffer::updateStreamWrite()
{
    // Test codec (default is Motion JPEG)
    checkFourcc();

    //Default API backend is FFMPEG
    cv::VideoWriter writer;
    writer.open(m_path, m_fourcc, (double)m_fps, cv::Size(m_width, m_height));
    if(!writer.isOpened())
    {
        m_bError = true;
        m_lastErrorMsg = "Failed to open video writer:" + m_path;
        return;
    }

    while(m_bStopWrite == false && m_bError == false)
    {
        try
        {
            cv::Mat img = m_queueWrite.pop();
            if(!img.empty())
            {
                m_mutex.lock();
                writer.write(img);
                m_mutex.unlock();
            }
        }
        catch(CException& e)
        {
            m_mutex.unlock();
            if (e.getCode() == CoreExCode::TIMEOUT_REACHED && m_timeout == -1)
            {
                // If no timeout is set, just log event
                m_queueWrite.activate();
                Utils::print(e.what(), QtDebugMsg);
            }
            else
            {
                m_bError = true;
                m_lastErrorMsg = std::string("Stream writing ended: ") + e.what();
            }
        }
        catch(std::exception& e)
        {
            m_mutex.unlock();
            m_bError = true;
            m_lastErrorMsg = std::string("Stream writing ended: ") + e.what();
        }
    }
}

void CDataVideoBuffer::writeImageSequenceThread()
{
    int count = 0;
    while(m_bStopWrite == false && count < m_nbFrames && m_bError == false)
    {
        try
        {
            cv::Mat img = m_queueWrite.pop();
            if(!img.empty())
            {
                auto filename = Utils::File::getPathFromPattern(m_path, count);
                cv::imwrite(filename, img);
                count++;
            }
        }
        catch(CException& e)
        {
            if (e.getCode() == CoreExCode::TIMEOUT_REACHED && m_timeout == -1)
            {
                // If no timeout is set, just log event
                m_queueWrite.activate();
                Utils::print(e.what(), QtDebugMsg);
            }
            else
            {
                m_bError = true;
                m_lastErrorMsg = std::string("Image sequence writing ended: ") + e.what();
            }
        }
        catch(std::exception& e)
        {
            m_bError = true;
            m_lastErrorMsg = std::string("Image sequence writing ended: ") + e.what();
        }
    }
}

void CDataVideoBuffer::writeVideoThread()
{
    // Test codec (default is Motion JPEG)
    checkFourcc();

    //Default API backend is FFMPEG
    cv::VideoWriter writer;
    writer.open(m_path, m_fourcc, (double)m_fps, cv::Size(m_width, m_height));

    if(!writer.isOpened())
    {
        m_bError = true;
        m_lastErrorMsg = "Failed to open video writer:" + m_path;
        return;
    }

    int count = 0;
    while(m_bStopWrite == false && count < m_nbFrames && m_bError == false)
    {
        try
        {
            cv::Mat img = m_queueWrite.pop();
            if(!img.empty())
            {
                m_mutex.lock();
                writer.write(img);
                count++;
                m_mutex.unlock();
            }
        }
        catch(CException& e)
        {
            m_mutex.unlock();
            if (e.getCode() == CoreExCode::TIMEOUT_REACHED && m_timeout == -1)
            {
                // If no timeout is set, just log event
                m_queueWrite.activate();
                Utils::print(e.what(), QtDebugMsg);
            }
            else
            {
                m_bError = true;
                m_lastErrorMsg = "Video writing ended: " + std::to_string(count) + "/" + std::to_string(m_nbFrames) + " images written.";
                m_lastErrorMsg += " Error occured:" + std::string(e.what());
            }
        }
        catch(std::exception& e)
        {
            m_mutex.unlock();
            m_bError = true;
            m_lastErrorMsg = "Video writing ended: " + std::to_string(count) + "/" + std::to_string(m_nbFrames) + " images written.";
            m_lastErrorMsg += " Error occured:" + std::string(e.what());
        }
    }
    writer.release();
}

bool CDataVideoBuffer::isStreamSource() const
{
    return m_type == OPENNI_STREAM || m_type == ID_STREAM || m_type == IP_STREAM || m_type == PATH_STREAM;
}

bool CDataVideoBuffer::isNumber(const std::string& s) const
{
    // Works for only positive integers
    return  !s.empty() &&
            std::find_if(s.begin(), s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

void CDataVideoBuffer::openStreamFromId(int id)
{
#if defined(_WIN32)
    auto api = cv::VideoCaptureAPIs::CAP_DSHOW;
#elif defined(Q_OS_MACOS)
    auto api = cv::VideoCaptureAPIs::CAP_ANY;
#elif (defined(linux) || defined(__linux__) || defined(__linux))
    auto api = cv::VideoCaptureAPIs::CAP_V4L2;
#endif

    if(!m_reader.open(id, api))
    {
        close();
        throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to open video file or camera", __func__, __FILE__, __LINE__);
    }
}

void CDataVideoBuffer::openStreamFromIP(const std::string& ipAdress)
{
    if(!m_reader.open(ipAdress))
    {
        close();
        throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to open video file or camera", __func__, __FILE__, __LINE__);
    }
}

void CDataVideoBuffer::openStreamFromPath(const std::string& path)
{
#if defined(_WIN32)
    auto api = cv::VideoCaptureAPIs::CAP_ANY;
#elif defined(Q_OS_MACOS)
    auto api = cv::VideoCaptureAPIs::CAP_ANY;
#elif (defined(linux) || defined(__linux__) || defined(__linux))
    auto api = cv::VideoCaptureAPIs::CAP_V4L2;
#endif

    if(m_type == PATH_STREAM)
    {
        // /dev/video* -> open with V4L backend
        if(!m_reader.open(path, api))
        {
            close();
            throw CException(DataIOExCode::FILE_NOT_EXISTS, std::string("Failed to open camera: ") + path, __func__, __FILE__, __LINE__);
        }
    }
    else
    {
        // video file: open with default backend (FFMPEG)
        // NB: boost::filesystem::exists(path) has been removed but maybe needed for a special case??
        if(!m_reader.open(path))
        {
            close();
            throw CException(DataIOExCode::FILE_NOT_EXISTS, std::string("Failed to open video file or camera: ") + path, __func__, __FILE__, __LINE__);
        }
    }
}

void CDataVideoBuffer::init()
{
    int pathId;

    if(isNumber(m_path))
        pathId = std::stoi(m_path);
    else
        pathId = -1;

    // Must be in range [0 7] because cv::CAP_OPENNI = 900 and display varies from 0 (depth map) to 7 (IR)
    if(pathId >= cv::CAP_OPENNI)
    {
        // Set if BGR or DEPTHMAP mode
        int mode = pathId%cv::CAP_OPENNI;
        setMode(mode);
        m_type = OPENNI_STREAM;
    }
    else
    {
        if(pathId >= 0)
            m_type = ID_STREAM;
        else
        {
            if(m_path.find("http://") != std::string::npos)
                m_type = IP_STREAM;
            else if(m_path.find("%") != std::string::npos)
                m_type = IMAGE_SEQUENCE;
            else
            {
                boost::filesystem::path p(m_path);
                if(p.extension().empty())
                {
                    // In no extension -> /dev/video* -> open with V4L backend
                    m_type = PATH_STREAM;
                }
                else
                {
                    // video file
                    m_type = VIDEO;
                }
            }
        }
    }
}

void CDataVideoBuffer::isWritable()
{
    if(m_type == CDataVideoBuffer::IMAGE_SEQUENCE)
        return;

    // Test codec (default is Motion JPEG)
    checkFourcc();

    cv::VideoWriter writer;
    if(writer.open(m_path, m_fourcc, (double)m_fps, cv::Size(m_width, m_height)) == false)
    {
        close();
        throw CException(DataIOExCode::FILE_NOT_EXISTS, "Failed to open video writer:" + m_path, __func__, __FILE__, __LINE__);
    }
}

void CDataVideoBuffer::checkFourcc()
{
    // Test codec (default is Motion JPEG)
    if(m_type == CDataVideoBuffer::IMAGE_SEQUENCE)
        m_fourcc = 0;
    else if(m_fourcc == -1)
        m_fourcc = cv::VideoWriter::fourcc('M','P','E','G');
}

#include "moc_CDataVideoBuffer.cpp"
