/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the Ikomia API libraries.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef CDATAVIDEOBUFFER_HPP
#define CDATAVIDEOBUFFER_HPP

#include <QObject>
#include <QtConcurrent>
#include <thread>
#include <mutex>
#include "DataIOGlobal.hpp"
#include "Containers/CQueue.hpp"
#include "Data/CMat.hpp"


/** @cond INTERNAL */

class DATAIOSHARED_EXPORT CDataVideoBuffer: public QObject
{
    Q_OBJECT

    public:

        enum Type { NONE, VIDEO, OPENNI_STREAM, ID_STREAM, IP_STREAM, PATH_STREAM, IMAGE_SEQUENCE, GSTREAMER_PIPE };

        CDataVideoBuffer();
        CDataVideoBuffer(const std::string& path);
        CDataVideoBuffer(const std::string& path, size_t frameCount);
        ~CDataVideoBuffer();

        void            close();
        void            openVideo();

        void            startRead(int timeout);
        void            stopRead();
        void            pauseRead();
        void            clearRead();

        void            startWrite(int width, int height, size_t nbFrames, size_t fps=25, int fourcc=-1, int timeout=-1);
        void            startStreamWrite(int width, int height, size_t fps=25, int fourcc=-1, int timeout=-1);
        void            stopWrite(int timeout);

        CMat            read();
        void            write(CMat image);
        CMat            grab();
        CMat            grab(size_t pos);

        void            waitWriteFinished(int timeout);
        void            waitReadFinished(int timeout);

        bool            hasReadImage() const;
        bool            isReadStarted() const;
        bool            isReadOpened() const;
        bool            isReadMode() const;

        bool            hasWriteImage() const;
        bool            isWriteStarted() const;

        void            setVideoPath(const std::string& path);
        void            setQueueSize(size_t queueSize);
        void            setPosition(size_t pos);
        void            setFPS(size_t fps);
        void            setSize(int width, int height);
        void            setFrameCount(size_t nb);
        void            setMode(int mode);
        void            setFourCC(int code);

        std::string     getCurrentPath() const;
        std::string     getSourceName() const;
        size_t          getFrameCount() const;
        size_t          getCurrentPos() const;
        size_t          getFPS() const;
        int             getWidth() const;
        int             getHeight() const;
        int             getCodec() const;
        Type            getSourceType() const;

        void            openStreamFromId(int id);
        void            openStreamFromIP(const std::string& ipAdress);
        void            openStreamFromPath(const std::string& path);

    private:

        void            init();
        void            initFourccList();

        void            updateRead();
        void            updateWrite();
        void            updateStreamWrite();

        void            writeImageSequenceThread();
        void            writeVideoThread();

        bool            isStreamSource() const;
        bool            isNumber(const std::string& s) const;
        void            isWritable();

        bool            checkVideoWriterConfig();

        int             getBestFourcc(int backend);
        std::string     getBackendName(int backend) const;
        int             getBackendFromName(const std::string& name) const;

    private:

        CQueue<CMat>            m_queueRead;
        CQueue<CMat>            m_queueWrite;
        cv::VideoCapture        m_reader;
        std::string             m_path;
        std::string             m_lastErrorMsg;
        size_t                  m_queueSize = 128;
        std::mutex              m_mutex;
        std::future<void>       m_readFuture;
        std::future<void>       m_writeFuture;
        std::atomic_bool        m_bStopRead{true};
        std::atomic_bool        m_bStopWrite{true};
        std::atomic_bool        m_bError{false};
        Type                    m_type = VIDEO;
        size_t                  m_fps = 0;
        size_t                  m_nbFrames = 0;
        size_t                  m_currentPos = 0;
        int                     m_width = 0;
        int                     m_height = 0;
        int                     m_fourcc = -1;
        int                     m_mode = 0;
        int                     m_writeBackend = cv::CAP_FFMPEG;
        int                     m_timeout = 5000; // in milliseconds
        std::vector<int>        m_orderedFourccList;
        std::map<int, std::string>  m_fourccNames;

};

using CDataVideoBufferPtr = std::unique_ptr<CDataVideoBuffer>;

/** @endcond */

#endif // CDATAVIDEOBUFFER_HPP
