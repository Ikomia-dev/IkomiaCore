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

#ifndef CVIDEOIO_H
#define CVIDEOIO_H

#include "DataProcessGlobal.hpp"
#include "CImageIO.h"
#include "CDataVideoBuffer.h"

/**
 * @ingroup groupCore
 * @brief The CVideoIO class defines an input or output for a workflow task dedicated to video management.
 * @details This class is designed to handle video as input or output of a workflow task.
 * A CVideoIO instance can be added as input or output to a CWorkflowTask or derived object.
 * It is the base class to define input or output of a video processing task.
 * Several video output type can be defined according to the nature of the implemented algorithm.
 * Video processing task can output binary, labelled or standard video.
 * Source video can be either a file, an image sequence or a stream.
 * For Python, image data is a numpy array.
 */
class DATAPROCESSSHARED_EXPORT CVideoIO : public CImageIO
{
    public:

        /**
         * @brief Default constructor
         */
        CVideoIO();
        /**
         * @brief Constructs a CVideoIO instance with the given video output type.
         * @param data: see ::IODataType for details.
         */
        CVideoIO(IODataType data);
        /**
         * @brief Constructs a CVideoIO instance with the given video output type and the given name.
         * @param data: see ::IODataType for details.
         * @param image: CMat object for C++ and Numpy array for Python.
         */
        CVideoIO(IODataType data, const CMat& image);
        /**
         * @brief Constructs a CVideoIO instance with the given video output type, name and the given image.
         * @param data: see ::IODataType for details.
         * @param image: CMat object for C++ and Numpy array for Python.
         * @param name: input or output name.
         */
        CVideoIO(IODataType data, const CMat& image, const std::string& name);
        /**
         * @brief Constructs a CVideoIO instance with the given image output type and the name.
         * @param data: see ::IODataType for details.
         * @param name: input or output name.
         */
        CVideoIO(IODataType data, const std::string& name);
        /**
         * @brief Constructs a CVideoIO instance with the given image output type and the image path.
         * @param data: see ::IODataType for details.
         * @param name: input or output name.
         * @param path: path to video file.
         */
        CVideoIO(IODataType data, const std::string& name, const std::string& path);
        /**
         * @brief Copy constructor.
         */
        CVideoIO(const CVideoIO& io);
        /**
         * @brief Universal reference copy constructor.
         */
        CVideoIO(const CVideoIO&& io);

        virtual ~CVideoIO() = default;

        /**
         * @brief Assignement operator.
         */
        CVideoIO& operator=(const CVideoIO& io);
        /**
         * @brief Universal reference assignement operator.
         */
        CVideoIO& operator=(const CVideoIO&& io);

        /**
         * @brief Sets the source path of the video.
         * @param path: it can be a file path, an IP address or an internal camera reference.
         */
        void                setVideoPath(const std::string& path);
        /**
         * @brief Sets the current frame of the video.
         * @param pos: index of the frame.
         */
        void                setVideoPos(size_t pos);
        /**
         * @brief Set index of the next frame to read.
         * @param index: index of the frame.
         */
        void                setFrameToRead(size_t index);
        /**
         * @brief Set video info (width, height, fps...).
         * @param infoPtr: CDataInfo based object.
         */
        void                setDataInfo(const CDataInfoPtr& infoPtr);

        /**
         * @brief Gets the total number of frames for the video.
         * @return Number of frames (size_t).
         */
        size_t              getVideoFrameCount() const;
        /**
         * @brief Gets all images extracted from the video.
         * @return Vector of CMat (C++) or numpy array (Python).
         */
        std::vector<CMat>   getVideoImages() const;
        /**
         * @brief Gets the path to the source video.
         * @return Full path.
         */
        std::string         getVideoPath() const;
        /**
         * @brief Gets the image of a single frame.
         * @param pos: index of the frame.
         * @return Image as CMat for C++ and Numpy array for Python.
         */
        CMat                getSnapshot(size_t pos);
        /**
         * @brief Gets the current frame index.
         * @return Current frame index.
         */
        size_t              getCurrentPos() const;

        CMat                getStaticImage() const;

        /**
         * @brief Get the image at the current frame index.
         * @return CMat for C++ and Numpy array for Python.
         */
        CMat                getImage() override;
        /**
         * @brief Gets the number of unit elements to process. Used to determine the number of steps for progress bar.
         * @return The number of frames of the video.
         */

        size_t              getUnitElementCount() const override;
        /**
         * @brief Checks if the input or output has a video source.
         * @return True if there is a video source, False otherwise.
         */
        CDataInfoPtr        getDataInfo() override;

        std::vector<DataFileFormat> getPossibleSaveFormats() const override;

        bool                hasVideo();

        /**
         * @brief Checks if the video contains valid data.
         * @return True if there is some valid data, False otherwise.
         */
        bool                isDataAvailable() const override;
        bool                isReadStarted() const;
        bool                isWriteStarted() const;
        /**
         * @brief Starts the video
         */
        void                startVideo(int timeout);
        /**
         * @brief Writes the video to disk. The path is built from the current video path and the given name.
         */
        void                startVideoWrite(int width, int height, int frames, int fps=25, int fourcc=-1, int timeout=-1);
        /**
         * @brief Stops the video
         */
        void                stopVideo();
        /**
         * @brief Stops writting to disk.
         */
        void                stopVideoWrite();
        /**
         * @brief Appends image to the video.
         * @param image: CMat for C++ and Numpy array for Python.
         */
        void                addVideoImage(const CMat& image);
        /**
         * @brief Appends a new image to the list of images to write to disk.
         * @param image: CMat for C++ and Numpy array for Python.
         */
        void                writeImage(CMat image);
        /**
         * @brief Clears the current image at the current frame index.
         */
        void                clearData() override;
        /**
         * @brief Performs a copy of the static information from input/output to another input/output.
         * @details For this class, the channel count is the only static data.
         * @param ioPtr: input/output to copy static data from.
         */
        void                copyStaticData(const WorkflowTaskIOPtr& ioPtr) override;
        /**
         * @brief Blocks current thread until video write operation finish
         */
        void                waitWriteFinished(int timeout);
        /**
         * @brief Blocks current thread until video fram from read operation is available
         */
        void                waitReadImage(int timeout) const;

        /**
         * @brief Makes a deep copy and returns it.
         * @return New CVideoIO instance as a shared pointer.
         */
        std::shared_ptr<CVideoIO> clone() const;

        void                save() override;

    private:

        virtual std::shared_ptr<CWorkflowTaskIO>    cloneImp() const override;

    private:

        CDataVideoBufferPtr m_pVideoBuffer = nullptr;
        std::vector<CMat>   m_videoImgList;
        int                 m_frameIndex = 0;
        int                 m_frameIndexRead = -1;
};

using VideoIOPtr = std::shared_ptr<CVideoIO>;

class DATAPROCESSSHARED_EXPORT CVideoIOFactory: public CImageIOFactory
{
    public:

        CVideoIOFactory()
        {
            m_name = "CVideoIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            return std::make_shared<CVideoIO>(dataType);
        }
};

#endif // CVIDEOIO_H
