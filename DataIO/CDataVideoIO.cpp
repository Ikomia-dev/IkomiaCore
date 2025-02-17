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

#include "CDataVideoIO.h"
#include <algorithm>
#include "UtilsTools.hpp"
#include "COpencvVideoIO.h"
#include "DataIOTools.hpp"

CDataVideoIO::CDataVideoIO()
{
    m_pVideoIO = nullptr;
    allocateDataIOPtr("");
}

CDataVideoIO::CDataVideoIO(const std::string &fileName)
{
    m_pVideoIO = nullptr;
    allocateDataIOPtr(fileName);
}

std::vector<std::string> CDataVideoIO::fileNames(const SubsetBounds &bounds)
{
    if(m_pVideoIO)
        return m_pVideoIO->fileNames(bounds);
    else
        return std::vector<std::string>();
}

Dimensions CDataVideoIO::dimensions()
{
    if(m_pVideoIO)
        return m_pVideoIO->dimensions();
    else
        return Dimensions();
}

Dimensions CDataVideoIO::dimensions(const SubsetBounds &bounds)
{
    if(m_pVideoIO)
        return m_pVideoIO->dimensions(bounds);
    else
        return Dimensions();
}

CDataInfoPtr CDataVideoIO::dataInfo()
{
    if(m_pVideoIO)
        return m_pVideoIO->dataInfo();
    else
        return nullptr;
}

CDataInfoPtr CDataVideoIO::dataInfo(const SubsetBounds &bounds)
{
    if(m_pVideoIO)
        return m_pVideoIO->dataInfo(bounds);
    else
        return nullptr;
}

CMat CDataVideoIO::read()
{
    if(m_pVideoIO)
        return m_pVideoIO->read();
    else
        return CMat();
}

CMat CDataVideoIO::read(const SubsetBounds &bounds)
{
    if(m_pVideoIO)
        return m_pVideoIO->read(bounds);
    else
        return CMat();
}

CMat CDataVideoIO::readLive(int timeout)
{
    if(m_pVideoIO)
        return m_pVideoIO->readLive(timeout);
    else
        return CMat();
}

CMat CDataVideoIO::readLive(const SubsetBounds& bounds, int timeout)
{
    if(m_pVideoIO)
        return m_pVideoIO->readLive(bounds, timeout);
    else
        return CMat();
}

void CDataVideoIO::write(const CMat& image)
{
    if(m_pVideoIO)
        m_pVideoIO->write(image);
    else
        throw CException(CoreExCode::NULL_POINTER, "Write error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

void CDataVideoIO::write(const CMat &image, const std::string &path)
{
    if(m_pVideoIO)
        m_pVideoIO->write(image, path);
    else
        throw CException(CoreExCode::NULL_POINTER, "Write error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

void CDataVideoIO::stopWrite(int timeout)
{
    if(m_pVideoIO)
        m_pVideoIO->stopWrite(timeout);
    else
        throw CException(CoreExCode::NULL_POINTER, "Stop write error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

void CDataVideoIO::stopRead()
{
    if (m_pVideoIO)
        m_pVideoIO->stopRead();
    else
        throw CException(CoreExCode::NULL_POINTER, "Stop read error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

void CDataVideoIO::waitWriteFinished(int timeout)
{
    if(m_pVideoIO)
        m_pVideoIO->waitWriteFinished(timeout);
    else
        throw CException(CoreExCode::NULL_POINTER, "Wait write finished error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

void CDataVideoIO::close()
{
    if(m_pVideoIO)
        m_pVideoIO->close();
    else
        throw CException(CoreExCode::NULL_POINTER, "Stop camera error: invalid DataImageIO object", __func__, __FILE__, __LINE__);
}

IODataType CDataVideoIO::dataType(const std::string &fileName)
{
    if(isImageFormat(Utils::File::extension(fileName)))
        return IODataType::IMAGE;
    else
        return IODataType::VIDEO;
}

Dimensions CDataVideoIO::dimensions(const std::string &fileName)
{
    auto pImageIO = _allocateDataIOPtr(fileName);
    if(pImageIO)
        return pImageIO->dimensions();
    else
        return Dimensions();
}

bool CDataVideoIO::isVideoFormat(const std::string &extension, bool bVideoOnly)
{
    bool bIsVideo = (extension == ".avi" || extension == ".mp4" || extension == ".webm" || extension == ".mkv" || extension == ".ts" || extension == ".mxf" || extension == ".mxg");
    if(bVideoOnly)
        return bIsVideo;
    else
        return bIsVideo || isImageFormat(extension);
}

bool CDataVideoIO::isSameImageSequence(const std::string& fileRenamed, const std::string& fileName)
{
    std::string extension = Utils::File::extension(fileName);
    if(isImageFormat(extension))
    {
        auto ret = getImageSequenceInfo(fileName);
        return fileRenamed == ret.first;
    }
    else
        return fileRenamed == fileName;
}

bool CDataVideoIO::isCameraStream(const std::string& streamName, const std::string& extension)
{
    if (streamName.find("http://") != std::string::npos ||
        streamName.find("https://") != std::string::npos ||
        streamName.find("rtsp://") != std::string::npos)
    {
        return true;
    }

    if (extension.empty())
        return true;

    return false;
}

void CDataVideoIO::allocateDataIOPtr(const std::string &fileName)
{
    m_pVideoIO = _allocateDataIOPtr(fileName);
}

CDataVideoIO::CVideoIOPtr CDataVideoIO::_allocateDataIOPtr(const std::string &fileName)
{
    std::string extension = Utils::File::extension(fileName);
    if (isImageFormat(extension))
    {
        auto ret = getImageSequenceInfo(fileName);
        return std::make_unique<COpencvVideoIO>(ret.first, ret.second);
    }
    else if (isVideoFormat(extension, true) || isCameraStream(fileName, extension))
        return std::make_unique<COpencvVideoIO>(fileName);
    else
        throw CException(CoreExCode::INVALID_FILE, "Video IO error: unhandled video file format", __func__, __FILE__, __LINE__);
}

bool CDataVideoIO::isImageFormat(const std::string &extension)
{
    /* Extract from OpenCV documentation
    Windows bitmaps - *.bmp, *.dib (always supported)
    JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
    JPEG 2000 files - *.jp2 (see the Notes section)
    Portable Network Graphics - *.png (see the Notes section)
    WebP - *.webp (see the Notes section)
    Portable image format - *.pbm, *.pgm, *.ppm *.pxm, *.pnm (always supported)
    Sun rasters - *.sr, *.ras (always supported)
    TIFF files - *.tiff, *.tif (see the Notes section)
    OpenEXR Image files - *.exr (see the Notes section)
    Radiance HDR - *.hdr, *.pic (always supported)
    Raster and Vector geospatial data supported by Gdal (see the Notes section)*/
    std::string ext = Utils::String::toLower(extension);
    return (ext == ".bmp" || ext == ".dib" || ext == ".jpeg" || ext == ".jpg" || ext == ".jpe" ||
            ext == ".jp2" || ext == ".png" || ext == ".webp" || ext == ".pbm" || ext == ".pgm" ||
            ext == ".ppm" || ext == ".pxm" || ext == ".pnm" || ext == ".sr" || ext == ".ras" ||
            ext == ".tiff" || ext == ".tif" || ext == ".exr" || ext == ".hdr" || ext == ".pic");
}

std::pair<std::string, int> CDataVideoIO::getImageSequenceInfo(const std::string& fileName)
{    
    // full path
    boost::filesystem::path fullpath(fileName);
    // Extract filename with extension
    auto filename = fullpath.filename();
    auto filenameStr = filename.string();

    // Check if fileName is already formatted (it is the case when saving)
    if(filenameStr.find("%") != std::string::npos)
        return std::make_pair(fileName, 1);

    // Use opencv glob function in order to get all files in current folder
    std::vector<cv::String> filenames;
    cv::String folderPath = fullpath.parent_path().string();
    cv::glob(folderPath, filenames);

    // Extract filename without extension
    auto stem = filename.stem();
    // Find digit position in filename
    size_t lastInd = stem.string().find_last_not_of("0123456789");
    // Extract digit part string
    auto digit = stem.string().substr(lastInd + 1);
    // Count number of digit
    auto numDigit = digit.length();

    // Check number of digit is same order than number of corresponding images
    // Throw if problem...we could also make renaming on the fly or asking user?
    auto numImg = std::to_string(filenames.size()).length();
    if(numDigit < numImg)
        throw CException(DataIOExCode::VIDEO_WRONG_IMG_NUMBERS, "Number of images doesn't correspond to number of digits in image sequence name.", __func__, __FILE__, __LINE__);

    // Replace digit string with opencv pattern
    auto pattern = "%"+std::to_string(numDigit)+"d";
    Utils::String::replace(filenameStr, digit, pattern);

    std::pair<std::string,int> ret;
    ret.first = Utils::File::makePath(fullpath.parent_path().string(), filenameStr);
    ret.second = filenames.size();
    return ret;
}

