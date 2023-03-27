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

#ifndef DATAPROCESSTOOLS_HPP
#define DATAPROCESSTOOLS_HPP

#include "opencv2/opencv.hpp"
#include "Data/CMat.hpp"
#include "base64.hpp"
#include "Data/CDataConversion.h"

namespace Ikomia
{
    namespace Utils
    {
        namespace Image
        {
            inline double       getMaxValueFromDepth(int depth)
            {
                double maxValue = 1;
                switch(depth)
                {
                    case CV_8S: maxValue = 127; break;
                    case CV_8U: maxValue = 255; break;
                    case CV_16S: maxValue = 32767; break;
                    case CV_16U: maxValue = 65535; break;
                    case CV_32S: maxValue = 2147483647; break;
                    case CV_32F: maxValue = 1.0; break;
                    case CV_64F: maxValue = 1.0; break;
                }
                return maxValue;
            }
            inline CMat         createOverlayMask(const CMat& image, const CMat& colorMap, bool bTransparentZero)
            {
                CMat srcOvrImg, ovrImg;

                if(image.depth() != CV_8U)
                    image.convertTo(srcOvrImg, CV_8U);
                else
                    srcOvrImg = image;

                cv::applyColorMap(srcOvrImg, ovrImg, colorMap);
                cv::cvtColor(ovrImg, ovrImg, cv::COLOR_RGB2RGBA);

                if (bTransparentZero)
                {
                    #pragma omp parallel for
                        for(int i=0; i<ovrImg.rows; ++i)
                        {
                            for(int j=0; j<ovrImg.cols; ++j)
                            {
                                if(ovrImg.at<cv::Vec4b>(i, j)[0] == 0 && ovrImg.at<cv::Vec4b>(i, j)[1] == 0 && ovrImg.at<cv::Vec4b>(i, j)[2] == 0)
                                    ovrImg.at<cv::Vec4b>(i, j)[3] = 0;
                            }
                        }
                }
                return ovrImg;
            }
            inline CMat         mergeColorMask(const CMat& image, const CMat& mask, const CMat& colormap, double opacity, bool bTransparentZero)
            {
                CMat result, colorMask;

                if (mask.empty() || mask.data == nullptr)
                    return image;

                if(mask.depth() != CV_8U)
                    mask.convertTo(colorMask, CV_8U);
                else
                    colorMask = mask;

                cv::applyColorMap(colorMask, colorMask, colormap);
                cv::addWeighted(image, (1.0 - opacity), colorMask, opacity, 0.0, result, image.depth());

                if (bTransparentZero)
                {
                    cv::Mat maskNot = mask > 0;
                    cv::bitwise_not(maskNot, maskNot);
                    image.copyTo(result, maskNot);
                }
                return result;
            }
            inline std::string  toJson(const CMat& image, const std::vector<std::string> &options)
            {
                if (image.data == nullptr)
                    return std::string();

                CMat img8bits;
                if (image.depth() != CV_8U)
                    CDataConversion::to8Bits(image, img8bits);
                else
                    img8bits = image;

                std::string format = ".jpg";
                auto it = std::find(options.begin(), options.end(), "image_format");

                if (it != options.end())
                {
                    size_t index = it - options.begin() + 1;
                    if (index < options.size())
                    {
                        if (options[index] == "png")
                            format = ".png";
                    }
                }

                std::vector<uchar> buffer;
                cv::imencode(format, img8bits, buffer);
                auto* pEncodedBuf = reinterpret_cast<unsigned char*>(buffer.data());
                return base64_encode(pEncodedBuf, buffer.size());
            }
            inline CMat         fromJson(const std::string& b64ImgStr)
            {
                std::string decoded = base64_decode_fast(b64ImgStr.c_str(), b64ImgStr.size());
                std::vector<uchar> data(decoded.begin(), decoded.end());
                return cv::imdecode(CMat(data), cv::IMREAD_COLOR);
            }
        }
    }
}

#endif // DATAPROCESSTOOLS_HPP
