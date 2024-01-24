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
#include "Graphics/CGraphicsConversion.h"
#include "Graphics/CGraphicsItem.hpp"

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
            inline void         burnGraphics(CMat &image, const std::vector<ProxyGraphicsItemPtr> &items)
            {
                CGraphicsConversion graphicsConv((int)image.getNbCols(), (int)image.getNbRows());

                //Double dispatch design pattern
                for(auto it : items)
                    it->insertToImage(image, graphicsConv, false, false);
            }
            inline CMat         mergeColorMask(const CMat& image, const CMat& mask, const CMat& colormap, double opacity, bool bTransparentZero)
            {
                CMat src, result, colorMask;

                if (mask.empty() || mask.data == nullptr)
                    return image;

                if(mask.depth() != CV_8U)
                    mask.convertTo(colorMask, CV_8U);
                else
                    colorMask = mask;

                if (image.channels() == 1)
                    cv::cvtColor(image, src, cv::COLOR_GRAY2RGB);
                else
                    src = image;

                cv::applyColorMap(colorMask, colorMask, colormap);
                cv::addWeighted(src, (1.0 - opacity), colorMask, opacity, 0.0, result, src.depth());

                if (bTransparentZero)
                {
                    cv::Mat maskNot = mask > 0;
                    cv::bitwise_not(maskNot, maskNot);
                    src.copyTo(result, maskNot);
                }
                return result;
            }
            inline CMat         mergeColorMask(const CMat& image, const CMat& mask, double opacity, bool bTransparentZero)
            {
                CMat src, result, colorMask;

                if (mask.empty() || mask.data == nullptr || mask.channels() == 1)
                    return image;

                if (image.channels() == 1)
                    cv::cvtColor(image, src, cv::COLOR_GRAY2RGB);
                else
                    src = image;

                if (mask.channels() == 4)
                    cv::cvtColor(mask, colorMask, cv::COLOR_RGBA2RGB);
                else
                    colorMask = mask;

                if(colorMask.depth() != CV_8U)
                    colorMask.convertTo(colorMask, CV_8U);

                cv::addWeighted(image, (1.0 - opacity), colorMask, opacity, 0.0, result, src.depth());

                if (bTransparentZero)
                {
                    cv::Mat maskNot = colorMask > 0;
                    cv::bitwise_not(maskNot, maskNot);
                    src.copyTo(result, maskNot);
                }
                return result;
            }
            inline CMat         createColorMap(const std::vector<CColor>& colors, bool bReserveZero)
            {
                int startIndex = 0;
                cv::Mat colormap = cv::Mat::zeros(256, 1, CV_8UC3);

                if (bReserveZero)
                    startIndex = 1;

                if(colors.size() == 0)
                {
                    //Random colors
                    std::srand(RANDOM_COLOR_SEED);
                    for(int i=startIndex; i<256; ++i)
                    {
                        for(int j=0; j<3; ++j)
                            colormap.at<cv::Vec3b>(i, 0)[j] = (uchar)((double)std::rand() / (double)RAND_MAX * 255.0);
                    }
                }
                else if(colors.size() == 1)
                {
                    if (colors[0].size() >= 3)
                    {
                        if (bReserveZero)
                            colormap.at<cv::Vec3b>(startIndex, 0) = {colors[0][0], colors[0][1], colors[0][2]};
                        else
                            colormap.at<cv::Vec3b>(255, 0) = {colors[0][0], colors[0][1], colors[0][2]};
                    }
                }
                else
                {
                    for(int i=0; i<std::min<int>(255, (int)colors.size()); ++i)
                    {
                        if (colors[i].size() >= 3)
                            colormap.at<cv::Vec3b>(i+startIndex, 0) = {colors[i][0], colors[i][1], colors[i][2]};
                    }

                    for(int i=(int)colors.size()+1; i<256; ++i)
                        colormap.at<cv::Vec3b>(i, 0) = {(uchar)i, (uchar)i, (uchar)i};
                }
                return colormap;
            }
            inline std::string  toJson(const CMat& image, const std::vector<std::string> &options)
            {
                if (image.data == nullptr)
                    return std::string();

                CMat img8bits;
                if (image.depth() != CV_8U)
                {
                    CDataConversion::to8Bits(image, img8bits);
                    if(image.channels() > 1)
                        cv::cvtColor(img8bits, img8bits, cv::COLOR_RGB2BGR);
                }
                else
                {
                    if(image.channels() > 1)
                        cv::cvtColor(image, img8bits, cv::COLOR_RGB2BGR);
                    else
                        img8bits = image;
                }

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
                // cv::imencode need BGR array
                cv::imencode(format, img8bits, buffer);
                auto* pEncodedBuf = reinterpret_cast<unsigned char*>(buffer.data());
                return base64_encode(pEncodedBuf, buffer.size());
            }
            inline CMat         fromJson(const std::string& b64ImgStr)
            {
                std::string decoded = base64_decode_fast(b64ImgStr.c_str(), b64ImgStr.size());
                std::vector<uchar> data(decoded.begin(), decoded.end());
                cv::Mat img = cv::imdecode(CMat(data), cv::IMREAD_UNCHANGED);

                // cv::imdecode return BGR array for color images
                int c = img.channels();
                if (c == 3)
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
                else if (c == 4)
                    cv::cvtColor(img, img, cv::COLOR_BGRA2RGBA);

                return img;
            }
        }
    }
}

#endif // DATAPROCESSTOOLS_HPP
