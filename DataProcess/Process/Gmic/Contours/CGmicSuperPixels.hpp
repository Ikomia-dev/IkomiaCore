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

#ifndef CGMICSUPERPIXELS_HPP
#define CGMICSUPERPIXELS_HPP

#include "Task/C2dImageTask.h"
#include "Process/Gmic/CGmicTaskParam.hpp"
#include "Process/Gmic/CGmicImageConverter.h"

//---------------------------------//
//----- CGmicSuperPixelsParam -----//
//---------------------------------//
class CGmicSuperPixelsParam: public CGmicTaskParam
{
    public:

        enum Colors { RANDOM, AVERAGE };

        CGmicSuperPixelsParam() : CGmicTaskParam(){}

        void        setParamMap(const UMapString& paramMap) override
        {
            m_size = std::stoi(paramMap.at("size"));
            m_regularity = std::stod(paramMap.at("regularity"));
            m_iterations = std::stoi(paramMap.at("iterations"));
            m_colors = std::stoi(paramMap.at("colors"));
            m_borderOpacity = std::stod(paramMap.at("borderOpacity"));
            m_borderColor.setRed(std::stoi(paramMap.at("borderColorR")));
            m_borderColor.setGreen(std::stoi(paramMap.at("borderColorG")));
            m_borderColor.setBlue(std::stoi(paramMap.at("borderColorB")));
            m_borderColor.setAlpha(std::stoi(paramMap.at("borderColorA")));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("size", std::to_string(m_size)));
            map.insert(std::make_pair("regularity", std::to_string(m_regularity)));
            map.insert(std::make_pair("iterations", std::to_string(m_iterations)));
            map.insert(std::make_pair("colors", std::to_string(m_colors)));
            map.insert(std::make_pair("borderOpacity", std::to_string(m_borderOpacity)));
            map.insert(std::make_pair("borderColorR", std::to_string(m_borderColor.red())));
            map.insert(std::make_pair("borderColorG", std::to_string(m_borderColor.green())));
            map.insert(std::make_pair("borderColorB", std::to_string(m_borderColor.blue())));
            map.insert(std::make_pair("borderColorA", std::to_string(m_borderColor.alpha())));
            return map;
        }

        std::string getParamString() const override
        {
            //Build string of parameters, take care of order.
            return std::to_string(m_size) + "," + std::to_string(m_regularity) + "," + std::to_string(m_iterations) + "," +
                    std::to_string(m_colors) + "," + std::to_string(m_borderOpacity) + "," + std::to_string(m_borderColor.red()) + "," +
                    std::to_string(m_borderColor.green()) + "," + std::to_string(m_borderColor.blue()) + "," + std::to_string(m_borderColor.alpha());
        }

    public:

        int     m_size = 16;
        float   m_regularity = 10;
        int     m_iterations = 5;
        int     m_colors = AVERAGE;
        float   m_borderOpacity = 1;
        QColor  m_borderColor = Qt::black;
};

//----------------------------//
//----- CGmicSuperPixels -----//
//----------------------------//
class CGmicSuperPixels : public C2dImageTask
{
    public:

        CGmicSuperPixels() : C2dImageTask()
        {
        }
        CGmicSuperPixels(const std::string name, const std::shared_ptr<CGmicSuperPixelsParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<CGmicSuperPixelsParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
            auto pParam = std::dynamic_pointer_cast<CGmicSuperPixelsParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid input", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            CMat imgSrc = pInput->getImage();
            CMat imgDst(imgSrc.rows, imgSrc.cols, imgSrc.type());
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                //Put image inputs into a gmic_list object
                gmic_list<float> imageList;
                gmic_list<char> image_names;

                //Allocate list, parameter is the number of image inputs
                imageList.assign(1);
                //Conversion from CMat(cv::Mat) to gmic_image(CImg)
                CGmicImageConverter::convert(imgSrc, imageList[0]);
                //Build command for gmic interpreter
                std::string command = "fx_superpixels[0] " + pParam->getParamString();
                //Call interpreter
                gmic(command.c_str(), imageList, image_names);
                //Conversion from gmic_image to CMat
                CGmicImageConverter::convert(imageList[0], imgDst);
                //Free memory
                imageList.assign(0);
            }
            catch(gmic_exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
            }

            endTaskRun();
            emit m_signalHandler->doProgress();

            applyGraphicsMask(imgSrc, imgDst, 0);

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            emit m_signalHandler->doProgress();
        }
};

class CGmicSuperPixelsFactory : public CTaskFactory
{
    public:

        CGmicSuperPixelsFactory()
        {
            m_info.m_name = "gmic_super_pixels";
            m_info.m_shortDescription = QObject::tr("This process performs super-pixels decomposition.").toStdString();
            m_info.m_path = QObject::tr("Gmic/Contours").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/gmic.png").toStdString();
            m_info.m_keywords = "binary,edge,segmentation";
            m_info.m_authors = "David Tschumperlé";
            m_info.m_year = 2017;
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pDerivedParam = std::dynamic_pointer_cast<CGmicSuperPixelsParam>(pParam);
            if(pDerivedParam != nullptr)
                return std::make_shared<CGmicSuperPixels>(m_info.m_name, pDerivedParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pParam = std::make_shared<CGmicSuperPixelsParam>();
            assert(pParam != nullptr);
            return std::make_shared<CGmicSuperPixels>(m_info.m_name, pParam);
        }
};

#endif // CGMICSUPERPIXELS_HPP
