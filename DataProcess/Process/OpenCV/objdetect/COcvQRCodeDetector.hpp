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

#ifndef COCVQRCODEDETECTOR_HPP
#define COCVQRCODEDETECTOR_HPP
#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"
#include <opencv2/objdetect.hpp>

//----------------------------//
//----- COcvQRCodeDetectorParam -----//
//----------------------------//
class COcvQRCodeDetectorParam: public CWorkflowTaskParam
{
    public:

        COcvQRCodeDetectorParam() : CWorkflowTaskParam(){}

        void        setParamMap(const UMapString& paramMap) override
        {
             m_eps_x = std::stod(paramMap.at("eps_x"));
             m_eps_y = std::stod(paramMap.at("eps_y"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("eps_x", std::to_string(m_eps_x)));
            map.insert(std::make_pair("eps_y", std::to_string(m_eps_y)));
            return map;
        }

    public:
        double m_eps_x = 0.2;
        double m_eps_y = 0.1;
};

//-----------------------//
//----- COcvQRCodeDetector -----//
//-----------------------//
class COcvQRCodeDetector : public C2dImageTask
{
    public:

        COcvQRCodeDetector() : C2dImageTask()
        {
            addOutput(std::make_shared<CGraphicsOutput>());
        }
        COcvQRCodeDetector(const std::string name, const std::shared_ptr<COcvQRCodeDetectorParam>& pParam) : C2dImageTask(name)
        {
            addOutput(std::make_shared<CGraphicsOutput>());
            m_pParam = std::make_shared<COcvQRCodeDetectorParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void    manageOuputGraphics(const std::vector<cv::Point>& points, const std::string& code)
        {
            auto pOutput = std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(getOutputCount() - 1));
            if(pOutput == nullptr)
                throw CException(CoreExCode::NULL_POINTER, "Invalid graphics output", __func__, __FILE__, __LINE__);

            pOutput->setNewLayer(getName());
            pOutput->setImageIndex(0);

            if(points.empty())
            {
                pOutput->clearData();
                return;
            }

            CGraphicsEllipseProperty ellipseProp = m_graphicsContextPtr->getEllipseProperty();
            CGraphicsTextProperty textProp = m_graphicsContextPtr->getTextProperty();

            const int w = 15;
            const int h = 15;

            ellipseProp.m_lineSize = 2;
            ellipseProp.m_penColor = {255,0,0,255};
            ellipseProp.m_brushColor = {255,0,0,100};
            pOutput->addEllipse(points[0].x-w/2, points[0].y-h/2, w, h, ellipseProp);

            ellipseProp.m_penColor = {0,255,0,255};
            ellipseProp.m_brushColor = {0, 255, 0, 100};
            pOutput->addEllipse(points[1].x-w/2, points[1].y-h/2, w, h, ellipseProp);

            ellipseProp.m_penColor = {0,0,255,255};
            ellipseProp.m_brushColor = {0,0,255,100};
            pOutput->addEllipse(points[2].x-w/2, points[2].y-h/2, w, h, ellipseProp);

            ellipseProp.m_penColor = {255,0,255,255};
            ellipseProp.m_brushColor = {255,0,255,100};
            pOutput->addEllipse(points[3].x-w/2, points[3].y-h/2, w, h, ellipseProp);

            QFont font;
            font.setFamily(QString::fromStdString(textProp.m_fontName));
            font.setPointSize(textProp.m_fontSize);
            font.setBold(textProp.m_bBold);
            font.setItalic(textProp.m_bItalic);
            font.setUnderline(textProp.m_bUnderline);
            font.setStrikeOut(textProp.m_bStrikeOut);

            QFontMetrics fm(font);
            int textWidth = fm.horizontalAdvance(QString::fromStdString(code));
            float minX = std::min(points[0].x, std::min(points[1].x, std::min(points[2].x, points[3].x)));
            float maxX = std::max(points[0].x, std::max(points[1].x, std::max(points[2].x, points[3].x)));

            pOutput->addText(code, (minX + maxX)/2.0 - (textWidth/2.0), 0);
        }

        void    run() override
        {
            beginTaskRun();
            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
            auto pParam = std::dynamic_pointer_cast<COcvQRCodeDetectorParam>(m_pParam);

            if(pInput == nullptr || pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

            std::vector<cv::Point> points;
            CMat imgSrc = pInput->getImage();
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                CMat img;
                if(imgSrc.channels() == 3)
                {
                    CMat imgLab, Lab[3];
                    cv::cvtColor(imgSrc, imgLab, cv::COLOR_RGB2Lab);
                    cv::split(imgLab, Lab);
                    img = Lab[0];
                }
                else
                    img = imgSrc;

                cv::QRCodeDetector detector;
                detector.setEpsX(pParam->m_eps_x);
                detector.setEpsY(pParam->m_eps_y);
                std::string code = detector.detectAndDecode(img, points);
                manageOuputGraphics(points, code);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }

            endTaskRun();
            emit m_signalHandler->doProgress();

            forwardInputImage();

            emit m_signalHandler->doProgress();
        }
};

class COcvQRCodeDetectorFactory : public CTaskFactory
{
    public:

        COcvQRCodeDetectorFactory()
        {
            m_info.m_name = "ocv_qrcode_detector";
            m_info.m_shortDescription = QObject::tr("This process detects QR code in images.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Object detection").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "QRCodeDetector";
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/de/dc3/classcv_1_1QRCodeDetector.html";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pQRCodeDetectorParam = std::dynamic_pointer_cast<COcvQRCodeDetectorParam>(pParam);
            if(pQRCodeDetectorParam != nullptr)
                return std::make_shared<COcvQRCodeDetector>(m_info.m_name, pQRCodeDetectorParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pQRCodeDetectorParam = std::make_shared<COcvQRCodeDetectorParam>();
            assert(pQRCodeDetectorParam != nullptr);
            return std::make_shared<COcvQRCodeDetector>(m_info.m_name, pQRCodeDetectorParam);
        }
};

#endif // COCVQRCODEDETECTOR_HPP
