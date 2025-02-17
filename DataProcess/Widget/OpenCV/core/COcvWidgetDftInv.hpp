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

#ifndef COCVWIDGETDftInvINV_HPP
#define COCVWIDGETDftInvINV_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/core/COcvDftInv.hpp"
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>

class COcvWidgetDftInv : public CWorkflowTaskWidget
{
    public:

        COcvWidgetDftInv(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetDftInv(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvDftInvParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvDftInvParam>();

            auto pLabelFlags = new QLabel(tr("Flags"));
            auto pComboFlags = new QComboBox;
            pComboFlags->addItem("Default", 0);
            pComboFlags->addItem("DFT_INVERSE", cv::DFT_INVERSE);
            pComboFlags->addItem("DFT_SCALE", cv::DFT_SCALE);
            pComboFlags->addItem("DFT_ROWS", cv::DFT_ROWS);
            pComboFlags->addItem("DFT_COMPLEX_OUTPUT", cv::DFT_COMPLEX_OUTPUT);
            pComboFlags->addItem("DFT_REAL_OUTPUT", cv::DFT_REAL_OUTPUT);
            pComboFlags->addItem("DFT_COMPLEX_INPUT", cv::DFT_COMPLEX_INPUT);
            pComboFlags->setCurrentIndex(pComboFlags->findData(m_pParam->m_flags));

            m_pLayout->addWidget(pLabelFlags, 0, 0);
            m_pLayout->addWidget(pComboFlags, 0, 1);       
        }

        void onApply() override
        {
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvDftInvParam> m_pParam = nullptr;
};

class COcvWidgetDftInvFactory : public CWidgetFactory
{
    public:

        COcvWidgetDftInvFactory()
        {
            m_name = "ocv_dft_inverse";
        }

        virtual WorkflowTaskWidgetPtr create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_unique<COcvWidgetDftInv>(pParam);
        }
};

#endif // COCVWIDGETDftInvINV_HPP
