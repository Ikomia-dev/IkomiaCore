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

#ifndef COCVWIDGETTHINNING_HPP
#define COCVWIDGETTHINNING_HPP

#include "Process/OpenCV/ximgproc/COcvThinning.hpp"
#include <QComboBox>
#include <QLabel>
#include <QGridLayout>

class COcvWidgetThinning : public CWorkflowTaskWidget
{
    public:

        COcvWidgetThinning(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetThinning(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvThinningParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvThinningParam>();

            auto pLabel = new QLabel(tr("Type"));
            m_pCombo = new QComboBox;
            m_pCombo->addItem("Zhang-Suen", cv::ximgproc::THINNING_ZHANGSUEN);
            m_pCombo->addItem("Guo-Hall", cv::ximgproc::THINNING_GUOHALL);
            m_pCombo->setCurrentIndex(m_pCombo->findData(m_pParam->m_type));
            
            m_pLayout->addWidget(pLabel, 0, 0);
            m_pLayout->addWidget(m_pCombo, 0, 1);
        }

        void onApply() override
        {
            m_pParam->m_type = m_pCombo->currentData().toInt();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvThinningParam> m_pParam = nullptr;
        QComboBox* m_pCombo = nullptr;
};

class COcvWidgetThinningFactory : public CWidgetFactory
{
    public:

        COcvWidgetThinningFactory()
        {
            m_name = "ocv_thinning";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetThinning>(pParam);
        }
};

#endif // COCVWIDGETTHINNING_HPP
