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

#ifndef COCVWIDGETFASTNLMEANS_HPP
#define COCVWIDGETFASTNLMEANS_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/photo/COcvFastNlMeans.hpp"

class COcvWidgetFastNlMeans : public CWorkflowTaskWidget
{
    public:

        COcvWidgetFastNlMeans(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }
        COcvWidgetFastNlMeans(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvFastNlMeansParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvFastNlMeansParam>();

            QLabel* pLabelH = new QLabel(QObject::tr("Filter strength"));
            m_pSpinH = new QDoubleSpinBox;
            m_pSpinH->setValue(m_pParam->m_h);
            m_pSpinH->setSingleStep(0.5);
            m_pSpinH->setRange(0, INT_MAX - 1);

            QLabel* pLabelBlockSize = new QLabel(QObject::tr("Block size"));
            m_pSpinBlockSize = new QDoubleSpinBox;
            m_pSpinBlockSize->setValue(m_pParam->m_blockSize);
            m_pSpinBlockSize->setSingleStep(2);
            m_pSpinBlockSize->setRange(1, INT_MAX - 1);

            QLabel* pLabelSearchSize = new QLabel(QObject::tr("Search size"));
            m_pSpinSearchSize = new QDoubleSpinBox;
            m_pSpinSearchSize->setValue(m_pParam->m_searchSize);
            m_pSpinSearchSize->setSingleStep(2);
            m_pSpinSearchSize->setRange(1, INT_MAX - 1);

            m_pLayout->addWidget(pLabelH, 0, 0);
            m_pLayout->addWidget(m_pSpinH, 0, 1);
            m_pLayout->addWidget(pLabelBlockSize, 1, 0);
            m_pLayout->addWidget(m_pSpinBlockSize, 1, 1);
            m_pLayout->addWidget(pLabelSearchSize, 2, 0);
            m_pLayout->addWidget(m_pSpinSearchSize, 2, 1);
        }

        void onApply() override
        {
            m_pParam->m_h = m_pSpinH->value();
            m_pParam->m_blockSize = m_pSpinBlockSize->value();
            m_pParam->m_searchSize = m_pSpinSearchSize->value();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvFastNlMeansParam>   m_pParam = nullptr;
        QDoubleSpinBox*                         m_pSpinH = nullptr;
        QDoubleSpinBox*                         m_pSpinBlockSize = nullptr;
        QDoubleSpinBox*                         m_pSpinSearchSize = nullptr;
};

class COcvWidgetFastNlMeansFactory : public CWidgetFactory
{
    public:

        COcvWidgetFastNlMeansFactory()
        {
            m_name = "ocv_non_local_means_filter";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetFastNlMeans>(pParam);
        }
};

#endif // COCVWIDGETFASTNLMEANS_HPP
