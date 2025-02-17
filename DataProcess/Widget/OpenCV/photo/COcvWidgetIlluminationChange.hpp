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

#ifndef COCVWIDGETILLUMINATIONCHANGE_HPP
#define COCVWIDGETILLUMINATIONCHANGE_HPP
#include "Process/OpenCV/photo/COcvIlluminationChange.hpp"
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QSlider>
#include <QGridLayout>

class COcvWidgetIlluminationChange : public CWorkflowTaskWidget
{
    public:

        COcvWidgetIlluminationChange(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetIlluminationChange(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvIlluminationChangeParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvIlluminationChangeParam>();

            m_pSpinAlpha = addDoubleSpin(0, tr("Alpha"), m_pParam->m_alpha, 0, 2, 0.1);
            m_pSpinBeta = addDoubleSpin(1, tr("Beta"), m_pParam->m_beta, 0, 2, 0.1);
        }

        void onApply() override
        {
            m_pParam->m_alpha = m_pSpinAlpha->value();
            m_pParam->m_beta = m_pSpinBeta->value();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvIlluminationChangeParam> m_pParam = nullptr;
        QDoubleSpinBox* m_pSpinAlpha = nullptr;
        QDoubleSpinBox* m_pSpinBeta = nullptr;
};

class COcvWidgetIlluminationChangeFactory : public CWidgetFactory
{
    public:

        COcvWidgetIlluminationChangeFactory()
        {
            m_name = "ocv_illumination_change";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetIlluminationChange>(pParam);
        }
};
#endif // COCVWIDGETILLUMINATIONCHANGE_HPP
