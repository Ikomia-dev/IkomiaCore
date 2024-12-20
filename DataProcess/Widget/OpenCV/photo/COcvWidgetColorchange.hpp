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

#ifndef COCVWIDGETCOLORCHANGE_HPP
#define COCVWIDGETCOLORCHANGE_HPP
#include "Process/OpenCV/photo/COcvColorchange.hpp"
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QSlider>
#include <QGridLayout>

class COcvWidgetColorchange : public CWorkflowTaskWidget
{
    public:

        COcvWidgetColorchange(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetColorchange(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvColorchangeParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvColorchangeParam>();

            m_pSpinRed = addDoubleSpin(0, tr("Red factor"), m_pParam->m_red_mul, 0.5, 2.5, 0.1);
            m_pSpinGreen = addDoubleSpin(1, tr("Green factor"), m_pParam->m_green_mul, 0.5, 2.5, 0.1);
            m_pSpinBlue = addDoubleSpin(2, tr("Blue factor"), m_pParam->m_blue_mul, 0.5, 2.5, 0.1);
        }

        void onApply() override
        {
            m_pParam->m_red_mul = m_pSpinRed->value();
            m_pParam->m_green_mul = m_pSpinGreen->value();
            m_pParam->m_blue_mul = m_pSpinBlue->value();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvColorchangeParam> m_pParam = nullptr;
        QDoubleSpinBox* m_pSpinRed = nullptr;
        QDoubleSpinBox* m_pSpinGreen = nullptr;
        QDoubleSpinBox* m_pSpinBlue = nullptr;
};

class COcvWidgetColorchangeFactory : public CWidgetFactory
{
    public:

        COcvWidgetColorchangeFactory()
        {
            m_name = "ocv_color_change";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetColorchange>(pParam);
        }
};

#endif // COCVWIDGETCOLORCHANGE_HPP
