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

#ifndef CGMICWIDGETBOOSTFADE_HPP
#define CGMICWIDGETBOOSTFADE_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/Gmic/Colors/CGmicBoostFade.hpp"

class CGmicWidgetBoostFade : public CWorkflowTaskWidget
{
    public:

        CGmicWidgetBoostFade(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        CGmicWidgetBoostFade(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<CGmicBoostFadeParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CGmicBoostFadeParam>();

            m_pSpinAmplitude = addSpin(0, tr("Amplitude"), m_pParam->m_amplitude, 0, 10);
            m_pComboColorSpace = addCombo(1, tr("Color space"));
            m_pComboColorSpace->addItem(tr("YCbCr"), CGmicBoostFadeParam::YCBCR);
            m_pComboColorSpace->addItem(tr("Lab"), CGmicBoostFadeParam::LAB);
        }

        void onApply() override
        {
            m_pParam->m_amplitude = m_pSpinAmplitude->value();
            m_pParam->m_colorSpace = m_pComboColorSpace->currentData().toInt();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<CGmicBoostFadeParam> m_pParam = nullptr;
        QSpinBox*   m_pSpinAmplitude = nullptr;
        QComboBox*  m_pComboColorSpace = nullptr;
};

class CGmicWidgetBoostFadeFactory : public CWidgetFactory
{
    public:

        CGmicWidgetBoostFadeFactory()
        {
            m_name = "gmic_boost_fade";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<CGmicWidgetBoostFade>(pParam);
        }
};

#endif // CGMICWIDGETBOOSTFADE_HPP
