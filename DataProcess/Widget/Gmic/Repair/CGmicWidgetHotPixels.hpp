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

#ifndef CGMICWIDGETHOTPIXELS_HPP
#define CGMICWIDGETHOTPIXELS_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/Gmic/Repair/CGmicHotPixels.hpp"

class CGmicWidgetHotPixels : public CWorkflowTaskWidget
{
    public:

        CGmicWidgetHotPixels(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        CGmicWidgetHotPixels(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<CGmicHotPixelsParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CGmicHotPixelsParam>();

            m_pSpinMaskSize = addSpin(0, tr("Mask size"), m_pParam->m_maskSize);
            m_pSpinThreshold = addSpin(1, tr("Threshold(%)"), m_pParam->m_threshold);
        }

        void onApply() override
        {
            m_pParam->m_maskSize = m_pSpinMaskSize->value();
            m_pParam->m_threshold = m_pSpinThreshold->value();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<CGmicHotPixelsParam> m_pParam = nullptr;
        QSpinBox*   m_pSpinMaskSize = nullptr;
        QSpinBox*   m_pSpinThreshold = nullptr;
};

class CGmicWidgetHotPixelsFactory : public CWidgetFactory
{
    public:

        CGmicWidgetHotPixelsFactory()
        {
            m_name = "gmic_hot_pixels";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<CGmicWidgetHotPixels>(pParam);
        }
};

#endif // CGMICWIDGETHOTPIXELS_HPP
