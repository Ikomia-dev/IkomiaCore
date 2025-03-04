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

#ifndef COCVWIDGETMAX_HPP
#define COCVWIDGETMAX_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/core/COcvMax.hpp"

class COcvWidgetMax : public CWorkflowTaskWidget
{
    public:

        COcvWidgetMax(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetMax(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvMaxParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvMaxParam>();

            m_pCheckScalar = addCheck(0, tr("Scalar operation"), m_pParam->m_bScalar);
            m_pSpinScalar = addDoubleSpin(1, tr("Scalar value"), m_pParam->m_scalar[0]);
        }

        void onApply() override
        {
            m_pParam->m_bScalar = m_pCheckScalar->isChecked();
            m_pParam->m_scalar= cv::Scalar::all(m_pSpinScalar->value());
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvMaxParam> m_pParam = nullptr;
        QCheckBox*      m_pCheckScalar = nullptr;
        QDoubleSpinBox* m_pSpinScalar = nullptr;
};

class COcvWidgetMaxFactory : public CWidgetFactory
{
    public:

        COcvWidgetMaxFactory()
        {
            m_name = "ocv_max";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetMax>(pParam);
        }
};

#endif // COCVWIDGETMAXHPP_H
