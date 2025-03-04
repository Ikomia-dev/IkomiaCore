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

#ifndef COCVWIDGETDUALTVL1OPTICALFLOW_HPP
#define COCVWIDGETDUALTVL1OPTICALFLOW_HPP
#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/optflow/COcvDualTVL1OpticalFlow.hpp"
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>

class COcvWidgetDualTVL1OF : public CWorkflowTaskWidget
{
    public:

        COcvWidgetDualTVL1OF(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetDualTVL1OF(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvDualTVL1OFParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvDualTVL1OFParam>();

            m_pSpinTau = addDoubleSpin(0, tr("Tau"), m_pParam->m_tau);
            m_pSpinLambda = addDoubleSpin(1, tr("Lambda"), m_pParam->m_lambda);
            m_pSpinTheta = addDoubleSpin(2, tr("Theta"), m_pParam->m_theta);
            m_pSpinScales = addSpin(3, tr("Scales"), m_pParam->m_nscales);
            m_pSpinWarps = addSpin(4, tr("Warps"), m_pParam->m_warps);
            m_pSpinEps = addDoubleSpin(5, tr("Epsilon"), m_pParam->m_epsilon);
            m_pSpinInner = addSpin(6, tr("Inner iterations"), m_pParam->m_innerIterations);
            m_pSpinOuter = addSpin(7, tr("Outer iterations"), m_pParam->m_outerIterations);
            m_pSpinScaleStep = addDoubleSpin(8, tr("Scale steps"), m_pParam->m_scaleStep);
            m_pSpinGamma = addDoubleSpin(9, tr("Gamma"), m_pParam->m_gamma);
            m_pSpinMedian = addSpin(10, tr("Median filter"), m_pParam->m_medianFiltering);

            m_pCheck = addCheck(11, tr("Use initial flow"), m_pParam->m_useInitialFlow);
        }

        void onApply() override
        {
            m_pParam->m_tau = m_pSpinTau->value();
            m_pParam->m_lambda = m_pSpinLambda->value();
            m_pParam->m_theta = m_pSpinTheta->value();
            m_pParam->m_nscales = m_pSpinScales->value();
            m_pParam->m_warps = m_pSpinWarps->value();
            m_pParam->m_epsilon = m_pSpinEps->value();
            m_pParam->m_innerIterations = m_pSpinInner->value();
            m_pParam->m_outerIterations = m_pSpinOuter->value();
            m_pParam->m_scaleStep = m_pSpinScaleStep->value();
            m_pParam->m_gamma = m_pSpinGamma->value();
            m_pParam->m_medianFiltering = m_pSpinMedian->value();
            m_pParam->m_useInitialFlow = m_pCheck->isChecked();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvDualTVL1OFParam> m_pParam = nullptr;
        QDoubleSpinBox* m_pSpinTau = nullptr;
        QDoubleSpinBox* m_pSpinLambda = nullptr;
        QDoubleSpinBox* m_pSpinTheta = nullptr;
        QDoubleSpinBox* m_pSpinEps = nullptr;
        QDoubleSpinBox* m_pSpinScaleStep = nullptr;
        QDoubleSpinBox* m_pSpinGamma = nullptr;
        QSpinBox*       m_pSpinScales = nullptr;
        QSpinBox*       m_pSpinWarps = nullptr;
        QSpinBox*       m_pSpinInner = nullptr;
        QSpinBox*       m_pSpinOuter = nullptr;
        QSpinBox*       m_pSpinMedian = nullptr;
        QCheckBox*      m_pCheck = nullptr;
};

class COcvWidgetDualTVL1OFFactory : public CWidgetFactory
{
    public:

        COcvWidgetDualTVL1OFFactory()
        {
            m_name = "ocv_dual_tvl1_flow";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetDualTVL1OF>(pParam);
        }
};

#endif // COCVWIDGETDUALTVL1OPTICALFLOW_HPP
