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

#ifndef COCVWIDGETDNNSEGMENTATION_HPP
#define COCVWIDGETDNNSEGMENTATION_HPP

#include "COcvWidgetDnnProcess.h"
#include "Process/OpenCV/dnn/COcvDnnInstanceSeg.h"

class COcvWidgetDnnSemanticSeg: public COcvWidgetDnnProcess
{
    public:

        COcvWidgetDnnSemanticSeg(QWidget *parent = Q_NULLPTR): COcvWidgetDnnProcess(parent)
        {
            init();
        }
        COcvWidgetDnnSemanticSeg(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR) : COcvWidgetDnnProcess(pParam, parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(pParam);
            init();
        }

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvDnnInstanceSegParam>();

            auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
            assert(pParam);

            m_pComboNetType = addCombo(tr("Network type"));
            m_pComboNetType->addItem("Mask RCNN", COcvDnnInstanceSegParam::NetworkType::MASK_RCNN);
            m_pComboNetType->setCurrentIndex(m_pComboNetType->findData(pParam->m_netType));

            m_pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
            m_pSpinMaskThreshold = addDoubleSpin(tr("Mask threshold"), pParam->m_maskThreshold, 0.0, 1.0, 0.1, 2);

            initConnections();
        }

    private:

        void    initConnections()
        {
            connect(m_pComboNetType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &COcvWidgetDnnSemanticSeg::onNetworkTypeChanged);
            connect(m_pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
                assert(pParam);
                pParam->m_confidence = val;
            });
            connect(m_pSpinMaskThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
                assert(pParam);
                pParam->m_maskThreshold = val;
            });
        }

    private slots:

        void    onNetworkTypeChanged(int index)
        {
            Q_UNUSED(index);
            auto pParam = std::dynamic_pointer_cast<COcvDnnInstanceSegParam>(m_pParam);
            assert(pParam);
            pParam->m_netType = m_pComboNetType->currentData().toInt();
        }

    private:

        QComboBox*      m_pComboNetType = nullptr;
        QDoubleSpinBox* m_pSpinConfidence = nullptr;
        QDoubleSpinBox* m_pSpinMaskThreshold = nullptr;
};

class COcvWidgetDnnSegmentationFactory : public CWidgetFactory
{
    public:

        COcvWidgetDnnSegmentationFactory()
        {
            m_name = "ocv_dnn_instance_segmentation";
        }

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam)
        {
            return std::make_shared<COcvWidgetDnnSemanticSeg>(pParam);
        }
};

#endif // COCVWIDGETDNNSEGMENTATION_HPP
