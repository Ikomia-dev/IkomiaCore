#ifndef COCVWIDGETBLUR_HPP
#define COCVWIDGETBLUR_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/imgproc/COcvBlur.hpp"
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>

class COcvWidgetBlur : public CWorkflowTaskWidget
{
    public:

        COcvWidgetBlur(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetBlur(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvBlurParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvBlurParam>();

            m_pSpin = new QSpinBox;
            m_pSpin->setValue(m_pParam->m_ksize.width);
            QLabel* pLabelSpin = new QLabel(tr("Filter size"));

            auto pLabelBorder = new QLabel(tr("Border type"));
            m_pComboBorder = new QComboBox;
            m_pComboBorder->addItem("Default", cv::BORDER_DEFAULT);
            m_pComboBorder->addItem("Replicate", cv::BORDER_REPLICATE);
            m_pComboBorder->setCurrentIndex(m_pComboBorder->findData(m_pParam->m_borderType));

            m_pLayout->addWidget(pLabelSpin, 0, 0);
            m_pLayout->addWidget(m_pSpin, 0, 1);
            m_pLayout->addWidget(pLabelBorder, 2, 0);
            m_pLayout->addWidget(m_pComboBorder, 2, 1);
        }

        void onApply() override
        {
            m_pParam->m_ksize = cv::Size(m_pSpin->value(), m_pSpin->value());
            m_pParam->m_borderType = m_pComboBorder->currentData().toInt();
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvBlurParam> m_pParam = nullptr;
        QSpinBox*                           m_pSpin = nullptr;
        QComboBox*                          m_pComboBorder = nullptr;
};

class COcvWidgetBlurFactory : public CWidgetFactory
{
    public:

        COcvWidgetBlurFactory()
        {
            m_name = "ocv_blur";
        }

        virtual WorkflowTaskWidgetPtr   create(std::shared_ptr<CWorkflowTaskParam> pParam)
        {
            return std::make_shared<COcvWidgetBlur>(pParam);
        }
};

#endif // COCVWIDGETBLUR_HPP
