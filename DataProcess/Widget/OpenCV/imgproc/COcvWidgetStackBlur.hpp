#ifndef COCVWIDGETSTACKBLUR_HPP
#define COCVWIDGETSTACKBLUR_HPP

#include "Core/CWidgetFactory.hpp"
#include "Process/OpenCV/imgproc/COcvStackBlur.hpp"
#include <QSpinBox>
#include <QLabel>

class COcvWidgetStackBlur : public CWorkflowTaskWidget
{
    public:

        COcvWidgetStackBlur(QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            init();
        }

        COcvWidgetStackBlur(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent = Q_NULLPTR) : CWorkflowTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<COcvStackBlurParam>(pParam);
            init();
        }

    protected:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<COcvStackBlurParam>();

            m_pSpin = new QSpinBox;
            m_pSpin->setValue(m_pParam->m_ksize.width);
            QLabel* pLabelSpin = new QLabel(tr("Filter size"));

            m_pLayout->addWidget(pLabelSpin, 0, 0);
            m_pLayout->addWidget(m_pSpin, 0, 1);
        }

        void onApply() override
        {
            m_pParam->m_ksize = cv::Size(m_pSpin->value(), m_pSpin->value());
            emit doApplyProcess(m_pParam);
        }

    private:

        std::shared_ptr<COcvStackBlurParam> m_pParam = nullptr;
        QSpinBox*                           m_pSpin = nullptr;
};

class COcvWidgetStackBlurFactory : public CWidgetFactory
{
    public:

        COcvWidgetStackBlurFactory()
        {
            m_name = "ocv_stack_blur";
        }

        virtual WorkflowTaskWidgetPtr   create(std::shared_ptr<CWorkflowTaskParam> pParam)
        {
            return std::make_shared<COcvWidgetStackBlur>(pParam);
        }
};

#endif // COCVWIDGETSTACKBLUR_HPP
