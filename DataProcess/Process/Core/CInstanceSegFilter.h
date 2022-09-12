#ifndef CINSTANCESEGFILTER_H
#define CINSTANCESEGFILTER_H

#include "Workflow/CWorkflowTask.h"
#include "Core/CTaskFactory.hpp"
#include "Workflow/CWorkflowTaskWidget.h"
#include "Core/CWidgetFactory.hpp"

//-----------------------------------//
//----- CInstanceSegFilterParam -----//
//-----------------------------------//
class CInstanceSegFilterParam: public CWorkflowTaskParam
{
    public:

        CInstanceSegFilterParam();
        ~CInstanceSegFilterParam() = default;

        void        setParamMap(const UMapString& paramMap) override;

        UMapString  getParamMap() const override;

    public:

        float       m_confidence = 0.5;
        std::string m_categories = "all";
};

//------------------------------//
//----- CInstanceSegFilter -----//
//------------------------------//
class CInstanceSegFilter: public CWorkflowTask
{
    public:

        CInstanceSegFilter();
        CInstanceSegFilter(const std::string name, const std::shared_ptr<CInstanceSegFilterParam>& pParam);

        size_t  getProgressSteps() override;

        void    run() override;
};

//-------------------------------------//
//----- CInstanceSegFilterFactory -----//
//-------------------------------------//
class CInstanceSegFilterFactory : public CTaskFactory
{
    public:

        CInstanceSegFilterFactory();
        ~CInstanceSegFilterFactory() = default;

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override;
        virtual WorkflowTaskPtr create() override;
};

//------------------------------------//
//----- CWidgetInstanceSegFilter -----//
//------------------------------------//
class CWidgetInstanceSegFilter : public CWorkflowTaskWidget
{
    public:

        CWidgetInstanceSegFilter(QWidget *parent = Q_NULLPTR);
        CWidgetInstanceSegFilter(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent = Q_NULLPTR);

    protected:

        void    init();

        void    onApply() override;

    private:

        std::shared_ptr<CInstanceSegFilterParam>    m_pParam = nullptr;
        QDoubleSpinBox*                             m_pSpinConfidence = nullptr;
        QLineEdit*                                  m_pEditCategories = nullptr;
};

//-------------------------------------------//
//----- CWidgetInstanceSegFilterFactory -----//
//-------------------------------------------//
class CWidgetInstanceSegFilterFactory : public CWidgetFactory
{
    public:

        CWidgetInstanceSegFilterFactory();
        ~CWidgetInstanceSegFilterFactory() = default;

        virtual WorkflowTaskWidgetPtr   create(std::shared_ptr<CWorkflowTaskParam> pParam);
};

#endif // CINSTANCESEGFILTER_H
