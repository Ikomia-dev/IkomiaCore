#ifndef CSEMANTICSEGFILTER_H
#define CSEMANTICSEGFILTER_H

#include "Task/CSemanticSegTask.h"
#include "Core/CTaskFactory.hpp"
#include "Workflow/CWorkflowTaskWidget.h"
#include "Core/CWidgetFactory.hpp"

//-----------------------------------//
//----- CSemanticSegFilterParam -----//
//-----------------------------------//
class CSemanticSegFilterParam: public CWorkflowTaskParam
{
    public:

        CSemanticSegFilterParam();
        ~CSemanticSegFilterParam() = default;

        void        setParamMap(const UMapString& paramMap) override;

        UMapString  getParamMap() const override;

    public:

        std::string m_categories = "all";
};

//------------------------------//
//----- CSemanticSegFilter -----//
//------------------------------//
class CSemanticSegFilter: public CSemanticSegTask
{
    public:

        CSemanticSegFilter();
        CSemanticSegFilter(const std::string name, const std::shared_ptr<CSemanticSegFilterParam>& pParam);

        size_t  getProgressSteps() override;

        void    run() override;

    private:

        void    initIO();
};

//-------------------------------------//
//----- CSemanticSegFilterFactory -----//
//-------------------------------------//
class CSemanticSegFilterFactory : public CTaskFactory
{
    public:

        CSemanticSegFilterFactory();
        ~CSemanticSegFilterFactory() = default;

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override;
        virtual WorkflowTaskPtr create() override;
};

//------------------------------------//
//----- CWidgetSemanticSegFilter -----//
//------------------------------------//
class CWidgetSemanticSegFilter : public CWorkflowTaskWidget
{
    public:

        CWidgetSemanticSegFilter(QWidget *parent = Q_NULLPTR);
        CWidgetSemanticSegFilter(const std::shared_ptr<CWorkflowTaskParam>& pParam, QWidget *parent = Q_NULLPTR);

    protected:

        void    init();

        void    onApply() override;

    private:

        std::shared_ptr<CSemanticSegFilterParam>    m_pParam = nullptr;
        QDoubleSpinBox*                             m_pSpinConfidence = nullptr;
        QLineEdit*                                  m_pEditCategories = nullptr;
};

//-------------------------------------------//
//----- CWidgetSemanticSegFilterFactory -----//
//-------------------------------------------//
class CWidgetSemanticSegFilterFactory : public CWidgetFactory
{
    public:

        CWidgetSemanticSegFilterFactory();
        ~CWidgetSemanticSegFilterFactory() = default;

        virtual WorkflowTaskWidgetPtr   create(const std::shared_ptr<CWorkflowTaskParam>& pParam);
};

#endif // CSEMANTICSEGFILTER_H
