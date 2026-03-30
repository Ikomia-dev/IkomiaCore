#ifndef CTASKIOFACTORYWRAP_H
#define CTASKIOFACTORYWRAP_H

#include "PyCoreGlobal.h"
#include "Workflow/CWorkflowTaskIO.h"


class CTaskIOFactoryWrap: public CWorkflowTaskIOFactory, public wrapper<CWorkflowTaskIOFactory>
{
    public:

        ~CTaskIOFactoryWrap() = default;

        std::vector<IODataType> getValidDataTypes() const override;

        WorkflowTaskIOPtr       create(IODataType dataType) override;
};

#endif // CTASKIOFACTORYWRAP_H
