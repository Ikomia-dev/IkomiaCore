#ifndef CTASKIOFACTORYWRAP_H
#define CTASKIOFACTORYWRAP_H

#include "PyCoreGlobal.h"
#include "Workflow/CWorkflowTaskIO.h"


class CTaskIOFactoryWrap: public CWorkflowTaskIOFactory, public wrapper<CWorkflowTaskIOFactory>
{
    public:

        ~CTaskIOFactoryWrap() = default;

        std::vector<IODataTypeEx>   getValidDataTypes() const override;

        WorkflowTaskIOPtr           create(IODataTypeEx dataType) override;
};

#endif // CTASKIOFACTORYWRAP_H
