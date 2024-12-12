#ifndef CTASKPARAMFACTORYWRAP_H
#define CTASKPARAMFACTORYWRAP_H

#include "PyDataProcessGlobal.h"
#include "Core/CTaskParamFactory.hpp"

class CTaskParamFactoryWrap: public CTaskParamFactory, public wrapper<CTaskParamFactory>
{
    public:

        ~CTaskParamFactoryWrap() = default;

        WorkflowTaskParamPtr create() override;
};

#endif // CTASKPARAMFACTORYWRAP_H
