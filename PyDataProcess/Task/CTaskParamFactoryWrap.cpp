#include "CTaskParamFactoryWrap.h"
#include "PythonThread.hpp"

WorkflowTaskParamPtr CTaskParamFactoryWrap::create()
{
    CPyEnsureGIL gil;
    try
    {
        return this->get_override("create")();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
