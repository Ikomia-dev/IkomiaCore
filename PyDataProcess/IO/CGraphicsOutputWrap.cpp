#include "CGraphicsOutputWrap.h"

CGraphicsOutputWrap::CGraphicsOutputWrap(): CGraphicsOutput()
{
}

CGraphicsOutputWrap::CGraphicsOutputWrap(const std::string &name): CGraphicsOutput(name)
{
}

CGraphicsOutputWrap::CGraphicsOutputWrap(const CGraphicsOutput &io): CGraphicsOutput(io)
{
}

std::string CGraphicsOutputWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CGraphicsOutput::toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
