#include "CBlobMeasureIOWrap.h"

CBlobMeasureIOWrap::CBlobMeasureIOWrap(): CBlobMeasureIO()
{
}

CBlobMeasureIOWrap::CBlobMeasureIOWrap(const std::string &name): CBlobMeasureIO(name)
{
}

CBlobMeasureIOWrap::CBlobMeasureIOWrap(const CBlobMeasureIO &io): CBlobMeasureIO(io)
{
}

std::string CBlobMeasureIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CBlobMeasureIO::toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
