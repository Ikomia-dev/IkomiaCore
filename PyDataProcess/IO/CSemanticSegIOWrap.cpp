#include "CSemanticSegIOWrap.h"

CSemanticSegIOWrap::CSemanticSegIOWrap(): CSemanticSegIO()
{
}

CSemanticSegIOWrap::CSemanticSegIOWrap(const CSemanticSegIO &io): CSemanticSegIO(io)
{
}

bool CSemanticSegIOWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();
        return CSemanticSegIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CSemanticSegIOWrap::default_isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CSemanticSegIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CSemanticSegIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        this->CSemanticSegIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            CSemanticSegIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::default_load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CSemanticSegIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            CSemanticSegIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::default_save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CSemanticSegIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CSemanticSegIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CSemanticSegIO::toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CSemanticSegIOWrap::default_toJsonNoOpt() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CSemanticSegIO::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CSemanticSegIOWrap::toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return CSemanticSegIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CSemanticSegIOWrap::default_toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CSemanticSegIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            CSemanticSegIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CSemanticSegIOWrap::default_fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CSemanticSegIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
