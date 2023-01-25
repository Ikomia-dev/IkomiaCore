#include "CInstanceSegIOWrap.h"

CInstanceSegIOWrap::CInstanceSegIOWrap(): CInstanceSegIO()
{
}

CInstanceSegIOWrap::CInstanceSegIOWrap(const CInstanceSegIO &io): CInstanceSegIO(io)
{
}

bool CInstanceSegIOWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();
        return CInstanceSegIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CInstanceSegIOWrap::default_isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CInstanceSegIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CInstanceSegIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        this->CInstanceSegIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            CInstanceSegIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::default_load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CInstanceSegIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            CInstanceSegIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::default_save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CInstanceSegIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CInstanceSegIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CInstanceSegIO::toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CInstanceSegIOWrap::default_toJsonNoOpt() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CInstanceSegIO::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CInstanceSegIOWrap::toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return CInstanceSegIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CInstanceSegIOWrap::default_toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CInstanceSegIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            CInstanceSegIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CInstanceSegIOWrap::default_fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CInstanceSegIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
