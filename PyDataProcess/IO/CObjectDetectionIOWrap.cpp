#include "CObjectDetectionIOWrap.h"

CObjectDetectionIOWrap::CObjectDetectionIOWrap() : CObjectDetectionIO()
{
}

CObjectDetectionIOWrap::CObjectDetectionIOWrap(const CObjectDetectionIO &io): CObjectDetectionIO(io)
{
}

bool CObjectDetectionIOWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();
        return CObjectDetectionIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CObjectDetectionIOWrap::default_isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CObjectDetectionIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CObjectDetectionIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            CObjectDetectionIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::default_load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            CObjectDetectionIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::default_save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CObjectDetectionIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return CObjectDetectionIO::toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CObjectDetectionIOWrap::default_toJsonNoOpt() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CObjectDetectionIO::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CObjectDetectionIOWrap::toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return CObjectDetectionIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CObjectDetectionIOWrap::default_toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CObjectDetectionIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            CObjectDetectionIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjectDetectionIOWrap::default_fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
