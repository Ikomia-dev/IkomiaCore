// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CGraphicsInputWrap.h"

CGraphicsInputWrap::CGraphicsInputWrap() : CGraphicsInput()
{    
}

CGraphicsInputWrap::CGraphicsInputWrap(const std::string &name) : CGraphicsInput(name)
{
}

CGraphicsInputWrap::CGraphicsInputWrap(const CGraphicsInput &io) : CGraphicsInput(io)
{
}

bool CGraphicsInputWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();

        return CGraphicsInput::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CGraphicsInputWrap::default_isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CGraphicsInput::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CGraphicsInput::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        this->CGraphicsInput::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            CGraphicsInput::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::default_load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CGraphicsInput::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            CGraphicsInput::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::default_save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CGraphicsInput::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CGraphicsInputWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver();
        else
            return CGraphicsInput::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CGraphicsInputWrap::default_toJsonNoOpt() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CGraphicsInput::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CGraphicsInputWrap::toJson(const std::vector<std::string>& options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return CGraphicsInput::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CGraphicsInputWrap::default_toJson(const std::vector<std::string>& options) const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CGraphicsInput::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            CGraphicsInput::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CGraphicsInputWrap::default_fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CGraphicsInput::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

