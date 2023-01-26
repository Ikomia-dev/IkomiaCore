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

#include "CImageIOWrap.h"

CImageIOWrap::CImageIOWrap() : CImageIO()
{
}

CImageIOWrap::CImageIOWrap(IODataType dataType) : CImageIO(dataType)
{
}

CImageIOWrap::CImageIOWrap(IODataType dataType, const CMat &image) : CImageIO(dataType, image)
{
}

CImageIOWrap::CImageIOWrap(IODataType dataType, const CMat &image, const std::string& name) : CImageIO(dataType, image, name)
{
}

CImageIOWrap::CImageIOWrap(IODataType datatype, const std::string &name) : CImageIO(datatype, name)
{
}

CImageIOWrap::CImageIOWrap(IODataType datatype, const std::string& name, const std::string &path) : CImageIO(datatype, name, path)
{
}

CImageIOWrap::CImageIOWrap(const CImageIO &io) : CImageIO(io)
{
}

size_t CImageIOWrap::getUnitElementCount() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override getOver = this->get_override("get_unit_element_count"))
            return getOver();
        return CImageIO::getUnitElementCount();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CImageIOWrap::default_getUnitElementCount() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CImageIO::getUnitElementCount();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

CMat CImageIOWrap::getImage()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getImageOver = this->get_override("get_image"))
            return getImageOver();
        return CImageIO::getImage();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

CMat CImageIOWrap::default_getImage()
{
    CPyEnsureGIL gil;
    try
    {
        return this->CImageIO::getImage();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CImageIOWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();
        return CImageIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CImageIOWrap::default_isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CImageIO::isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CImageIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        this->CImageIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::copyStaticData(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override copyOver = this->get_override("copy_static_data"))
            copyOver(ioPtr);
        else
            CImageIO::copyStaticData(ioPtr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::default_copyStaticData(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CImageIO::copyStaticData(ioPtr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            CImageIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::default_load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CImageIO::load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            CImageIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::default_save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        this->CImageIO::save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CImageIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver();
        else
            return CImageIO::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CImageIOWrap::default_toJsonNoOpt() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CImageIO::toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CImageIOWrap::toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return CImageIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CImageIOWrap::default_toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CImageIO::toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            CImageIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CImageIOWrap::default_fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        this->CImageIO::fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
