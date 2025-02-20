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

#include "CPluginProcessInterfaceWrap.h"

std::shared_ptr<CTaskFactory> CPluginProcessInterfaceWrap::getProcessFactory()
{
    CPyEnsureGIL gil;
    try
    {
        return this->get_override("get_process_factory")();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::shared_ptr<CWidgetFactory> CPluginProcessInterfaceWrap::getWidgetFactory()
{
    CPyEnsureGIL gil;
    try
    {
        return this->get_override("get_widget_factory")();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::shared_ptr<CTaskParamFactory> CPluginProcessInterfaceWrap::getParamFactory()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getParamFactoryOver = this->get_override("get_param_factory"))
            return getParamFactoryOver();

        return this->CPluginProcessInterface::getParamFactory();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::shared_ptr<CTaskParamFactory> CPluginProcessInterfaceWrap::default_getParamFactory()
{
    CPyEnsureGIL gil;
    try
    {
        return this->CPluginProcessInterface::getParamFactory();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}
