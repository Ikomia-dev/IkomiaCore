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

#include "CWorkflowTaskWidgetWrap.h"
#include <QLayout>
#include <QDebug>

CWorkflowTaskWidgetWrap::CWorkflowTaskWidgetWrap() : CWorkflowTaskWidget(nullptr)
{
}

CWorkflowTaskWidgetWrap::CWorkflowTaskWidgetWrap(QWidget *parent) : CWorkflowTaskWidget(parent)
{
}

void CWorkflowTaskWidgetWrap::onApply()
{
    CPyEnsureGIL gil;
    try
    {
        this->get_override("on_apply")();
    }
    catch(boost::python::error_already_set&)
    {
        //Do not throw exceptions from slot
        Utils::print(Utils::Python::handlePythonException(), QtCriticalMsg);
    }
}

void CWorkflowTaskWidgetWrap::onParametersChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override onParamsModifiedOver = this->get_override("on_parameters_changed"))
            onParamsModifiedOver();
        else
            CWorkflowTaskWidget::onParametersChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::default_onParametersModified()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTaskWidget::onParametersChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::setLayout(long long layoutPtr)
{
    CPyEnsureGIL gil;
    try
    {
        auto pLayout = reinterpret_cast<QLayout*>(layoutPtr);
        m_pContainerLayout->insertLayout(0, pLayout);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::emitApply(const WorkflowTaskParamPtr& paramPtr)
{
    CPyEnsureGIL gil;
    try
    {
        emit doApplyProcess(paramPtr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::emitSendProcessAction(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        emit doSendProcessAction(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::emitSetGraphicsTool(GraphicsShape tool)
{
    CPyEnsureGIL gil;
    try
    {
        emit doSetGraphicsTool(tool);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWidgetWrap::emitSetGraphicsCategory(const std::string &category)
{
    CPyEnsureGIL gil;
    try
    {
        emit doSetGraphicsCategory(QString::fromStdString(category));
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

