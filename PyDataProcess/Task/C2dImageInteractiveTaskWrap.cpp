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

#include "C2dImageInteractiveTaskWrap.h"

//------------------------------------------------------//
//-----------  C2dImageInteractiveTaskWrap  ------------//
//------------------------------------------------------//
C2dImageInteractiveTaskWrap::C2dImageInteractiveTaskWrap() : C2dImageInteractiveTask()
{
}

C2dImageInteractiveTaskWrap::C2dImageInteractiveTaskWrap(const std::string &name) : C2dImageInteractiveTask(name)
{
}

C2dImageInteractiveTaskWrap::C2dImageInteractiveTaskWrap(const C2dImageInteractiveTask &process) : C2dImageInteractiveTask(process)
{
}

std::string C2dImageInteractiveTaskWrap::repr() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override reprOver = this->get_override("__repr__"))
            return reprOver();

        return C2dImageInteractiveTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string C2dImageInteractiveTaskWrap::default_repr() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->C2dImageInteractiveTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t C2dImageInteractiveTaskWrap::getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getProgressStepsOver = this->get_override("get_progress_steps"))
            return getProgressStepsOver();

        return C2dImageInteractiveTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t C2dImageInteractiveTaskWrap::default_getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        return this->C2dImageInteractiveTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setActiveOver = this->get_override("set_active"))
            setActiveOver(bActive);
        else
            C2dImageInteractiveTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        if(override updateStaticOutputsOver = this->get_override("update_static_outputs"))
            updateStaticOutputsOver();
        else
            C2dImageInteractiveTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override beginTaskRunOver = this->get_override("begin_task_run"))
            beginTaskRunOver();
        else
            C2dImageInteractiveTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override endTaskRunOver = this->get_override("end_task_run"))
            endTaskRunOver();
        else
            C2dImageInteractiveTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        if(override executeActionsOver = this->get_override("execute_actions"))
            executeActionsOver(flags);
        else
            C2dImageInteractiveTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::run()
{
    CPyEnsureGIL gil;
    try
    {
        if(override runOver = this->get_override("run"))
            runOver();
        else
            C2dImageInteractiveTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_run()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::stop()
{
    CPyEnsureGIL gil;
    try
    {
        if(override stopOver = this->get_override("stop"))
            stopOver();
        else
            C2dImageInteractiveTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_stop()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override graphicsChangedOver = this->get_override("graphics_changed"))
            graphicsChangedOver();
        else
            C2dImageInteractiveTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        if(override globalInputChangedOver = this->get_override("global_input_changed"))
            globalInputChangedOver(bNewSequence);
        else
            C2dImageInteractiveTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::default_globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageInteractiveTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageInteractiveTaskWrap::emitAddSubProgressSteps(int count)
{
    emit m_signalHandler->doAddSubTotalSteps(count);
}

void C2dImageInteractiveTaskWrap::emitGraphicsContextChanged()
{
    emit m_signalHandler->doGraphicsContextChanged();
}

void C2dImageInteractiveTaskWrap::emitStepProgress()
{
    emit m_signalHandler->doProgress();
}

void C2dImageInteractiveTaskWrap::emitOutputChanged()
{
    emit m_signalHandler->doOutputChanged();
}

void C2dImageInteractiveTaskWrap::emitParametersChanged()
{
    emit m_signalHandler->doParametersChanged();
}


