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

#include "C2dImageTaskWrap.h"

//----------------------------------------------//
//-----------  C2dImageTaskWrap  ------------//
//----------------------------------------------//
C2dImageTaskWrap::C2dImageTaskWrap() : C2dImageTask()
{
}

C2dImageTaskWrap::C2dImageTaskWrap(bool bGraphicsInput) : C2dImageTask(bGraphicsInput)
{
}

C2dImageTaskWrap::C2dImageTaskWrap(const std::string &name) : C2dImageTask(name)
{
}

C2dImageTaskWrap::C2dImageTaskWrap(const std::string &name, bool bGraphicsInput) : C2dImageTask(name, bGraphicsInput)
{
}

C2dImageTaskWrap::C2dImageTaskWrap(const C2dImageTask &process) : C2dImageTask(process)
{
}

size_t C2dImageTaskWrap::getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getProgressStepsOver = this->get_override("get_progress_steps"))
            return getProgressStepsOver();

        return C2dImageTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t C2dImageTaskWrap::default_getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        return this->C2dImageTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setActiveOver = this->get_override("set_active"))
            setActiveOver(bActive);
        else
            C2dImageTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

CMat C2dImageTaskWrap::applyGraphicsMask(const CMat &src, int maskIndex)
{
    CPyEnsureGIL gil;
    try
    {
        CMat res = src.clone();
        this->C2dImageTask::applyGraphicsMask(src, res, (size_t)(maskIndex));
        return res;
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

CMat C2dImageTaskWrap::applyGraphicsMaskToBinary(const CMat &src, int maskIndex)
{
    CPyEnsureGIL gil;
    try
    {
        CMat res = src.clone();
        this->C2dImageTask::applyGraphicsMaskToBinary(src, res, (size_t)(maskIndex));
        return res;
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        if(override updateStaticOutputsOver = this->get_override("update_static_outputs"))
            updateStaticOutputsOver();
        else
            C2dImageTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override beginTaskRunOver = this->get_override("begin_task_run"))
            beginTaskRunOver();
        else
            C2dImageTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override endTaskRunOver = this->get_override("end_task_run"))
            endTaskRunOver();
        else
            C2dImageTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        if(override executeActionsOver = this->get_override("execute_actions"))
            executeActionsOver(flags);
        else
            C2dImageTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::run()
{
    CPyEnsureGIL gil;
    try
    {
        if(override runOver = this->get_override("run"))
            runOver();
        else
            C2dImageTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_run()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::stop()
{
    CPyEnsureGIL gil;
    try
    {
        if(override stopOver = this->get_override("stop"))
            stopOver();
        else
            C2dImageTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_stop()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override graphicsChangedOver = this->get_override("graphics_changed"))
            graphicsChangedOver();
        else
            C2dImageTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        if(override globalInputChangedOver = this->get_override("global_input_changed"))
            globalInputChangedOver(bNewSequence);
        else
            C2dImageTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::default_globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        this->C2dImageTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void C2dImageTaskWrap::emitAddSubProgressSteps(int count)
{
    emit m_signalHandler->doAddSubTotalSteps(count);
}

void C2dImageTaskWrap::emitStepProgress()
{
    emit m_signalHandler->doProgress();
}

void C2dImageTaskWrap::emitGraphicsContextChanged()
{
    emit m_signalHandler->doGraphicsContextChanged();
}

void C2dImageTaskWrap::emitOutputChanged()
{
    emit m_signalHandler->doOutputChanged();
}

