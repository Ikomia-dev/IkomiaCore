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

#include "CVideoTaskWrap.h"

//--------------------------------------------//
//-----------  CVideoTaskWrap  ------------//
//--------------------------------------------//
CVideoTaskWrap::CVideoTaskWrap() : CVideoTask()
{
}

CVideoTaskWrap::CVideoTaskWrap(const std::string &name) : CVideoTask(name)
{
}

CVideoTaskWrap::CVideoTaskWrap(const CVideoTask &process) : CVideoTask(process)
{
}

std::string CVideoTaskWrap::repr() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override reprOver = this->get_override("__repr__"))
            return reprOver();

        return CVideoTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CVideoTaskWrap::default_repr() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CVideoTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CVideoTaskWrap::getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getProgressStepsOver = this->get_override("get_progress_steps"))
            return getProgressStepsOver();

        return CVideoTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CVideoTaskWrap::default_getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        return this->CVideoTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setActiveOver = this->get_override("set_active"))
            setActiveOver(bActive);
        else
            CVideoTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        if(override updateStaticOutputsOver = this->get_override("update_static_outputs"))
            updateStaticOutputsOver();
        else
            CVideoTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override beginTaskRunOver = this->get_override("begin_task_run"))
            beginTaskRunOver();
        else
            CVideoTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override endTaskRunOver = this->get_override("end_task_run"))
            endTaskRunOver();
        else
            CVideoTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::run()
{
    CPyEnsureGIL gil;
    try
    {
        if(override runOver = this->get_override("run"))
            runOver();
        else
            CVideoTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_run()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::stop()
{
    CPyEnsureGIL gil;
    try
    {
        if(override stopOver = this->get_override("stop"))
            stopOver();
        else
            CVideoTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_stop()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override graphicsChangedOver = this->get_override("graphics_changed"))
            graphicsChangedOver();
        else
            CVideoTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        if(override globalInputChangedOver = this->get_override("global_input_changed"))
            globalInputChangedOver(bNewSequence);
        else
            CVideoTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        if(override executeActionsOver = this->get_override("execute_actions"))
            executeActionsOver(flags);
        else
            CVideoTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::notifyVideoStart(int frameCount)
{
    CPyEnsureGIL gil;
    try
    {
        if(override notifyVideoStartOver = this->get_override("notify_video_start"))
            notifyVideoStartOver(frameCount);
        else
            CVideoTask::notifyVideoStart(frameCount);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_notifyVideoStart(int frameCount)
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::notifyVideoStart(frameCount);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::notifyVideoEnd()
{
    CPyEnsureGIL gil;
    try
    {
        if(override notifyVideoEndOver = this->get_override("notify_video_end"))
            notifyVideoEndOver();
        else
            CVideoTask::notifyVideoEnd();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::default_notifyVideoEnd()
{
    CPyEnsureGIL gil;
    try
    {
        this->CVideoTask::notifyVideoEnd();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CVideoTaskWrap::emitAddSubProgressSteps(int count)
{
    emit m_signalHandler->doAddSubTotalSteps(count);
}

void CVideoTaskWrap::emitGraphicsContextChanged()
{
    emit m_signalHandler->doGraphicsContextChanged();
}

void CVideoTaskWrap::emitStepProgress()
{
    emit m_signalHandler->doProgress();
}

void CVideoTaskWrap::emitOutputChanged()
{
    emit m_signalHandler->doOutputChanged();
}

void CVideoTaskWrap::emitParametersChanged()
{
    emit m_signalHandler->doParametersChanged();
}
