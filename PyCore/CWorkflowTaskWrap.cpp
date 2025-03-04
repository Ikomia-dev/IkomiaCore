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

#include "CWorkflowTaskWrap.h"
#include "PyCoreTools.hpp"

//----------------------------------------------//
//------------  CWorkflowTaskWrap  -------------//
//----------------------------------------------//
CWorkflowTaskWrap::CWorkflowTaskWrap() : CWorkflowTask()
{
}

CWorkflowTaskWrap::CWorkflowTaskWrap(const std::string &name) : CWorkflowTask(name)
{
}

CWorkflowTaskWrap::CWorkflowTaskWrap(const CWorkflowTask &task) : CWorkflowTask(task)
{
}

void CWorkflowTaskWrap::initLongProcess()
{
    CPyEnsureGIL gil;
    try
    {
        if(override initOver = this->get_override("init_long_process"))
            initOver();
        else
            CWorkflowTask::initLongProcess();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_initLongProcess()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::initLongProcess();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setInputDataType(const IODataType &dataType, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setOver = this->get_override("set_input_data_type"))
            setOver(dataType, index);
        else
            CWorkflowTask::setInputDataType(dataType, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setInputDataType(const IODataType &dataType, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setInputDataType(dataType, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setInput(const WorkflowTaskIOPtr &pInput, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setInputOver = this->get_override("set_input"))
            setInputOver(pInput, index);
        else
            CWorkflowTask::setInput(pInput, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setInput(const WorkflowTaskIOPtr &pInput, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setInput(pInput, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setInputs(const InputOutputVect &inputs)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setInputsOver = this->get_override("set_inputs"))
            setInputsOver(inputs);
        else
            CWorkflowTask::setInputs(inputs);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setInputs(const InputOutputVect &inputs)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setInputs(inputs);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setOutputDataType(const IODataType &dataType, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setOutputDataOver = this->get_override("set_output_data_type"))
            setOutputDataOver(dataType, index);
        else
            CWorkflowTask::setOutputDataType(dataType, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setOutputDataType(const IODataType &dataType, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setOutputDataType(dataType, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setOutput(const WorkflowTaskIOPtr &pOutput, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setOutputOver = this->get_override("set_output"))
            setOutputOver(pOutput, index);
        else
            CWorkflowTask::setOutput(pOutput, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setOutput(const WorkflowTaskIOPtr &pOutput, size_t index)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setOutput(pOutput, index);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setOutputs(const InputOutputVect &outputs)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setOutputsOver = this->get_override("set_outputs"))
            setOutputsOver(outputs);
        else
            CWorkflowTask::setOutputs(outputs);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setOutputs(const InputOutputVect &outputs)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setOutputs(outputs);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setActiveOver = this->get_override("set_active"))
            setActiveOver(bActive);
        else
            CWorkflowTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CWorkflowTaskWrap::getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getProgressStepsOver = this->get_override("get_progress_steps"))
            return getProgressStepsOver();

        return CWorkflowTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CWorkflowTaskWrap::default_getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        return this->CWorkflowTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CWorkflowTaskWrap::isGraphicsChangedListening() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isGraphicsChangedOver = this->get_override("is_graphics_changed_listening"))
            return isGraphicsChangedOver();

        return CWorkflowTask::isGraphicsChangedListening();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CWorkflowTaskWrap::default_isGraphicsChangedListening() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CWorkflowTask::isGraphicsChangedListening();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::addInput(const WorkflowTaskIOPtr &pInput)
{
    CPyEnsureGIL gil;
    try
    {
        if(override addInputOver = this->get_override("add_input"))
            addInputOver(pInput);
        else
            CWorkflowTask::addInput(pInput);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_addInput(const WorkflowTaskIOPtr &pInput)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::addInput(pInput);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::addOutput(const WorkflowTaskIOPtr &pOutput)
{
    CPyEnsureGIL gil;
    try
    {
        if(override addOutputOver = this->get_override("add_output"))
            addOutputOver(pOutput);
        else
            CWorkflowTask::addOutput(pOutput);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_addOutput(const WorkflowTaskIOPtr &pOutput)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::addOutput(pOutput);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::run()
{
    CPyEnsureGIL gil;
    try
    {
        if(override runOver = this->get_override("run"))
            runOver();
        else
            CWorkflowTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_run()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::stop()
{
    CPyEnsureGIL gil;
    try
    {
        if(override stopOver = this->get_override("stop"))
            stopOver();
        else
            CWorkflowTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_stop()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        if(override executeActionsOver = this->get_override("execute_actions"))
            executeActionsOver(flags);
        else
            CWorkflowTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        if(override updateStaticOutputsOver = this->get_override("update_static_outputs"))
            updateStaticOutputsOver();
        else
            CWorkflowTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override beginTaskRunOver = this->get_override("begin_task_run"))
            beginTaskRunOver();
        else
            CWorkflowTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override endTaskRunOver = this->get_override("end_task_run"))
            endTaskRunOver();
        else
            CWorkflowTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::parametersModified()
{
    CPyEnsureGIL gil;
    try
    {
        if(override parametersModifiedOver = this->get_override("parameters_modified"))
            parametersModifiedOver();
        else
            CWorkflowTask::parametersModified();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_parametersModified()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::parametersModified();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override graphicsChangedOver = this->get_override("graphics_changed"))
            graphicsChangedOver();
        else
            CWorkflowTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        if(override globalInputChangedOver = this->get_override("global_input_changed"))
            globalInputChangedOver(bNewSequence);
        else
            CWorkflowTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::workflowStarted()
{
    CPyEnsureGIL gil;
    try
    {
        if(override workflowStartedOver = this->get_override("workflow_started"))
            workflowStartedOver();
        else
            CWorkflowTask::workflowStarted();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_workflowStarted()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::workflowStarted();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::workflowFinished()
{
    CPyEnsureGIL gil;
    try
    {
        if(override workflowFinishedOver = this->get_override("workflow_finished"))
            workflowFinishedOver();
        else
            CWorkflowTask::workflowFinished();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::default_workflowFinished()
{
    CPyEnsureGIL gil;
    try
    {
        this->CWorkflowTask::workflowFinished();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CWorkflowTaskWrap::emitAddSubProgressSteps(int count)
{
    m_signalHandler->doAddSubTotalSteps(count);
}

void CWorkflowTaskWrap::emitGraphicsContextChanged()
{
    emit m_signalHandler->doGraphicsContextChanged();
}

void CWorkflowTaskWrap::emitStepProgress()
{
    emit m_signalHandler->doProgress();
}

void CWorkflowTaskWrap::emitOutputChanged()
{
    emit m_signalHandler->doOutputChanged();
}

void CWorkflowTaskWrap::emitParametersChanged()
{
    emit m_signalHandler->doParametersChanged();
}

