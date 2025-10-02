#include "CObjDetectTaskWrap.h"

CObjDetectTaskWrap::CObjDetectTaskWrap()
{
}

CObjDetectTaskWrap::CObjDetectTaskWrap(const std::string &name) : CObjectDetectionTask(name)
{
}

CObjDetectTaskWrap::CObjDetectTaskWrap(const CObjectDetectionTask &task) : CObjectDetectionTask(task)
{
}

std::string CObjDetectTaskWrap::repr() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override reprOver = this->get_override("__repr__"))
            return reprOver();

        return CObjectDetectionTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CObjDetectTaskWrap::default_repr() const
{
    CPyEnsureGIL gil;
    try
    {
        return this->CObjectDetectionTask::repr();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::initLongProcess()
{
    CPyEnsureGIL gil;
    try
    {
        if(override initOver = this->get_override("init_long_process"))
            initOver();
        else
            CObjDetectTaskWrap::initLongProcess();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_initLongProcess()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjDetectTaskWrap::initLongProcess();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CObjDetectTaskWrap::getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        if(override getProgressStepsOver = this->get_override("get_progress_steps"))
            return getProgressStepsOver();

        return CObjectDetectionTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

size_t CObjDetectTaskWrap::default_getProgressSteps()
{
    CPyEnsureGIL gil;
    try
    {
        return this->CObjectDetectionTask::getProgressSteps();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        if(override setActiveOver = this->get_override("set_active"))
            setActiveOver(bActive);
        else
            CObjectDetectionTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_setActive(bool bActive)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::setActive(bActive);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        if(override updateStaticOutputsOver = this->get_override("update_static_outputs"))
            updateStaticOutputsOver();
        else
            CObjectDetectionTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_updateStaticOutputs()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::updateStaticOutputs();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override beginTaskRunOver = this->get_override("begin_task_run"))
            beginTaskRunOver();
        else
            CObjectDetectionTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_beginTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::beginTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        if(override endTaskRunOver = this->get_override("end_task_run"))
            endTaskRunOver();
        else
            CObjectDetectionTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_endTaskRun()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::endTaskRun();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        if(override executeActionsOver = this->get_override("execute_actions"))
            executeActionsOver(flags);
        else
            CObjectDetectionTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_executeActions(int flags)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::executeActions(flags);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::run()
{
    CPyEnsureGIL gil;
    try
    {
        if(override runOver = this->get_override("run"))
            runOver();
        else
            CObjectDetectionTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_run()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::run();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::stop()
{
    CPyEnsureGIL gil;
    try
    {
        if(override stopOver = this->get_override("stop"))
            stopOver();
        else
            CObjectDetectionTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_stop()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::stop();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        if(override graphicsChangedOver = this->get_override("graphics_changed"))
            graphicsChangedOver();
        else
            CObjectDetectionTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_graphicsChanged()
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::graphicsChanged();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        if(override globalInputChangedOver = this->get_override("global_input_changed"))
            globalInputChangedOver(bNewSequence);
        else
            CObjectDetectionTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::default_globalInputChanged(bool bNewSequence)
{
    CPyEnsureGIL gil;
    try
    {
        this->CObjectDetectionTask::globalInputChanged(bNewSequence);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CObjDetectTaskWrap::emitAddSubProgressSteps(int count)
{
    emit m_signalHandler->doAddSubTotalSteps(count);
}

void CObjDetectTaskWrap::emitStepProgress()
{
    emit m_signalHandler->doProgress();
}

void CObjDetectTaskWrap::emitGraphicsContextChanged()
{
    emit m_signalHandler->doGraphicsContextChanged();
}

void CObjDetectTaskWrap::emitOutputChanged()
{
    emit m_signalHandler->doOutputChanged();
}

void CObjDetectTaskWrap::emitParametersChanged()
{
    emit m_signalHandler->doParametersChanged();
}
