#include "CWorkflowOutput.h"

CWorkflowOutput::CWorkflowOutput()
{

}

CWorkflowOutput::CWorkflowOutput(const std::string &description, const std::uintptr_t &taskId, int taskOutputIndex)
{
    m_description = description;
    m_taskId = taskId;
    m_taskOutputIndex = taskOutputIndex;
}

void CWorkflowOutput::setDescription(const std::string &description)
{
    m_description = description;
}

void CWorkflowOutput::setTaskOutput(std::uintptr_t taskId, int outputIndex)
{
    m_taskId = taskId;
    m_taskOutputIndex = outputIndex;
}

std::string CWorkflowOutput::getDescription() const
{
    return m_description;
}

std::uintptr_t CWorkflowOutput::getTaskId() const
{
    return m_taskId;
}

int CWorkflowOutput::getTaskOutputIndex() const
{
    return m_taskOutputIndex;
}

QJsonObject CWorkflowOutput::toJson() const
{
    QJsonObject jsonOutput;
    jsonOutput["description"] = QString::fromStdString(m_description);
    jsonOutput["task_id"] = QString::fromStdString(std::to_string(m_taskId));
    jsonOutput["task_output_index"] = m_taskOutputIndex;
    return jsonOutput;
}

void CWorkflowOutput::fromJson(const QJsonObject &output, const uintptr_t &taskId)
{
    m_taskId = taskId;
    m_description = output["description"].toString().toStdString();
    m_taskOutputIndex = output["task_output_index"].toInt();
}
