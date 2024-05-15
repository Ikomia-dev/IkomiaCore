#include "CWorkflowParam.h"

CWorkflowParam::CWorkflowParam()
{
}

CWorkflowParam::CWorkflowParam(const std::string &name, const std::string &description)
{
    m_name = name;
    m_description = description;
}

CWorkflowParam::CWorkflowParam(const std::string &name, const std::string &description, const std::uintptr_t &taskId, const std::string &taskParamName)
{
    m_name = name;
    m_description = description;
    m_taskId = taskId;
    m_taskParamName = taskParamName;
}

void CWorkflowParam::setName(const std::string &name)
{
    m_name = name;
}

void CWorkflowParam::setDescription(const std::string &description)
{
    m_description = description;
}

void CWorkflowParam::setTaskParam(std::uintptr_t taskId, const std::string &paramName)
{
    m_taskId = taskId;
    m_taskParamName = paramName;
}

std::string CWorkflowParam::getName() const
{
    return m_name;
}

std::string CWorkflowParam::getDescription() const
{
    return m_description;
}

std::string CWorkflowParam::getTaskParamName() const
{
    return m_taskParamName;
}

std::uintptr_t CWorkflowParam::getTaskId() const
{
    return m_taskId;
}

QJsonObject CWorkflowParam::toJson() const
{
    QJsonObject jsonParam;
    jsonParam["name"] = QString::fromStdString(m_name);
    jsonParam["description"] = QString::fromStdString(m_description);
    jsonParam["task_id"] = QString::fromStdString(std::to_string(m_taskId));
    jsonParam["task_param_name"] = QString::fromStdString(m_taskParamName);
    return jsonParam;
}

void CWorkflowParam::fromJson(const QJsonObject &param, uintptr_t taskId)
{
    m_taskId = taskId;
    m_name = param["name"].toString().toStdString();
    m_description = param["description"].toString().toStdString();
    m_taskParamName = param["task_param_name"].toString().toStdString();
}
