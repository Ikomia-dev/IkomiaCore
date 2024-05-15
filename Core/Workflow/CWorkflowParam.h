#ifndef CWORKFLOWPARAM_H
#define CWORKFLOWPARAM_H

#include <string>
#include <cstdint>
#include <QJsonObject>


class CWorkflowParam
{
    public:

        CWorkflowParam();
        CWorkflowParam(const std::string& name, const std::string& description);
        CWorkflowParam(const std::string& name, const std::string& description, const std::uintptr_t& taskId, const std::string& taskParamName);

        ~CWorkflowParam() = default;

        void            setName(const std::string& name);
        void            setDescription(const std::string& description);
        void            setTaskParam(std::uintptr_t taskId, const std::string& paramName);

        std::string     getName() const;
        std::string     getDescription() const;
        std::string     getTaskParamName() const;
        std::uintptr_t  getTaskId() const;

        QJsonObject     toJson() const;
        void            fromJson(const QJsonObject& param, std::uintptr_t taskId);

    private:

        std::string     m_name;
        std::string     m_description;
        std::string     m_taskParamName;
        std::uintptr_t  m_taskId;
};

#endif // CWORKFLOWPARAM_H
