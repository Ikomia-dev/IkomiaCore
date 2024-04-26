#ifndef CWORKFLOWOUTPUT_H
#define CWORKFLOWOUTPUT_H

#include <string>
#include <cstdint>
#include <QJsonObject>
#include "Main/CoreDefine.hpp"
#include "Workflow/CWorkflowTaskIO.h"


class CWorkflowOutput
{
    public:

        CWorkflowOutput();
        CWorkflowOutput(const std::string& description, const std::uintptr_t&  taskId, int taskOutputIndex);

        ~CWorkflowOutput() = default;

        void                setDescription(const std::string& description);
        void                setTaskOutput(std::uintptr_t taskId, int outputIndex);

        std::string         getDescription() const;
        std::uintptr_t      getTaskId() const;
        int                 getTaskOutputIndex() const;

        QJsonObject         toJson() const;
        void                fromJson(const QJsonObject& output, const std::uintptr_t& taskId);

    private:

        std::string     m_description;
        std::uintptr_t  m_taskId;
        int             m_taskOutputIndex;

};

#endif // CWORKFLOWOUTPUT_H
