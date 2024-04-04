/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the Ikomia API libraries.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
#ifndef CWORKFLOWWRAP_H
#define CWORKFLOWWRAP_H

#include "PyDataProcessGlobal.h"
#include "Core/CWorkflow.h"
#include "Core/CIkomiaRegistry.h"


class CWorkflowWrap: public CWorkflow, public wrapper<CWorkflow>
{
    public:

        CWorkflowWrap();
        CWorkflowWrap(const std::string& name);
        CWorkflowWrap(const std::string &name, const std::shared_ptr<CIkomiaRegistry>& registryPtr);
        CWorkflowWrap(const CWorkflow &workflow);

        std::string                 repr() const override;
        std::string                 default_repr() const;

        std::uintptr_t              getRootID();
        std::vector<std::uintptr_t> getTaskIDs();
        std::uintptr_t              getLastTaskID();
        WorkflowTaskPtr             getTask(std::uintptr_t id);
        double                      getElapsedTimeTo(std::uintptr_t id);
        std::vector<std::uintptr_t> getParents(std::uintptr_t id);
        std::vector<std::uintptr_t> getChildren(std::uintptr_t id);
        std::vector<size_t>         getInEdges(std::uintptr_t id);
        std::vector<size_t>         getOutEdges(std::uintptr_t id);
        boost::python::tuple        getEdgeInfo(size_t id);
        std::uintptr_t              getEdgeSource(size_t id);
        std::uintptr_t              getEdgeTarget(size_t id);

        std::uintptr_t              addTaskWrap(const WorkflowTaskPtr &taskPtr);
        void                        addParameter(const std::string& name, const std::string& description, const std::uintptr_t& taskId, const std::string& targetParamName);

        void                        connectWrap(const std::uintptr_t& src, const std::uintptr_t& target, int srcIndex, int targetIndex);

        void                        deleteTaskWrap(std::uintptr_t id);
        void                        deleteEdgeWrap(size_t id);

        void                        run() override;
        void                        default_run();

        void                        clearWrap();

        void                        loadWrap(const std::string& path);

    private:

        std::pair<bool, WorkflowEdge>   getEdgeDescriptor(size_t index) const;

        void                            removeEdgeIndex(const WorkflowEdge &edge);

    private:

        size_t                          m_edgeIndex = 0;
        std::map<WorkflowEdge, size_t>  m_edgeDescToIndex;
};

#endif // CWORKFLOWWRAP_H
