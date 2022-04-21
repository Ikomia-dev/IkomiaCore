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

#ifndef CRUNTASKMANAGER_H
#define CRUNTASKMANAGER_H

#include "Workflow/CWorkflowTask.h"
#include "Data/CDataVideoInfo.h"

class CRunTaskManager
{
    public:

        CRunTaskManager();

        void    setCfg(std::map<std::string, std::string>* pCfg);

        void    run(const WorkflowTaskPtr& pTask, const std::string inputName);

        void    stop(const WorkflowTaskPtr &taskPtr);

    private:

        void    runImageProcess2D(const WorkflowTaskPtr& taskPtr);
        void    runVideoProcess(const WorkflowTaskPtr& taskPtr);

        void    manageOutputs(const WorkflowTaskPtr& taskPtr, const std::string& inputName);

        void    saveVideoOutputs(const WorkflowTaskPtr &taskPtr, const std::string &inputName);

    private:

        std::atomic_bool    m_bStop{false};
        size_t              m_outputVideoId = SIZE_MAX;
        std::map<std::string, std::string>* m_pCfg = nullptr;
        const int           m_timeout = 30000; //in milliseconds
};

#endif // CRUNTASKMANAGER_H
