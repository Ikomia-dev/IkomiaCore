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

#ifndef CTASKPARAMFACTORY_HPP
#define CTASKPARAMFACTORY_HPP

#include "DesignPattern/CAbstractFactory.hpp"
#include "Workflow/CWorkflowTaskParam.h"

//------------------------//
//----- Task factory -----//
//------------------------//
/**
 * @ingroup groupDataProcess
 * @brief The CTaskFactory class is an abstract class defining the core structure of the process factory.
 * The system extensibility for the process library is based on the well known factory design pattern.
 * Each process task must implement a factory class derived from this class.
 * Then the system is able to instantiate dynamically a process object (even user-defined).
 */
class CTaskParamFactory
{
    public:

        virtual ~CTaskParamFactory() = default;

        std::string                     getName() const
        {
            return m_name;
        }

        virtual WorkflowTaskParamPtr    create() = 0;

    protected:

        std::string m_name = "";
};

using TaskParamFactoryPtr = std::shared_ptr<CTaskParamFactory>;
using TaskParamFactories = std::vector<TaskParamFactoryPtr>;


//----- Task parameters abstract factory -----//
class CTaskParamAbstractFactory: public CAbstractFactory<std::string, WorkflowTaskParamPtr>
{
    public:

        TaskParamFactories&  getList()
        {
            return m_factories;
        }

        TaskParamFactoryPtr  getFactory(const std::string& name) const
        {
            auto it = std::find_if(m_factories.begin(),
                                   m_factories.end(),
                                   [&name](const TaskParamFactoryPtr& factoryPtr){ return factoryPtr->getName() == name;});

            if (it == m_factories.end())
                return nullptr;
            else
                return *it;
        }

        void            remove(const std::string& name)
        {
            m_factories.erase(std::remove_if(m_factories.begin(),
                                             m_factories.end(),
                                             [name](const TaskParamFactoryPtr& factoryPtr){return factoryPtr->getName() == name;}),
                              m_factories.end());
        }

        void            clear() override
        {
            CAbstractFactory<std::string, WorkflowTaskParamPtr>::clear();
            m_factories.clear();
        }

    private:

        TaskParamFactories m_factories;
};

#endif // CTASKPARAMFACTORY_HPP
