/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the IkomiaStudio software.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CPLUGINTOOLS_H
#define CPLUGINTOOLS_H

#include "UtilsTools.hpp"
#include "CPluginProcessInterface.hpp"

namespace Ikomia
{
    namespace Utils
    {
        class DATAPROCESSSHARED_EXPORT CPluginTools
        {
            public:

                CPluginTools();

                static std::string              getTransferPath();
                static std::string              getDirectory(const std::string& name, int language);
                static std::string              getDirectory(const std::string& name);
                static std::string              getCppValidPluginFolder(const std::string &name);
                static std::string              getPythonPluginFolder(const std::string &name);
                static std::string              getReadmeDescription(const std::string &name);
                static boost::python::object    loadPythonModule(const std::string& name, bool bReload);

            private:

                static std::string              getPythonValidPluginFolder(const std::string &name);
        };
    }
}

#endif // CPLUGINTOOLS_H
