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

#ifndef CWORKFLOWTASKIOWRAP_H
#define CWORKFLOWTASKIOWRAP_H

#include "PyCoreGlobal.h"
#include "Workflow/CWorkflowTaskIO.h"

class CWorkflowTaskIOWrap : public CWorkflowTaskIO, public wrapper<CWorkflowTaskIO>
{
    public:

        CWorkflowTaskIOWrap();
        CWorkflowTaskIOWrap(IODataType dataType);
        CWorkflowTaskIOWrap(IODataType dataType, const std::string& name);
        CWorkflowTaskIOWrap(const CWorkflowTaskIO& io);

        size_t      getUnitElementCount() const override;
        size_t      default_getUnitElementCount() const;

        bool        isDataAvailable() const override;
        bool        default_isDataAvailable() const;

        bool        isAutoInput() const override;
        bool        default_isAutoInput() const;

        void        clearData() override;
        void        default_clearData();

        void        copyStaticData(const std::shared_ptr<CWorkflowTaskIO>& ioPtr) override;
        void        default_copyStaticData(const std::shared_ptr<CWorkflowTaskIO>& ioPtr);

        void        load(const std::string& path) override;
        void        default_load(const std::string& path);

        void        save(const std::string& path) override;
        void        default_save(const std::string& path);

        std::string toJson() const override;
        std::string default_toJsonNoOpt() const;

        std::string toJson(const std::vector<std::string>& options) const override;
        std::string default_toJson(const std::vector<std::string>& options) const;

        void        fromJson(const std::string& jsonStr) override;
        void        default_fromJson(const std::string& jsonStr);
};

#endif // CWORKFLOWTASKIOWRAP_H
