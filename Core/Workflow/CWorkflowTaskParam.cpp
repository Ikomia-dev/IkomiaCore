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

#include "CWorkflowTaskParam.h"
#include <QHash>
#include <ostream>

CWorkflowTaskParam::CWorkflowTaskParam()
{
}

void CWorkflowTaskParam::setParamMap(const UMapString &paramMap)
{
    for(auto it : paramMap)
        m_cfg[it.first] = it.second;
}

UMapString CWorkflowTaskParam::getParamMap() const
{
    return m_cfg;
}

uint CWorkflowTaskParam::getHashValue() const
{
    std::vector<QString> values;
    auto paramMap = getParamMap();

    for(auto it=paramMap.begin(); it!=paramMap.end(); ++it)
        values.push_back(QString::fromStdString(it->second));

    return qHashRange(values.begin(), values.end());
}

void CWorkflowTaskParam::merge(const UMapString& newValues)
{
    // Allow partial update
    UMapString params = getParamMap();
    for (auto it=newValues.begin(); it!=newValues.end(); ++it)
    {
        if (params.find(it->first) != params.end())
            params[it->first] = it->second;
    }
    setParamMap(params);
}

std::ostream& operator<<(std::ostream& os, const CWorkflowTaskParam& param)
{
    param.to_ostream(os);
    return os;
}

void CWorkflowTaskParam::to_ostream(std::ostream &os) const
{
    auto params = getParamMap();
    for(auto it=params.begin(); it!=params.end(); ++it)
        os << it->first << ":" << it->second << std::endl;
}
