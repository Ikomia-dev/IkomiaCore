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

#ifndef CDATAVIDEOINFO_H
#define CDATAVIDEOINFO_H

#include "CDataImageInfo.h"
#include "Main/CoreGlobal.hpp"

/** @cond INTERNAL */

class CORESHARED_EXPORT CDataVideoInfo : public CDataImageInfo
{
    public:

        //Constructors
        CDataVideoInfo();
        CDataVideoInfo(IODataType type);
        CDataVideoInfo(const std::string &fileName);
        CDataVideoInfo(IODataType type, const std::string &fileName);
        CDataVideoInfo(size_t fps, size_t frameCount, size_t currentPos, int fourcc);
        CDataVideoInfo(const CDataVideoInfo& info);
        CDataVideoInfo(CDataVideoInfo&& info);

        //Destructor
        virtual ~CDataVideoInfo();

        //Operators
        CDataVideoInfo& operator=(const CDataVideoInfo& info);
        CDataVideoInfo& operator=(CDataVideoInfo&& info);

        virtual VectorPairString    getStringList() const override;

    public:

        int     m_sourceType = 0;
        size_t  m_fps = 0;
        size_t  m_frameCount = 0;
        size_t  m_currentPos = 0;
        int     m_fourcc = 0;
};

using CDataVideoInfoPtr = std::shared_ptr<CDataVideoInfo>;

/** @endcond */

#endif // CDATAVIDEOINFO_H
