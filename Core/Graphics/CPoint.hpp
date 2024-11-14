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

#ifndef CPOINT_H
#define CPOINT_H

template<typename T>
class CPoint
{
    public:

        CPoint()
        {
        }
        CPoint(T x, T y)
        {
            m_x = x;
            m_y = y;
        }
        CPoint(const T& x, const T& y, const T& z)
        {
            m_x = x;
            m_y = y;
            m_z = z;
        }

        void setX(const T& x)
        {
            m_x = x;
        }

        void setY(const T& y)
        {
            m_y = y;
        }
        void setZ(const T& z)
        {
            m_z = z;
        }

        T   getX() const
        {
            return m_x;
        }

        T   getY() const
        {
            return m_y;
        }
        T   getZ() const
        {
            return m_z;
        }

    public:

        T m_x;
        T m_y;
        T m_z;
};

using CPointI = CPoint<int>;
using CPointF = CPoint<float>;
using PolygonF = std::vector<CPointF>;

#endif // CPOINT_H
