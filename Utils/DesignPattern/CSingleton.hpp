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

#ifndef CSINGLETON_H
#define CSINGLETON_H

/// @cond INTERNAL

template<class Type>
class CSingleton
{
    public:

        static Type& getInstance()
        {
            // Use static function scope variable to
            // correctly define lifespan of object.
            static Type instance;
            return instance;
        }
};

template<class Type>
class CSingletonPtr
{
    public:

        static Type* getInstance()
        {
            // Use static function scope variable to
            // correctly define lifespan of object.
            static Type* pInstance = new Type;
            return pInstance;
        }
};

/// @endcond

#endif // CSINGLETON_HPP
