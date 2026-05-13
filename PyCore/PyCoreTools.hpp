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

#ifndef PYCORETOOLS_HPP
#define PYCORETOOLS_HPP

#include "Main/CoreTools.hpp"
#include "PyCoreGlobal.h"

namespace Ikomia
{
    namespace Utils
    {
        //Place tool function here into specific namespace
    }
}


// Returns true if a to-Python converter for T is already in Boost.Python's registry.
// Use this guard before calling to_python_converter<T,...>() or any registerStd*<T>()
// to avoid "second conversion method ignored" warnings when multiple .pyd modules
// share the same converter set.
template<typename T>
inline bool isConverterRegistered()
{
    namespace bpc = boost::python::converter;
    const bpc::registration* reg = bpc::registry::query(boost::python::type_id<T>());
    return reg != nullptr && reg->m_to_python != nullptr;
}


// Drop-in replacement for register_ptr_to_python<P>() that skips registration if
// a to-Python converter for P is already present (e.g. registered by another .pyd).
template<typename P>
inline void safeRegisterPtrToPython()
{
    if (!isConverterRegistered<P>())
        boost::python::register_ptr_to_python<P>();
}

#endif // PYCORETOOLS_HPP
