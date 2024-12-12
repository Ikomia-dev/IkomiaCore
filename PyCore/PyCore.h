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

#ifndef PYCORE_H
#define PYCORE_H

#include "PyCoreGlobal.h"

using namespace boost::python;


//Special call policy that increases the reference count of the object being constructed
template <class Base = default_call_policies>
struct incref_return_value_policy : Base
{
    static PyObject *postcall(PyObject *args, PyObject *result)
    {
        PyObject *self = PyTuple_GET_ITEM(args, 0);
        Py_INCREF(self);
        return result;
    }
};

// Generic override of Python __copy__ and __deepcopy__
template<class T>
inline PyObject * managingPyObject(T *p)
{
    return typename manage_new_object::apply<T*>::type()(p);
}

template<class Copyable>
object generic_copy(object copyable)
{
    Copyable *newCopyable(new Copyable(extract<const Copyable&>(copyable)));
    object result(detail::new_reference(managingPyObject(newCopyable)));
    extract<dict>(result.attr("__dict__"))().update(copyable.attr("__dict__"));
    return result;
}

template<class Copyable>
object generic_deepcopy(object copyable, dict memo)
{
    object copyMod = import("copy");
    object deepcopy = copyMod.attr("deepcopy");
    Copyable *newCopyable(new Copyable(extract<const Copyable&>(copyable)));
    object result(detail::new_reference(managingPyObject(newCopyable)));

    // HACK: copyableId shall be the same as the result of id(copyable) in Python -
    // please tell me that there is a better way! (and which ;-p)
    std::uintptr_t copyableId = reinterpret_cast<std::uintptr_t>(copyable.ptr());
    memo[copyableId] = result;

    extract<dict>(result.attr("__dict__"))().update(deepcopy(extract<dict>(copyable.attr("__dict__"))(), memo));
    return result;
}

#endif // PYCORE_H
