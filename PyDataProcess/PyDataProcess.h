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

#ifndef PYDATAPROCESS_H
#define PYDATAPROCESS_H

#include "PyDataProcessGlobal.h"

using namespace boost::python;

//Special call policy that increases the reference count of the object being constructed
//Needed for our CWorkflowTask or CWorkflowTaskWidget based objects that can be extended in Python
//Solve the issue of reference count for std::shared_ptr return from extended python class
//and lead to segmentation fault caused by double delete
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

//Template class for bi-directionnal conversion:
//std::vector<T> <-> python list
template <typename T>
struct vector_to_python_list
{
    static PyObject* convert(const std::vector<T>& vec)
    {
        list pylist;
        for(size_t i=0; i<vec.size(); i++)
            pylist.append(vec[i]);

        return incref(pylist.ptr());
    }
};

template <typename T>
struct vector_from_python_list
{
    static void* convertible(PyObject *obj_ptr)
    {
        if(!PyList_Check(obj_ptr))
            return 0;

        return obj_ptr;
    }

    static void construct(PyObject *obj_ptr, converter::rvalue_from_python_stage1_data* data)
    {
        std::vector<T> vec;
        for(Py_ssize_t i=0; i<PyList_Size(obj_ptr); i++)
        {
            PyObject *pyvalue = PyList_GetItem(obj_ptr, i);
            T value = typename extract<T>::extract(pyvalue);
            vec.push_back(value);
        }

        void* storage = ((converter::rvalue_from_python_storage<std::vector<T> >*)data)->storage.bytes;
        new (storage) std::vector<T>(vec);
        data->convertible = storage;
    }

    vector_from_python_list()
    {
        converter::registry::push_back(&convertible, &construct, type_id<std::vector<T> >());
    }
};

//Register std::vector<T> <-> python list converters
template<typename T>
void registerStdVector()
{
    to_python_converter<std::vector<T>, vector_to_python_list<T>>();
    vector_from_python_list<T>();
}

// Generic override of Python __copy__ and __deepcopy__
template<class T>
inline PyObject * managingPyObject(T *p)
{
    return typename manage_new_object::apply<T *>::type()(p);
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
    int copyableId = (int)((long long)copyable.ptr());
    memo[copyableId] = result;

    extract<dict>(result.attr("__dict__"))().update(deepcopy(extract<dict>(copyable.attr("__dict__"))(), memo));
    return result;
}

#endif // PYDATAPROCESS_H
