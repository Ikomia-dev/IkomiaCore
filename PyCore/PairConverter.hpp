#ifndef PAIRCONVERTER_HPP
#define PAIRCONVERTER_HPP

#include "PyCoreGlobal.h"

//----------------------//
//- std::pair converter -//
//----------------------//
template <typename T1, typename T2>
struct pair_to_python_tuple
{
    static PyObject* convert(const std::pair<T1, T2>& pair)
    {
        return incref(make_tuple(pair.first, pair.second).ptr());
    }
};

template <typename T1, typename T2>
struct pair_from_python_tuple
{
    pair_from_python_tuple()
    {
        converter::registry::push_back(&convertible, &construct, type_id<std::pair<T1, T2>>());
    }

    static void* convertible(PyObject *obj_ptr)
    {
        if (!PyTuple_CheckExact(obj_ptr))
            return nullptr;

        if (PyTuple_Size(obj_ptr) != 2)
            return nullptr;

        return obj_ptr;
    }

    static void construct(PyObject *obj_ptr, converter::rvalue_from_python_stage1_data* data)
    {
        tuple tuple(borrowed(obj_ptr));
        void* storage = ((converter::rvalue_from_python_storage<std::pair<T1, T2> >*) data)->storage.bytes;
        new (storage) std::pair<T1, T2>(extract<T1>(tuple[0]), extract<T2>(tuple[1]));
        data->convertible = storage;
    }
};

//Register std::pair<T1, T2> <-> python tuple converters
template <typename T1, typename T2>
void registerStdPair()
{
    to_python_converter<std::pair<T1, T2>, pair_to_python_tuple<T1, T2>>();
    pair_from_python_tuple<T1, T2>();
}

#endif // PAIRCONVERTER_HPP
