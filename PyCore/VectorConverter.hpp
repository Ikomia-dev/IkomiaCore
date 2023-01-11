#ifndef VECTORCONVERTER_HPP
#define VECTORCONVERTER_HPP

#include "PyCoreGlobal.h"

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

#endif // VECTORCONVERTER_HPP
