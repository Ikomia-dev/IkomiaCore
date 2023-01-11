#ifndef MAPCONVERTER_HPP
#define MAPCONVERTER_HPP

#include "PyCoreGlobal.h"

//----------------------//
//- std::map converter -//
//----------------------//
template <typename T, typename U>
struct map_to_python_dict
{
    static PyObject* convert(const std::map<T, U>& map)
    {
        dict pydict;
        for (auto it=map.begin(); it!=map.end(); it++)
        {
            pydict[it->first] = it->second;
        }
        return incref(pydict.ptr());
    }
};

template <typename T, typename U>
struct map_from_python_dict
{
    map_from_python_dict()
    {
        converter::registry::push_back(&convertible, &construct, type_id<std::map<T, U>>());
    }

    static void* convertible(PyObject *obj_ptr)
    {
        if(!PyDict_Check(obj_ptr))
            return nullptr;

        return obj_ptr;
    }

    static void construct(PyObject *obj_ptr, converter::rvalue_from_python_stage1_data* data)
    {
        std::map<T, U> map;
        PyObject *pKey, *pValue;
        Py_ssize_t pos = 0;

        while (PyDict_Next(obj_ptr, &pos, &pKey, &pValue))
        {
            T key = typename extract<T>::extract(pKey);
            U val = typename extract<U>::extract(pValue);
            map.insert(std::make_pair(key, val));
        }

        void* storage = ((converter::rvalue_from_python_storage<std::unordered_map<T, U>>*)data)->storage.bytes;
        new (storage) std::map<T, U>(map);
        data->convertible = storage;
    }
};

//Register std::vector<T> <-> python list converters
template <typename T, typename U>
void registerStdMap()
{
    to_python_converter<std::map<T, U>, map_to_python_dict<T, U>>();
    map_from_python_dict<T, U>();
}

//--------------------------------//
//- std::unordered_map converter -//
//--------------------------------//
template <typename T, typename U>
struct umap_to_python_dict
{
    static PyObject* convert(const std::unordered_map<T, U>& map)
    {
        dict pydict;
        for (auto it=map.begin(); it!=map.end(); it++)
        {
            pydict[it->first] = it->second;
        }
        return incref(pydict.ptr());
    }
};

template <typename T, typename U>
struct umap_from_python_dict
{
    umap_from_python_dict()
    {
        converter::registry::push_back(&convertible, &construct, type_id<std::unordered_map<T, U>>());
    }

    static void* convertible(PyObject *obj_ptr)
    {
        if(!PyDict_Check(obj_ptr))
            return nullptr;

        return obj_ptr;
    }

    static void construct(PyObject *obj_ptr, converter::rvalue_from_python_stage1_data* data)
    {
        std::unordered_map<T, U> umap;
        PyObject *pKey, *pValue;
        Py_ssize_t pos = 0;

        while (PyDict_Next(obj_ptr, &pos, &pKey, &pValue))
        {
            T key = typename extract<T>::extract(pKey);
            U val = typename extract<U>::extract(pValue);
            umap.insert(std::make_pair(key, val));
        }

        void* storage = ((converter::rvalue_from_python_storage<std::unordered_map<T, U>>*)data)->storage.bytes;
        new (storage) std::unordered_map<T, U>(umap);
        data->convertible = storage;
    }
};

//Register std::vector<T> <-> python list converters
template<typename T, typename U>
void registerStdUMap()
{
    to_python_converter<std::unordered_map<T, U>, umap_to_python_dict<T, U>>();
    umap_from_python_dict<T, U>();
}

#endif // MAPCONVERTER_HPP
