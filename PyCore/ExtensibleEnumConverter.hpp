#ifndef EXTENSIBLEENUMCONVERTER_HPP
#define EXTENSIBLEENUMCONVERTER_HPP

#include "PyCoreGlobal.h"
#include "CExtensibleEnum.hpp"

// From-Python rvalue converter for CExtensibleEnum<E>.
//
// Boost.Python's implicitly_convertible<E, CExtensibleEnum<E>>() only fires
// when the incoming Python object is recognised as the exact Boost.Python
// enum_<E> wrapper type.  Any int-like Python object (plain int, Python
// IntEnum subclass, or Boost.Python enum value in a context where the C++
// enum type is not found first) falls through that check and raises:
//
//   "No registered converter was able to produce a C++ rvalue of type
//    CExtensibleEnum<E> from this Python object of type <E>"
//
// This converter accepts *any* Python integer-like object (PyLong), extracts
// its integer value, and constructs CExtensibleEnum<E>(int) directly.  It
// covers all practical call sites:
//   - task_io = CWorkflowTaskIO(IODataType.IMAGE)      # Boost.Python enum
//   - task_io = CWorkflowTaskIO(42)                    # extended / raw int
//   - task_io = CWorkflowTaskIO(MyIntEnum.IMAGE)       # Python IntEnum

template<typename E>
struct extensible_enum_from_python
{
    static void* convertible(PyObject* obj)
    {
        return PyLong_Check(obj) ? obj : nullptr;
    }

    static void construct(PyObject* obj, converter::rvalue_from_python_stage1_data* data)
    {
        const int value = static_cast<int>(PyLong_AsLong(obj));
        void* storage = reinterpret_cast<converter::rvalue_from_python_storage<CExtensibleEnum<E>>*>(data)->storage.bytes;
        new (storage) CExtensibleEnum<E>(value);
        data->convertible = storage;
    }

    extensible_enum_from_python()
    {
        converter::registry::push_back(
            &convertible,
            &construct,
            type_id<CExtensibleEnum<E>>()
        );
    }
};

#endif // EXTENSIBLEENUMCONVERTER_HPP
