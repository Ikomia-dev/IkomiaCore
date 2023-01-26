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

#include "PyDataIO.h"
#include "Data/CvMatNumpyArrayConverter.h"
#include "PyDataIODocString.hpp"
#include "COpencvImageIO.h"
#include "CDataImageIO.h"
#include "CDataVideoIO.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL IKOMIA_ARRAY_API
#include <numpy/ndarrayobject.h>

//Numpy initialization
static bool init_numpy()
{
    import_array();

    if(PyArray_API == NULL)
        return false;
    else
        return true;
}

void translateCException(const CException& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE(pydataio)
{
    // Enable user-defined docstrings and python signatures, while disabling the C++ signatures
    docstring_options local_docstring_options(true, true, false);

    // Set the docstring of the current module scope
    scope().attr("__doc__") = _moduleDocString;

    //Numpy initialization
    init_numpy();

    //Register exceptions
    register_exception_translator<CException>(&translateCException);

    //------------------------//
    //----- CDataImageIO -----//
    //------------------------//
    //Overloaded member functions
    CMat (CDataImageIO::*read_img)() = &CDataImageIO::read;

    class_<CDataImageIO, boost::noncopyable>("CDataImageIO", _dataImageIODocString, init<const std::string&>(_ctorDataImageIO))
        .def("read", read_img, _readDataImageDocString, args("self"))
        .def("write", &CDataImageIO::write, _writeDataImageDocString, args("self"))
        .def("is_image_format", &CDataImageIO::isImageFormat, _isImageFormatDocString, args("extension")).staticmethod("is_image_format")
    ;

    //------------------------//
    //----- CDataVideoIO -----//
    //------------------------//
    //Overloaded member functions
    CMat (CDataVideoIO::*read_video)() = &CDataVideoIO::read;
    void (CDataVideoIO::*write_video1)(const CMat&) = &CDataVideoIO::write;
    void (CDataVideoIO::*write_video2)(const CMat&, const std::string&) = &CDataVideoIO::write;

    class_<CDataVideoIO, boost::noncopyable>("CDataVideoIO", _dataVideoIODocString, init<const std::string&>(_ctorDataVideoIO))
        .def("read", read_video, _readDataVideoDocString, args("self"))
        .def("write", write_video1, _writeDataVideo1DocString, args("self", "frame_data"))
        .def("write", write_video2, _writeDataVideo2DocString, args("self", "frame_data", "path"))
        .def("stop_read", &CDataVideoIO::stopRead, _stopReadDocString, args("self"))
        .def("stop_write", &CDataVideoIO::stopWrite, _stopWriteDocString, args("self", "timeout"))
        .def("wait_write_finished", &CDataVideoIO::waitWriteFinished, _waitWriteFinishedDocString, args("self", "timeout"))
        .def("is_video_format", &CDataVideoIO::isVideoFormat, _isVideoFormatDocString, args("extension", "video_only")).staticmethod("is_video_format")
    ;
}
