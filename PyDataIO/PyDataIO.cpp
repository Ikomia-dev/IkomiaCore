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

BOOST_PYTHON_MODULE(pydataio)
{
    //Numpy initialization
    init_numpy();

    //CMat <-> Numpy NdArray converters
    to_python_converter<CMat, BoostCvMatToNumpyArrayConverter>();
    BoostNumpyArrayToCvMatConverter();

    //------------------------//
    //----- CDataImageIO -----//
    //------------------------//
    //Overloaded member functions
    CMat (CDataImageIO::*read_img)() = &CDataImageIO::read;

    class_<CDataImageIO, boost::noncopyable>("CDataImageIO", _dataImageIODocString, init<const std::string&>(_ctorDataImageIO))
        .def("read", read_img, _readDataImageDocString)
        .def("write", &CDataImageIO::write, _writeDataImageDocString)
        .def("isImageFormat", &CDataImageIO::isImageFormat, _isImageFormatDocString).staticmethod("isImageFormat")
    ;

    //------------------------//
    //----- CDataVideoIO -----//
    //------------------------//
    //Overloaded member functions
    CMat (CDataVideoIO::*read_video)() = &CDataVideoIO::read;
    void (CDataVideoIO::*write_video1)(const CMat&) = &CDataVideoIO::write;
    void (CDataVideoIO::*write_video2)(const CMat&, const std::string&) = &CDataVideoIO::write;

    class_<CDataVideoIO, boost::noncopyable>("CDataImageIO", _dataVideoIODocString, init<const std::string&>(_ctorDataVideoIO))
        .def("read", read_video, _readDataVideoDocString)
        .def("write", write_video1, _writeDataVideo1DocString)
        .def("write", write_video2, _writeDataVideo2DocString)
        .def("stopRead", &CDataVideoIO::stopRead, _stopReadDocString)
        .def("stopWrite", &CDataVideoIO::stopWrite, _stopWriteDocString)
        .def("waitWriteFinished", &CDataVideoIO::waitWriteFinished, _waitWriteFinishedDocString)
        .def("isVideoFormat", &CDataVideoIO::isVideoFormat, _isVideoFormatDocString).staticmethod("isVideoFormat")
    ;

    //--------------------------//
    //----- COpencvImageIO -----//
    //--------------------------//
    //Overloaded member functions
    CMat (COpencvImageIO::*read_ocv1)() = &COpencvImageIO::read;

    class_<COpencvImageIO>("COpencvImageIO", init<const std::string&>())
        .def("read", read_ocv1)
    ;
}
