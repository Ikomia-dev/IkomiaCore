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

#include "PyDataProcess.h"
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "Data/CvMatNumpyArrayConverter.h"
#include "PyDataProcessDocString.hpp"
#include "Task/CTaskFactoryWrap.h"
#include "Task/C2dImageTaskWrap.h"
#include "Task/C2dImageInteractiveTaskWrap.h"
#include "Task/CVideoTaskWrap.h"
#include "Task/CVideoOFTaskWrap.h"
#include "Task/CVideoTrackingTaskWrap.h"
#include "Task/CDnnTrainTaskWrap.h"
#include "CWidgetFactoryWrap.h"
#include "CPluginProcessInterfaceWrap.h"
#include "IO/CNumericIOWrap.hpp"
#include "IO/CGraphicsInputWrap.h"
#include "IO/CImageIOWrap.h"
#include "IO/CVideoIOWrap.h"
#include "IO/CWidgetOutputWrap.h"
#include "IO/CDatasetIOWrap.h"
#include "IO/CPathIOWrap.h"
#include "IO/CArrayIOWrap.h"
#include "IO/CObjectDetectionIOWrap.h"
#include "IO/CInstanceSegIOWrap.h"
#include "IO/CSemanticSegIOWrap.h"
#include "CIkomiaRegistryWrap.h"
#include "CWorkflowWrap.h"

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

template<typename Type>
void exposeNumericIO(const std::string& className)
{
    //Overload member functions
    void (CNumericIO<Type>::*addValueList1)(const std::vector<Type>&) = &CNumericIO<Type>::addValueList;
    void (CNumericIO<Type>::*addValueList2)(const std::vector<Type>&, const std::string&) = &CNumericIO<Type>::addValueList;
    void (CNumericIO<Type>::*addValueList3)(const std::vector<Type>&, const std::vector<std::string>&) = &CNumericIO<Type>::addValueList;
    void (CNumericIO<Type>::*addValueList4)(const std::vector<Type>&, const std::string&, const std::vector<std::string>&) = &CNumericIO<Type>::addValueList;
    void (CNumericIO<Type>::*saveNumeric)(const std::string&) = &CNumericIO<Type>::save;

    class_<CNumericIOWrap<Type>, bases<CWorkflowTaskIO>, std::shared_ptr<CNumericIOWrap<Type>>>(className.c_str(), _featureProcessIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorFeatureIODocString, args("self", "name")))
        .def(init<const CNumericIO<Type>&>("Copy constructor"))
        .def("setOutputType", &CNumericIO<Type>::setOutputType, _setOutputTypeDocString, args("self", "type"))
        .def("setPlotType", &CNumericIO<Type>::setPlotType, _setPlotTypeDocString, args("self", "type"))
        .def("addValueList", addValueList1, _addValueList1DocString, args("self", "values"))
        .def("addValueList", addValueList2, _addValueList2DocString, args("self", "values", "header_label"))
        .def("addValueList", addValueList3, _addValueList3DocString, args("self", "values", "labels"))
        .def("addValueList", addValueList4, _addValueList4DocString, args("self", "values", "header_label", "labels"))
        .def("getOutputType", &CNumericIO<Type>::getOutputType, _getOutputTypeDocString, args("self"))
        .def("getPlotType", &CNumericIO<Type>::getPlotType, _getPlotTypeDocString, args("self"))
        .def("getValueList", &CNumericIO<Type>::getValueList, _getValueListDocString, args("self", "index"))
        .def("getAllValueList", &CNumericIO<Type>::getAllValues, _getAllValueListDocString, args("self"))
        .def("getAllLabelList", &CNumericIO<Type>::getAllValueLabels, _getAllLabelListDocString, args("self"))
        .def("getAllHeaderLabels", &CNumericIO<Type>::getAllHeaderLabels, _getAllHeaderLabelsDocString, args("self"))
        .def("getUnitElementCount", &CNumericIO<Type>::getUnitElementCount, &CNumericIOWrap<Type>::default_getUnitElementCount, _getUnitEltCountDerivedDocString, args("self"))
        .def("isDataAvailable", &CNumericIO<Type>::isDataAvailable, &CNumericIOWrap<Type>::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clearData", &CNumericIO<Type>::clearData, &CNumericIOWrap<Type>::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("copyStaticData", &CNumericIO<Type>::copyStaticData, &CNumericIOWrap<Type>::default_copyStaticData, _copyStaticDataDerivedDocString, args("self", "io"))
        .def("load", &CNumericIO<Type>::load, _numericIOLoadDocString, args("self", "path"))
        .def("save", saveNumeric, _numericIOSaveDocString, args("self", "path"))
        .def("toJson", &CNumericIO<Type>::toJson, _blobIOToJsonDocString, args("self", "options"))
        .def("fromJson", &CNumericIO<Type>::fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;
}


BOOST_PYTHON_MODULE(pydataprocess)
{
    // Enable user-defined docstrings and python signatures, while disabling the C++ signatures
    docstring_options local_docstring_options(true, true, false);

    // Set the docstring of the current module scope
    scope().attr("__doc__") = _moduleDocString;

    //Numpy initialization
    init_numpy();

    //Register smart pointers
    register_ptr_to_python<std::shared_ptr<CTaskFactory>>();
    register_ptr_to_python<std::shared_ptr<CWidgetFactory>>();
    register_ptr_to_python<std::shared_ptr<CGraphicsInput>>();
    register_ptr_to_python<std::shared_ptr<CImageIO>>();
    register_ptr_to_python<std::shared_ptr<CVideoIO>>();
    register_ptr_to_python<std::shared_ptr<CWidgetOutput>>();
    register_ptr_to_python<std::shared_ptr<CPathIO>>();
    register_ptr_to_python<std::shared_ptr<CDatasetIO>>();
    register_ptr_to_python<std::shared_ptr<CArrayIO>>();
    register_ptr_to_python<std::shared_ptr<C2dImageTask>>();
    register_ptr_to_python<std::shared_ptr<C2dImageInteractiveTask>>();
    register_ptr_to_python<std::shared_ptr<CVideoTask>>();
    register_ptr_to_python<std::shared_ptr<CVideoOFTask>>();
    register_ptr_to_python<std::shared_ptr<CVideoTrackingTask>>();
    register_ptr_to_python<std::shared_ptr<CDnnTrainTask>>();

    //Register std::vector<T> <-> python list converters
    registerStdVector<uchar>();
    registerStdVector<std::vector<uchar>>();
    registerStdVector<std::vector<std::string>>();
    registerStdVector<std::vector<double>>();
    registerStdVector<CObjectMeasure>();
    registerStdVector<std::vector<CObjectMeasure>>();
    registerStdVector<CObjectDetection>();
    registerStdVector<CInstanceSegmentation>();

    //---------------------//
    //----- CTaskInfo -----//
    //---------------------//
    class_<CTaskInfo>("CTaskInfo", _processInfoDocString, init<>("Default constructor"))
        .add_property("name", &CTaskInfo::getName, &CTaskInfo::setName, "Name of the plugin (mandatory - must be unique)")
        .add_property("path", &CTaskInfo::getPath, &CTaskInfo::setPath, "Path in the library tree view of Ikomia")
        .add_property("shortDescription", &CTaskInfo::getShortDescription, &CTaskInfo::setShortDescription, "Short description of the plugin (mandatory)")
        .add_property("description", &CTaskInfo::getDescription, &CTaskInfo::setDescription, "Full description of the plugin (mandatory)")
        .add_property("documentationLink", &CTaskInfo::getDocumentationLink, &CTaskInfo::setDocumentationLink, "Address (URL) of online documentation")
        .add_property("iconPath", &CTaskInfo::getIconPath, &CTaskInfo::setIconPath, "Relative path to the plugin icon")
        .add_property("keywords", &CTaskInfo::getKeywords, &CTaskInfo::setKeywords, "Keywords associated with the plugin (Used for Ikomia search engine)")
        .add_property("authors", &CTaskInfo::getAuthors, &CTaskInfo::setAuthors, "Authors of the plugin and/or corresponding paper (mandatory)")
        .add_property("article", &CTaskInfo::getArticle, &CTaskInfo::setArticle, "Title of the corresponding paper")
        .add_property("journal", &CTaskInfo::getJournal, &CTaskInfo::setJournal, "Paper journal")
        .add_property("version", &CTaskInfo::getVersion, &CTaskInfo::setVersion, "Plugin version (mandatory)")
        .add_property("ikomiaVersion", &CTaskInfo::getIkomiaVersion, "Ikomia API version")
        .add_property("year", &CTaskInfo::getYear, &CTaskInfo::setYear, "Year of paper publication")
        .add_property("language", &CTaskInfo::getLanguage, &CTaskInfo::setLanguage, "Python")
        .add_property("license", &CTaskInfo::getLicense, &CTaskInfo::setLicense, "License of the plugin")
        .add_property("repository", &CTaskInfo::getRepository, &CTaskInfo::setRepository, "Address of code repository (GitHub, GitLab, BitBucket...)")
        .add_property("createdDate", &CTaskInfo::getCreatedDate, &CTaskInfo::setCreatedDate, "Date of first publication")
        .add_property("modifiedDate", &CTaskInfo::getModifiedDate, &CTaskInfo::getModifiedDate, "Date of last update")
        .add_property("os", &CTaskInfo::getOS, &CTaskInfo::setOS, "Operating system")
        .add_property("internal", &CTaskInfo::isInternal, &CTaskInfo::setInternal, "Indicate a built-in algorithm.")
        .def(self_ns::str(self_ns::self))
    ;

    //------------------------//
    //----- CTaskFactory -----//
    //------------------------//
    //Overload member functions
    WorkflowTaskPtr (CTaskFactory::*create_void)() = &CTaskFactory::create;
    WorkflowTaskPtr (CTaskFactory::*create_param)(const WorkflowTaskParamPtr&) = &CTaskFactory::create;
    CTaskInfo& (CTaskFactory::*getInfoByRef)() = &CTaskFactory::getInfo;

    class_<CTaskFactoryWrap, std::shared_ptr<CTaskFactoryWrap>, boost::noncopyable>("CTaskFactory", _processFactoryDocString)
        .add_property("info", make_function(getInfoByRef, return_internal_reference<>()), &CTaskFactory::setInfo, _processFactoryInfoDocString)
        .def("create", pure_virtual(create_void), _create1DocString, args("self"))
        .def("create", pure_virtual(create_param), _create2DocString, args("self", "param"))
    ;

    //--------------------------//
    //----- CWidgetFactory -----//
    //--------------------------//
    class_<CWidgetFactoryWrap, std::shared_ptr<CWidgetFactoryWrap>, boost::noncopyable>("CWidgetFactory", _widgetFactoryDocString)
        .add_property("name", &CWidgetFactory::getName, &CWidgetFactory::setName, _widgetFactoryNameDocString)
        .def("create", pure_virtual(&CWidgetFactory::create), _createWidgetDocString, args("self", "param"))
    ;

    //-----------------------------------//
    //----- CPluginProcessInterface -----//
    //-----------------------------------//
    class_<CPluginProcessInterfaceWrap, boost::noncopyable>("CPluginProcessInterface", _pluginInterfaceDocString)
        .def("getProcessFactory", pure_virtual(&CPluginProcessInterface::getProcessFactory), _getProcessFactoryDocString, args("self"))
        .def("getWidgetFactory", pure_virtual(&CPluginProcessInterface::getWidgetFactory), _getWidgetFactoryDocString, args("self"))
    ;

    //--------------------------//
    //----- CObjectMeasure -----//
    //--------------------------//
    class_<CObjectMeasure>("CObjectMeasure", _objectMeasureDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CMeasure&, double, size_t, const std::string&>(_ctor1ObjMeasureDocString, args("self", "measure", "value", "graphicsId", "label")))
        .def(init<const CMeasure&, const std::vector<double>&, size_t, const std::string&>(_ctor2ObjMeasureDocString))
        .def("getMeasureInfo", &CObjectMeasure::getMeasureInfo, _getMeasureInfoDocString, args("self"))
        .add_property("values", &CObjectMeasure::getValues, &CObjectMeasure::setValues, "Values of the measure")
        .def_readwrite("graphicsId", &CObjectMeasure::m_graphicsId, "Identifier of the associated graphics item")
        .def_readwrite("label", &CObjectMeasure::m_label, "Label of the measure")
    ;

    //--------------------------//
    //----- CBlobMeasureIO -----//
    //--------------------------//
    void (CBlobMeasureIO::*saveBlob)(const std::string&) = &CBlobMeasureIO::save;

    class_<CBlobMeasureIO, bases<CWorkflowTaskIO>, std::shared_ptr<CBlobMeasureIO>>("CBlobMeasureIO", _measureIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorMeasureIODocString, args("self", "name")))
        .def(init<const CBlobMeasureIO&>("Copy constructor"))
        .def("setObjectMeasure", &CBlobMeasureIO::setObjectMeasure, _setObjMeasureDocString, args("self", "index", "measure"))
        .def("getMeasures", &CBlobMeasureIO::getMeasures, _getMeasuresDocString, args("self"))
        .def("isDataAvailable", &CBlobMeasureIO::isDataAvailable, _isMeasureDataAvailableDocString, args("self"))
        .def("addObjectMeasure", &CBlobMeasureIO::addObjectMeasure, _addObjMeasureDocString, args("self", "measure"))
        .def("addObjectMeasures", &CBlobMeasureIO::addObjectMeasures, _addObjMeasuresDocString, args("self", "measures"))
        .def("clearData", &CBlobMeasureIO::clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CBlobMeasureIO::load, _blobMeasureIOLoadDocString, args("self", "path"))
        .def("save", saveBlob, _blobMeasureIOSaveDocString, args("self", "path"))
        .def("toJson", &CBlobMeasureIO::toJson, _blobIOToJsonDocString, args("self", "options"))
        .def("fromJson", &CBlobMeasureIO::fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //--------------------------//
    //----- CGraphicsInput -----//
    //--------------------------//
    void (CGraphicsInput::*saveGraphicsIn)(const std::string&) = &CGraphicsInput::save;
    std::string (CGraphicsInput::*toJsonIn)(const std::vector<std::string>&) const = &CGraphicsInput::toJson;
    void (CGraphicsInput::*fromJsonIn)(const std::string&) = &CGraphicsInput::fromJson;

    class_<CGraphicsInputWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CGraphicsInputWrap>>("CGraphicsInput", _graphicsInputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorGraphicsInDocString, args("self", "name")))
        .def(init<const CGraphicsInput&>("Copy constructor"))
        .def("setItems", &CGraphicsInput::setItems, _setItemsDocString, args("self", "items"))
        .def("getItems", &CGraphicsInput::getItems, _getItemsDocString, args("self"))
        .def("isDataAvailable", &CGraphicsInput::isDataAvailable, &CGraphicsInputWrap::default_isDataAvailable, _isGraphicsDataAvailableDocString, args("self"))
        .def("clearData", &CGraphicsInput::clearData, &CGraphicsInputWrap::default_clearData, _clearGraphicsDataDocString, args("self"))
        .def("load", &CGraphicsInput::load, _graphicsInputLoadDocString, args("self", "path"))
        .def("save", saveGraphicsIn, _graphicsInputSaveDocString, args("self", "path"))
        .def("toJson", toJsonIn, _blobIOToJsonDocString, args("self", "options"))
        .def("fromJson", fromJsonIn, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //---------------------------//
    //----- CGraphicsOutput -----//
    //---------------------------//
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPoint1)(const CPointF&) = &CGraphicsOutput::addPoint;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPoint2)(const CPointF&, const CGraphicsPointProperty&) = &CGraphicsOutput::addPoint;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addRectangle1)(float, float, float, float) = &CGraphicsOutput::addRectangle;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addRectangle2)(float, float, float, float, const CGraphicsRectProperty&) = &CGraphicsOutput::addRectangle;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addEllipse1)(float, float, float, float) = &CGraphicsOutput::addEllipse;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addEllipse2)(float, float, float, float, const CGraphicsEllipseProperty&) = &CGraphicsOutput::addEllipse;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPolygon1)(const std::vector<CPointF>&) = &CGraphicsOutput::addPolygon;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPolygon2)(const std::vector<CPointF>&, const CGraphicsPolygonProperty&) = &CGraphicsOutput::addPolygon;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPolyline1)(const std::vector<CPointF>&) = &CGraphicsOutput::addPolyline;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addPolyline2)(const std::vector<CPointF>&, const CGraphicsPolylineProperty&) = &CGraphicsOutput::addPolyline;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addComplexPolygon1)(const PolygonF&, const std::vector<PolygonF>&) = &CGraphicsOutput::addComplexPolygon;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addComplexPolygon2)(const PolygonF&, const std::vector<PolygonF>&, const CGraphicsPolygonProperty&) = &CGraphicsOutput::addComplexPolygon;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addText1)(const std::string&, float x, float y) = &CGraphicsOutput::addText;
    ProxyGraphicsItemPtr (CGraphicsOutput::*addText2)(const std::string&, float x, float y, const CGraphicsTextProperty&) = &CGraphicsOutput::addText;
    void (CGraphicsOutput::*saveGraphicsOut)(const std::string&) = &CGraphicsOutput::save;
    std::string (CGraphicsOutput::*toJsonOut)(const std::vector<std::string>&) const = &CGraphicsOutput::toJson;
    void (CGraphicsOutput::*fromJsonOut)(const std::string&) = &CGraphicsOutput::fromJson;

    class_<CGraphicsOutput, bases<CWorkflowTaskIO>, std::shared_ptr<CGraphicsOutput>>("CGraphicsOutput", _graphicsOutputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorGraphicsOutDocString, args("self", "name")))
        .def("setNewLayer", &CGraphicsOutput::setNewLayer, _setNewLayerDocString, args("self", "name"))
        .def("setImageIndex", &CGraphicsOutput::setImageIndex, _setImageIndexDocString, args("self", "index"))
        .def("setItems", &CGraphicsOutput::setItems, _setItemsDocString, args("self", "items"))
        .def("getItems", &CGraphicsOutput::getItems, _getItemsDocString, args("self"))
        .def("getImageIndex", &CGraphicsOutput::getImageIndex, _getImageIndexDocString, args("self"))
        .def("addItem", &CGraphicsOutput::addItem, _addItemDocString, args("self", "item"))
        .def("addPoint", addPoint1, _addPoint1DocString, args("self", "point"))
        .def("addPoint", addPoint2, _addPoint2DocString, args("self", "point", "properties"))
        .def("addRectangle", addRectangle1, _addRectangle1DocString, args("self", "x", "y", "width", "height"))
        .def("addRectangle", addRectangle2, _addRectangle2DocString, args("self", "x", "y", "width", "height", "properties"))
        .def("addEllipse", addEllipse1, _addEllipse1DocString, args("self", "x", "y", "width", "height"))
        .def("addEllipse", addEllipse2, _addEllipse2DocString, args("self", "x", "y", "width", "height", "properties"))
        .def("addPolygon", addPolygon1, _addPolygon1DocString, args("self", "points"))
        .def("addPolygon", addPolygon2, _addPolygon2DocString, args("self", "points", "properties"))
        .def("addPolyline", addPolyline1, _addPolyline1DocString, args("self", "points"))
        .def("addPolyline", addPolyline2, _addPolyline2DocString, args("self", "points", "properties"))
        .def("addComplexPolygon", addComplexPolygon1, _addComplexPolygon1DocString, args("self", "outer", "inners"))
        .def("addComplexPolygon", addComplexPolygon2, _addComplexPolygon2DocString, args("self", "outer", "inners", "properties"))
        .def("addText", addText1, _addText1DocString, args("self", "text", "x", "y"))
        .def("addText", addText2, _addText2DocString, args("self", "text", "x", "y", "properties"))
        .def("load", &CGraphicsOutput::load, _graphicsOutputLoadDocString, args("self", "path"))
        .def("save", saveGraphicsOut, _graphicsOutputSaveDocString, args("self", "path"))
        .def("toJson", toJsonOut, _blobIOToJsonDocString, args("self", "options"))
        .def("fromJson", fromJsonOut, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //--------------------//
    //----- CImageIO -----//
    //--------------------//
    void (CImageIO::*drawGraphicsIn)(const GraphicsInputPtr&) = &CImageIO::drawGraphics;
    void (CImageIO::*drawGraphicsOut)(const GraphicsOutputPtr&) = &CImageIO::drawGraphics;
    CMat (CImageIO::*getImageWithGraphicsIn)(const GraphicsInputPtr&) = &CImageIO::getImageWithGraphics;
    CMat (CImageIO::*getImageWithGraphicsOut)(const GraphicsOutputPtr&) = &CImageIO::getImageWithGraphics;
    void (CImageIO::*saveImageIO)(const std::string&) = &CImageIO::save;

    class_<CImageIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CImageIOWrap>>("CImageIO", _imageProcessIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1imageProcessIODocString, args("self", "dataType")))
        .def(init<IODataType, const CMat&>(_ctor2imageProcessIODocString, args("self", "dataType", "image")))
        .def(init<IODataType, const CMat&, const std::string&>(_ctor3imageProcessIODocString, args("self", "dataType", "image", "name")))
        .def(init<IODataType, const std::string&>(_ctor4imageProcessIODocString, args("self", "dataType", "name")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor5imageProcessIODocString, args("self", "dataType", "name", "path")))
        .def(init<const CImageIO&>("Copy constructor"))
        .def("setImage", &CImageIO::setImage, _setImageDocString, args("self", "image"))
        .def("setOverlayMask", &CImageIO::setOverlayMask, _setOverlayMaskDocString, args("self", "mask"))
        .def("setChannelCount", &CImageIO::setChannelCount, _setChannelCountDocString, args("self", "nb"))
        .def("setCurrentImage", &CImageIO::setCurrentImage, _setCurrentImageDocString, args("self", "index"))
        .def("getChannelCount", &CImageIO::getChannelCount, _getChannelCountDocString, args("self"))
        .def("getData", &CImageIO::getData, _getDataDocString, args("self"))
        .def("getImage", &CImageIO::getImage, &CImageIOWrap::default_getImage, _getImageDocString, args("self"))
        .def("getImageWithGraphics", getImageWithGraphicsIn, _getImageWithGraphicsInDocString, args("self", "graphics"))
        .def("getImageWithGraphics", getImageWithGraphicsOut, _getImageWithGraphicsOutDocString, args("self", "graphics"))
        .def("getOverlayMask", &CImageIO::getOverlayMask, _getOverlayMaskDocString, args("self"))
        .def("getUnitElementCount", &CImageIO::getUnitElementCount, &CImageIOWrap::default_getUnitElementCount, _getImageUnitElementCountDocString, args("self"))
        .def("isDataAvailable", &CImageIO::isDataAvailable, &CImageIOWrap::default_isDataAvailable, _isImageDataAvailableDocString, args("self"))
        .def("isOverlayAvailable", &CImageIO::isOverlayAvailable, _isOverlayAvailableDocString, args("self"))
        .def("clearData", &CImageIO::clearData, &CImageIOWrap::default_clearData, _clearImageDataDocString, args("self"))
        .def("copyStaticData", &CImageIO::copyStaticData, &CImageIOWrap::default_copyStaticData, _copyImageStaticDataDocString, args("self", "io"))
        .def("drawGraphics", drawGraphicsIn, _drawGraphicsInDocString, args("self", "graphics"))
        .def("drawGraphics", drawGraphicsOut, _drawGraphicsOutDocString, args("self", "graphics"))
        .def("load", &CImageIO::load, _imageIOLoadDocString, args("self", "path"))
        .def("save", saveImageIO, _imageIOSaveDocString, args("self", "path"))
        .def("toJson", &CImageIO::toJson, &CImageIOWrap::default_toJson, _imageIOToJsonDocString, args("self", "options"))
        .def("fromJson", &CImageIO::fromJson, &CImageIOWrap::default_fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //----------------------//
    //----- CNumericIO -----//
    //----------------------//
    enum_<NumericOutputType>("NumericOutputType", "Enum - List of a display types for numeric values")
        .value("NONE", NumericOutputType::NONE)
        .value("TABLE", NumericOutputType::TABLE)
        .value("PLOT", NumericOutputType::PLOT)
        .value("NUMERIC_FILE", NumericOutputType::NUMERIC_FILE)
    ;

    enum_<PlotType>("PlotType", "Enum - List of plot types")
        .value("CURVE", PlotType::CURVE)
        .value("BAR", PlotType::BAR)
        .value("MULTIBAR", PlotType::MULTIBAR)
        .value("HISTOGRAM", PlotType::HISTOGRAM)
        .value("PIE", PlotType::PIE)
    ;

    exposeNumericIO<double>("CNumericIO");
    exposeNumericIO<std::string>("CDataStringIO");

    //--------------------//
    //----- CVideoIO -----//
    //--------------------//
    class_<CVideoIOWrap, bases<CImageIO>, std::shared_ptr<CVideoIOWrap>>("CVideoIO", _videoProcessIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1VideoProcessIODocString, args("self", "dataType")))
        .def(init<IODataType, const CMat&>(_ctor2VideoProcessIODocString, args("self", "dataType", "frame")))
        .def(init<IODataType, const CMat&, const std::string&>(_ctor3VideoProcessIODocString, args("self", "dataType", "frame", "name")))
        .def(init<IODataType, const std::string&>(_ctor4VideoProcessIODocString, args("self", "dataType", "name")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor5VideoProcessIODocString, args("self", "dataType", "name", "path")))
        .def(init<const CVideoIO&>("Copy constructor"))
        .def("setVideoPath", &CVideoIO::setVideoPath, _setVideoPathDocString, args("self", "path"))
        .def("setVideoPos", &CVideoIO::setVideoPos, _setVideoPosDocString, args("self", "position"))
        .def("getVideoFrameCount", &CVideoIO::getVideoFrameCount, _getVideoFrameCountDocString, args("self"))
        .def("getVideoImages", &CVideoIO::getVideoImages, _getVideoImagesDocString, args("self"))
        .def("getVideoPath", &CVideoIO::getVideoPath, _getVideoPathDocString, args("self"))
        .def("getSnapshot", &CVideoIO::getSnapshot, _getSnapshotDocString, args("self", "position"))
        .def("getCurrentPos", &CVideoIO::getCurrentPos, _getCurrentPosDocString, args("self"))
        .def("startVideo", &CVideoIO::startVideo, _startVideoDocString, args("self"))
        .def("stopVideo", &CVideoIO::stopVideo, _stopVideoDocString, args("self"))
        .def("startVideoWrite", &CVideoIO::startVideoWrite, _startVideoWriteDocString, args("self"))
        .def("stopVideoWrite", &CVideoIO::stopVideoWrite, _stopVideoWriteDocString, args("self"))
        .def("addVideoImage", &CVideoIO::addVideoImage, _addVideoImageDocString, args("self", "image"))
        .def("writeImage", &CVideoIO::writeImage, _writeImageDocString, args("self", "image"))
        .def("hasVideo", &CVideoIO::hasVideo, _hasVideoDocString, args("self"))
        .def("getImage", &CVideoIO::getImage, &CVideoIOWrap::default_getImage, _getVideoImageDocString, args("self"))
        .def("getUnitElementCount", &CVideoIO::getUnitElementCount, &CVideoIOWrap::default_getUnitElementCount, _getVideoUnitElementCountDocString, args("self"))
        .def("isDataAvailable", &CVideoIO::isDataAvailable, &CVideoIOWrap::default_isDataAvailable, _isVideoDataAvailableDocString, args("self"))
        .def("clearData", &CVideoIO::clearData, &CVideoIOWrap::default_clearData, _clearVideoDataDocString, args("self"))
        .def("copyStaticData", &CVideoIO::copyStaticData, &CVideoIOWrap::default_copyStaticData, _copyStaticDataDerivedDocString, args("self"))
    ;

    //-------------------------//
    //----- CWidgetOutput -----//
    //-------------------------//
    class_<CWidgetOutputWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CWidgetOutputWrap>>("CWidgetOutput", _widgetOutputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1WidgetOutputDocString, args("self", "dataType")))
        .def(init<IODataType, const std::string&>(_ctor2WidgetOutputDocString, args("self", "dataType", "name")))
        .def("setWidget", &CWidgetOutputWrap::setWidget, _setWidgetDocString, args("self", "widget"))
        .def("isDataAvailable", &CWidgetOutput::isDataAvailable, &CWidgetOutputWrap::default_isDataAvailable, _isWidgetDataAvailableDocString, args("self"))
        .def("clearData", &CWidgetOutput::clearData, &CWidgetOutputWrap::default_clearData, _clearWidgetDataDocString, args("self"))
    ;

    //-------------------//
    //----- CPathIO -----//
    //-------------------//
    class_<CPathIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CPathIOWrap>>("CPathIO", _pathIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1PathIODocString, args("self", "dataType")))
        .def(init<IODataType, const std::string&>(_ctor2PathIODocString, args("self", "dataType", "path")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor3PathIODocString, args("self", "dataType", "path", "name")))
        .def("setPath", &CPathIO::setPath, _setPathDocString, args("self", "path"))
        .def("getPath", &CPathIO::getPath, _getPathDocString, args("self"))
        .def("isDataAvailable", &CPathIO::isDataAvailable, &CPathIOWrap::default_isDataAvailable, _isVideoDataAvailableDocString, args("self"))
        .def("clearData", &CPathIO::clearData, &CPathIOWrap::default_clearData, _clearDataDocString, args("self"))
    ;

    //----------------------//
    //----- CDatasetIO -----//
    //----------------------//
    class_<std::map<int, std::string>>("MapIntStr", "Data structure (same as Python dict) to store int key and string value")
        .def(map_indexing_suite<std::map<int, std::string>>())
    ;

    class_<CDatasetIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CDatasetIOWrap>, boost::noncopyable>("CDatasetIO", _datasetIODocString)
        .def(init<>("Default constructor"))
        .def(init<const std::string&>(_ctor1DatasetIODocString, args("self", "name")))
        .def(init<const std::string&, const std::string&>(_ctor2DatasetIODocString, args("self", "name", "sourceFormat")))
        .def("getImagePaths", &CDatasetIOWrap::getImagePaths, &CDatasetIOWrap::default_getImagePaths, _getImagePathsDocStr)
        .def("getCategories", &CDatasetIOWrap::getCategories, &CDatasetIOWrap::default_getCategories, _getCategoriesDocStr)
        .def("getCategoryCount", &CDatasetIOWrap::getCategoryCount, &CDatasetIOWrap::default_getCategoryCount, _getCategoryCountDocStr)
        .def("getMaskPath", &CDatasetIOWrap::getMaskPath, &CDatasetIOWrap::default_getMaskPath, _getMaskPathDocStr)
        .def("getGraphicsAnnotations", &CDatasetIOWrap::getGraphicsAnnotations, &CDatasetIOWrap::default_getGraphicsAnnotations, _getGraphicsAnnotationsDocStr)
        .def("getSourceFormat", &CDatasetIOWrap::getSourceFormat, _getSourceFormatDocStr)
        .def("isDataAvailable", &CDatasetIOWrap::isDataAvailable, &CDatasetIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clearData", &CDatasetIOWrap::clearData, &CDatasetIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("save", &CDatasetIOWrap::save, &CDatasetIOWrap::default_save, _saveDocStr)
        .def("load", &CDatasetIOWrap::load, &CDatasetIOWrap::default_load, _loadDocStr)
        .def("toJson", &CDatasetIOWrap::toJson, &CDatasetIOWrap::default_toJson, _datasetIOToJsonDocStr, args("self", "options"))
        .def("fromJson", &CDatasetIOWrap::fromJson, &CDatasetIOWrap::default_fromJson, _datasetIOFromJsonDocStr, args("self", "jsonStr"))
    ;

    //--------------------//
    //----- CArrayIO -----//
    //--------------------//
    class_<CArrayIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CArrayIOWrap>>("CArrayIO", _arrayIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1ArrayIODocString, args("self", "name")))
        .def(init<const CMat&, const std::string&>(_ctor2ArrayIODocString, args("self", "array", "name")))
        .def(init<const CArrayIO&>("Copy constructor"))
        .def("setArray", &CArrayIO::setArray, _setArrayDocString, args("self", "array"))
        .def("getArray", &CArrayIO::getArray, _getArrayDocString, args("self"))
        .def("getUnitElementCount", &CArrayIO::getUnitElementCount, &CArrayIOWrap::default_getUnitElementCount, _getArrayUnitElementCountDocString, args("self"))
        .def("isDataAvailable", &CArrayIO::isDataAvailable, &CArrayIOWrap::default_isDataAvailable, _isArrayDataAvailableDocString, args("self"))
        .def("clearData", &CArrayIO::clearData, &CArrayIOWrap::default_clearData, _clearArrayDataDocString, args("self"))
    ;

    //------------------------------//
    //----- CObjectDetectionIO -----//
    //------------------------------//
    class_<CObjectDetection>("CObjectDetection", _objDetectionDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("id", &CObjectDetection::getId, &CObjectDetection::setId, "Object ID (int)")
        .add_property("label", &CObjectDetection::getLabel, &CObjectDetection::setLabel, "Object label (str)")
        .add_property("confidence", &CObjectDetection::getConfidence, &CObjectDetection::setConfidence, "Prediction confidence (double)")
        .add_property("box", &CObjectDetection::getBox, &CObjectDetection::setBox, "Object bounding box [x, y, width, height]")
        .add_property("color", &CObjectDetection::getColor, &CObjectDetection::setColor, "Object display color [r, g, b, a]")
    ;

    class_<CObjectDetectionIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CObjectDetectionIOWrap>>("CObjectDetectionIO", _objDetectionIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CObjectDetectionIO&>("Copy constructor"))
        .def("getObjectCount", &CObjectDetectionIO::getObjectCount, _getObjectCountDocString, args("self"))
        .def("getObject", &CObjectDetectionIO::getObject, _getObjectDocString, args("self", "index"))
        .def("getObjects", &CObjectDetectionIO::getObjects, _getObjectsDocString, args("self"))
        .def("isDataAvailable", &CObjectDetectionIOWrap::isDataAvailable, &CObjectDetectionIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("init", &CObjectDetectionIO::init, _initObjDetectIODocString, args("self", "taskName", "refImageIndex"))
        .def("addObject", &CObjectDetectionIO::addObject, _addObjectDocString, args("self", "id", "label", "confidence", "boxX", "boxY", "boxWidth", "boxHeight", "color"))
        .def("clearData", &CObjectDetectionIOWrap::clearData, &CObjectDetectionIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CObjectDetectionIOWrap::load, &CObjectDetectionIOWrap::default_load, _objDetectLoadDocString, args("self", "path"))
        .def("save", &CObjectDetectionIOWrap::save, &CObjectDetectionIOWrap::default_save, _objDetectSaveDocString, args("self", "path"))
        .def("toJson", &CObjectDetectionIOWrap::toJson, &CObjectDetectionIOWrap::default_toJson, _objDetectToJsonDocString, args("self", "options"))
        .def("fromJson", &CObjectDetectionIOWrap::fromJson, &CObjectDetectionIOWrap::default_fromJson, _objDetectFromJsonDocString, args("self", "jsonStr"))
    ;

    //--------------------------//
    //----- CInstanceSegIO -----//
    //--------------------------//
    class_<CInstanceSegmentation>("CInstanceSegmentation", _instanceSegDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("id", &CInstanceSegmentation::getId, &CInstanceSegmentation::setId, "Object ID (int)")
        .add_property("type", &CInstanceSegmentation::getType, &CInstanceSegmentation::setType, "Object type (int 0:THING or 1:STUFF)")
        .add_property("class_index", &CInstanceSegmentation::getClassIndex, &CInstanceSegmentation::setClassIndex, "Object class index (int)")
        .add_property("label", &CInstanceSegmentation::getLabel, &CInstanceSegmentation::setLabel, "Object label (str)")
        .add_property("confidence", &CInstanceSegmentation::getConfidence, &CInstanceSegmentation::setConfidence, "Prediction confidence (double)")
        .add_property("box", &CInstanceSegmentation::getBox, &CInstanceSegmentation::setBox, "Object bounding box [x, y, width, height]")
        .add_property("mask", &CInstanceSegmentation::getMask, &CInstanceSegmentation::setMask, "Object mask (numpy array)")
        .add_property("color", &CInstanceSegmentation::getColor, &CInstanceSegmentation::setColor, "Object display color [r, g, b, a]")
    ;

    class_<CInstanceSegIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CInstanceSegIOWrap>>("CInstanceSegIO", _instanceSegIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CInstanceSegIO&>("Copy constructor"))
        .def("getInstanceCount", &CInstanceSegIO::getInstanceCount, _getInstanceCountDocString, args("self"))
        .def("getInstance", &CInstanceSegIO::getInstance, _getInstanceDocString, args("self", "index"))
        .def("getInstances", &CInstanceSegIO::getInstances, _getInstancesDocString, args("self"))
        .def("getMergeMask", &CInstanceSegIO::getMergeMask, _getMergeMaskDocString, args("self"))
        .def("isDataAvailable", &CInstanceSegIOWrap::isDataAvailable, &CInstanceSegIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("init", &CInstanceSegIO::init, _initInstanceSegIODocString, args("self", "taskName", "refImageIndex", "width", "heigh"))
        .def("addInstance", &CInstanceSegIO::addInstance, _addInstanceDocString, args("self", "id", "type", "classIndex", "label", "confidence", "boxX", "boxY", "boxWidth", "boxHeight", "mask", "color"))
        .def("clearData", &CInstanceSegIOWrap::clearData, &CInstanceSegIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CInstanceSegIOWrap::load, &CInstanceSegIOWrap::default_load, _instanceSegLoadDocString, args("self", "path"))
        .def("save", &CInstanceSegIOWrap::save, &CInstanceSegIOWrap::default_save, _instanceSegSaveDocString, args("self", "path"))
        .def("toJson", &CInstanceSegIOWrap::toJson, &CInstanceSegIOWrap::default_toJson, _instanceSegToJsonDocString, args("self", "options"))
        .def("fromJson", &CInstanceSegIOWrap::fromJson, &CInstanceSegIOWrap::default_fromJson, _instanceSegFromJsonDocString, args("self", "jsonStr"))
    ;

    //--------------------------//
    //----- CSemanticSegIO -----//
    //--------------------------//
    class_<CSemanticSegIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CSemanticSegIOWrap>>("CSemanticSegIO", _semanticSegIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CSemanticSegIO&>("Copy constructor"))
        .def("getMask", &CSemanticSegIO::getMask, _getMaskDocString, args("self"))
        .def("getClassNames", &CSemanticSegIO::getClassNames, _getClassNamesDocString, args("self"))
        .def("setMask", &CSemanticSegIO::setMask, _setMaskDocString, args("self", "mask"))
        .def("setClassNames", &CSemanticSegIOWrap::setClassNames, _setClassNamesDocString, args("self", "names", "colors"))
        .def("isDataAvailable", &CSemanticSegIOWrap::isDataAvailable, &CSemanticSegIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clearData", &CSemanticSegIOWrap::clearData, &CSemanticSegIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CSemanticSegIOWrap::load, &CSemanticSegIOWrap::default_load, _instanceSegLoadDocString, args("self", "path"))
        .def("save", &CSemanticSegIOWrap::save, &CSemanticSegIOWrap::default_save, _instanceSegSaveDocString, args("self", "path"))
        .def("toJson", &CSemanticSegIOWrap::toJson, &CSemanticSegIOWrap::default_toJson, _instanceSegToJsonDocString, args("self", "options"))
        .def("fromJson", &CSemanticSegIOWrap::fromJson, &CSemanticSegIOWrap::default_fromJson, _instanceSegFromJsonDocString, args("self", "jsonStr"))
    ;

    //------------------------//
    //----- C2dImageTask -----//
    //------------------------//
    class_<C2dImageTaskWrap, bases<CWorkflowTask>, std::shared_ptr<C2dImageTaskWrap>>("C2dImageTask", _imageProcess2dDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<bool>(_ctor1ImageProcess2dDocString, args("self", "hasGraphicsInput")))
        .def(init<const std::string&>(_ctor2ImageProcess2dDocString, args("self", "name")))
        .def(init<const std::string&, bool>(_ctor3ImageProcess2dDocString, args("self", "name", "hasGraphicsInput")))
        .def("setActive", &C2dImageTask::setActive, &C2dImageTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("setOutputColorMap", &C2dImageTaskWrap::setOutputColorMap, _setOutputColorMapDocString, args("self", "index", "mask_index", "colors"))
        .def("updateStaticOutputs", &C2dImageTask::updateStaticOutputs, &C2dImageTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("beginTaskRun", &C2dImageTask::beginTaskRun, &C2dImageTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("endTaskRun", &C2dImageTask::endTaskRun, &C2dImageTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphicsChanged", &C2dImageTask::graphicsChanged, &C2dImageTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("globalInputChanged", &C2dImageTask::globalInputChanged, &C2dImageTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("createGraphicsMask", &C2dImageTask::createGraphicsMask, _createGraphicsMaskDocString, args("self", "width", "height", "graphics"))
        .def("applyGraphicsMask", &C2dImageTask::applyGraphicsMask, _applyGraphicsMaskDocString, args("self", "src", "dst", "index"))
        .def("applyGraphicsMaskToBinary", &C2dImageTask::applyGraphicsMaskToBinary, _applyGraphicsMaskToBinaryDocString, args("self", "src", "dst", "index"))
        .def("getProgressSteps", &C2dImageTaskWrap::getProgressSteps, &C2dImageTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("getGraphicsMask", &C2dImageTask::getGraphicsMask, _getGraphicsMaskDocString, args("self", "index"))
        .def("isMaskAvailable", &C2dImageTask::isMaskAvailable, _isMaskAvailableDocString, args("self", "index"))
        .def("run", &C2dImageTask::run, &C2dImageTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &C2dImageTask::stop, &C2dImageTaskWrap::default_stop, _stopDocString, args("self"))
        .def("forwardInputImage", &C2dImageTask::forwardInputImage, _forwardInputImageDocString, args("self", "input_index", "output_index"))
        .def("emitAddSubProgressSteps", &C2dImageTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &C2dImageTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitGraphicsContextChanged", &C2dImageTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emitOutputChanged", &C2dImageTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
    ;

    //-----------------------------------//
    //----- C2dImageInteractiveTask -----//
    //-----------------------------------//
    class_<C2dImageInteractiveTaskWrap, bases<C2dImageTask>, std::shared_ptr<C2dImageInteractiveTaskWrap>>("C2dImageInteractiveTask", _interactiveImageProcess2d)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorInteractiveImageProcessDocString, args("self", "name")))
        .def("setActive", &C2dImageInteractiveTask::setActive, &C2dImageInteractiveTaskWrap::default_setActive, _setActiveInteractiveDocString, args("self", "is_active"))
        .def("updateStaticOutputs", &C2dImageInteractiveTask::updateStaticOutputs, &C2dImageInteractiveTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("beginTaskRun", &C2dImageInteractiveTask::beginTaskRun, &C2dImageInteractiveTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("endTaskRun", &C2dImageInteractiveTask::endTaskRun, &C2dImageInteractiveTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphicsChanged", &C2dImageInteractiveTask::graphicsChanged, &C2dImageInteractiveTaskWrap::default_graphicsChanged, _graphicsChangedInteractiveDocString, args("self"))
        .def("globalInputChanged", &C2dImageInteractiveTask::globalInputChanged, &C2dImageInteractiveTaskWrap::default_globalInputChanged, _globalInputChangedInteractiveDocString, args("self", "is_new_sequence"))
        .def("getProgressSteps", &C2dImageInteractiveTaskWrap::getProgressSteps, &C2dImageInteractiveTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("getInteractionMask", &C2dImageInteractiveTask::getInteractionMask, _getInteractionMaskDocString, args("self"))
        .def("getBlobs", &C2dImageInteractiveTask::getBlobs, _getBlobsDocString, args("self"))
        .def("createInteractionMask", &C2dImageInteractiveTask::createInteractionMask, _createInteractionMaskDocString, args("self", "width", "height"))
        .def("computeBlobs", &C2dImageInteractiveTask::computeBlobs, _computeBlobsDocString, args("self"))
        .def("clearInteractionLayer", &C2dImageInteractiveTask::clearInteractionLayer, _clearInteractionLayerDocString, args("self"))
        .def("run", &C2dImageInteractiveTask::run, &C2dImageInteractiveTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &C2dImageInteractiveTask::stop, &C2dImageInteractiveTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emitAddSubProgressSteps", &C2dImageInteractiveTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &C2dImageInteractiveTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitGraphicsContextChanged", &C2dImageInteractiveTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emitOutputChanged", &C2dImageInteractiveTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
    ;

    //----------------------//
    //----- CVideoTask -----//
    //----------------------//
    class_<CVideoTaskWrap, bases<C2dImageTask>, std::shared_ptr<CVideoTaskWrap>>("CVideoTask", _videoProcessDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoProcessDocString, args("self", "name")))
        .def("setActive", &CVideoTask::setActive, &CVideoTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("updateStaticOutputs", &CVideoTask::updateStaticOutputs, &CVideoTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("beginTaskRun", &CVideoTask::beginTaskRun, &CVideoTaskWrap::default_beginTaskRun, _beginTaskRunVideoDocString, args("self"))
        .def("endTaskRun", &CVideoTask::endTaskRun, &CVideoTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("notifyVideoStart", &CVideoTask::notifyVideoStart, &CVideoTaskWrap::default_notifyVideoStart, _notifyVideoStartDocString, args("self", "frame_count"))
        .def("notifyVideoEnd", &CVideoTask::notifyVideoEnd, &CVideoTaskWrap::default_notifyVideoEnd, _notifyVideoEndDocString, args("self"))
        .def("graphicsChanged", &CVideoTask::graphicsChanged, &CVideoTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("globalInputChanged", &CVideoTask::globalInputChanged, &CVideoTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("getProgressSteps", &CVideoTaskWrap::getProgressSteps, &CVideoTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoTask::run, &CVideoTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoTask::stop, &CVideoTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emitAddSubProgressSteps", &CVideoTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &CVideoTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitGraphicsContextChanged", &CVideoTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emitOutputChanged", &CVideoTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
    ;

    //------------------------//
    //----- CVideoOFTask -----//
    //------------------------//
    class_<CVideoOFTaskWrap, bases<CVideoTask>, std::shared_ptr<CVideoOFTaskWrap>>("CVideoOFTask", _videoProcessOFDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoProcessOFDocString, args("self", "name")))
        .def("setActive", &CVideoOFTask::setActive, &CVideoOFTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("updateStaticOutputs", &CVideoOFTask::updateStaticOutputs, &CVideoOFTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("beginTaskRun", &CVideoOFTask::beginTaskRun, &CVideoOFTaskWrap::default_beginTaskRun, _beginTaskRunVideoOFDocString, args("self"))
        .def("endTaskRun", &CVideoOFTask::endTaskRun, &CVideoOFTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphicsChanged", &CVideoOFTask::graphicsChanged, &CVideoOFTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("globalInputChanged", &CVideoOFTask::globalInputChanged, &CVideoOFTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self" "is_new_sequence"))
        .def("getProgressSteps", &CVideoOFTaskWrap::getProgressSteps, &CVideoOFTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoOFTask::run, &CVideoOFTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoOFTask::stop, &CVideoOFTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emitAddSubProgressSteps", &CVideoOFTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &CVideoOFTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitGraphicsContextChanged", &CVideoOFTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emitOutputChanged", &CVideoOFTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("drawOptFlowMap", &CVideoOFTask::drawOptFlowMap, _drawOptFlowMapDocString, args("self", "flow", "vectors", "step"))
        .def("flowToDisplay", &CVideoOFTask::flowToDisplay, _flowToDisplayDocString, args("self", "flow"))
    ;

    //------------------------------//
    //----- CVideoTrackingTask -----//
    //------------------------------//
    class_<CVideoTrackingTaskWrap, bases<CVideoTask>, std::shared_ptr<CVideoTrackingTaskWrap>>("CVideoTrackingTask", _videoProcessTrackingDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoTrackingDocString, args("self", "name")))
        .def("setActive", &CVideoTrackingTask::setActive, &CVideoTrackingTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("updateStaticOutputs", &CVideoTrackingTask::updateStaticOutputs, &CVideoTrackingTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("beginTaskRun", &CVideoTrackingTask::beginTaskRun, &CVideoTrackingTaskWrap::default_beginTaskRun, _beginTaskRunVideoOFDocString, args("self"))
        .def("endTaskRun", &CVideoTrackingTask::endTaskRun, &CVideoTrackingTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphicsChanged", &CVideoTrackingTask::graphicsChanged, &CVideoTrackingTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("globalInputChanged", &CVideoTrackingTask::globalInputChanged, &CVideoTrackingTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("getProgressSteps", &CVideoTrackingTaskWrap::getProgressSteps, &CVideoTrackingTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoTrackingTask::run, &CVideoTrackingTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoTrackingTask::stop, &CVideoTrackingTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emitAddSubProgressSteps", &CVideoTrackingTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &CVideoTrackingTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitGraphicsContextChanged", &CVideoTrackingTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emitOutputChanged", &CVideoTrackingTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("setRoiToTrack", &CVideoTrackingTask::setRoiToTrack, _setRoiToTrackDocString, args("self"))
        .def("manageOutputs", &CVideoTrackingTask::manageOutputs, _manageOutputsDocString, args("self"))
    ;

    //-------------------------//
    //----- CDnnTrainTask -----//
    //-------------------------//
    class_<CDnnTrainTaskWrap, bases<CWorkflowTask>, std::shared_ptr<CDnnTrainTaskWrap>>("CDnnTrainTask", _dnnTrainProcessDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1DnnTrainProcessDocString, args("self", "name")))
        .def(init<const std::string&, const std::shared_ptr<CWorkflowTaskParam>&>(_ctor2DnnTrainProcessDocString, args("self", "name", "param")))
        .def("beginTaskRun", &CDnnTrainTask::beginTaskRun, &CDnnTrainTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("emitAddSubProgressSteps", &CDnnTrainTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emitStepProgress", &CDnnTrainTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emitOutputChanged", &CDnnTrainTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("enableMlflow", &CDnnTrainTask::enableMlflow, _enableMlflowDocString, args("self", "enable"))
        .def("enableTensorboard", &CDnnTrainTask::enableTensorboard, _enableTensorboardDocString, args("self", "enable"))
        .def("endTaskRun", &CDnnTrainTask::endTaskRun, &CDnnTrainTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("getProgressSteps", &CDnnTrainTaskWrap::getProgressSteps, &CDnnTrainTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CDnnTrainTask::run, &CDnnTrainTaskWrap::default_run, _runDocString, args("self"))
        .def("setActive", &CDnnTrainTask::setActive, &CDnnTrainTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("stop", &CDnnTrainTask::stop, &CDnnTrainTaskWrap::default_stop, _stopDocString, args("self"))
    ;

    //---------------------------//
    //----- CIkomiaRegistry -----//
    //---------------------------//
    WorkflowTaskPtr (CIkomiaRegistry::*createInstance1)(const std::string&) = &CIkomiaRegistryWrap::createInstance;
    WorkflowTaskPtr (CIkomiaRegistry::*createInstance2)(const std::string&, const WorkflowTaskParamPtr&) = &CIkomiaRegistryWrap::createInstance;

    class_<CIkomiaRegistryWrap, std::shared_ptr<CIkomiaRegistryWrap>>("CIkomiaRegistry", _ikomiaRegistryDocString)
        .def(init<>("Default constructor", args("self")))
        .def("setPluginsDirectory", &CIkomiaRegistryWrap::setPluginsDirectory, _setPluginsDirDocString, args("self", "directory"))
        .def("getPluginsDirectory", &CIkomiaRegistryWrap::getPluginsDirectory, _getPluginsDirDocString, args("self"))
        .def("getAlgorithms", &CIkomiaRegistryWrap::getAlgorithms, _getAlgorithmsDocString, args("self)"))
        .def("getAlgorithmInfo", &CIkomiaRegistryWrap::getAlgorithmInfo, _getAlgorithmInfoDocString, args("self", "name"))
        .def("createInstance", createInstance1, _createInstance1DocString, args("self", "name"))
        .def("createInstance", createInstance2, _createInstance2DocString, args("self", "name", "parameters"))
        .def("registerTask", &CIkomiaRegistryWrap::registerTask, _registerTaskDocString, args("self", "factory"))
        .def("registerIO", &CIkomiaRegistryWrap::registerIO, _registerIODocString, args("self", "factory"))
        .def("loadCppPlugin", &CIkomiaRegistryWrap::loadCppPlugin, _loadCppPluginDocString, args("self", "path"))
        .def("getBlackListedPackages", &CIkomiaRegistry::getBlackListedPackages)
        .staticmethod("getBlackListedPackages")
    ;

    //---------------------//
    //----- CWorkflow -----//
    //---------------------//
    registerStdVector<std::intptr_t>();

    void (CWorkflowWrap::*addInputRef)(const WorkflowTaskIOPtr&) = &CWorkflowWrap::addInput;

    class_<CWorkflowWrap, bases<CWorkflowTask>, std::shared_ptr<CWorkflowWrap>>("CWorkflow", _workflowDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1WorkflowDocString, args("self", "name")))
        .def(init<const std::string&, const std::shared_ptr<CIkomiaRegistry>&>(_ctor2WorkflowDocString, args("self", "name", "registry")))
        .add_property("description", &CWorkflowWrap::getDescription, &CWorkflow::setDescription, "Workflow description")
        .add_property("keywords", &CWorkflowWrap::getKeywords, &CWorkflow::setKeywords, "Workflow associated keywords")
        .def("setInput", &CWorkflowWrap::setInput, _wfSetInputDocString, args("self", "input", "index", "new_sequence"))
        .def("setOutputFolder", &CWorkflowWrap::setOutputFolder, _wfSetOutputFolderDocString, args("self", "path"))
        .def("setAutoSave", &CWorkflowWrap::setAutoSave, _wfSetAutoSaveDocString, args("self", "enable"))
        .def("setCfgEntry", &CWorkflowWrap::setCfgEntry, _wfSetCfgEntryDocString, args("self", "key", "value"))
        .def("getTaskCount", &CWorkflowWrap::getTaskCount, _wfGetTaskCountDocString, args("self"))
        .def("getRootID", &CWorkflowWrap::getRootID, _wfGetRootIDDocString, args("self"))
        .def("getTaskIDs", &CWorkflowWrap::getTaskIDs, _wfGetTaskIDsDocString, args("self"))
        .def("getTask", &CWorkflowWrap::getTask, _wfGetTaskDocString, args("self", "id"))
        .def("getParents", &CWorkflowWrap::getParents, _wfGetParentsDocString, args("self", "id"))
        .def("getChildren", &CWorkflowWrap::getChildren, _wfGetChildrenDocString, args("self", "id"))
        .def("getInEdges", &CWorkflowWrap::getInEdges, _wfGetInEdgesDocString, args("self", "id"))
        .def("getOutEdges", &CWorkflowWrap::getOutEdges, _wfGetOutEdgesDocString, args("self", "id"))
        .def("getEdgeInfo", &CWorkflowWrap::getEdgeInfo, _wfGetEdgeInfoDocString, args("self", "id"))
        .def("getEdgeSource", &CWorkflowWrap::getEdgeSource, _wfGetEdgeSourceDocString, args("self", "id"))
        .def("getEdgeTarget", &CWorkflowWrap::getEdgeTarget, _wfGetEdgeTargetDocString, args("self", "id"))
        .def("getFinalTasks", &CWorkflowWrap::getFinalTasks, _wfGetFinalTasks, args("self"))
        .def("getRootTargetTypes", &CWorkflowWrap::getRootTargetTypes, _wfGetRootTargetTypesDocString, args("self"))
        .def("getTotalElapsedTime", &CWorkflowWrap::getTotalElapsedTime, _wfGetTotalElapsedTimeDocString, args("self"))
        .def("getElapsedTimeTo", &CWorkflowWrap::getElapsedTimeTo, _wfGetElapsedTimeToDocString, args("self"))
        .def("getRequiredTasks", &CWorkflow::getRequiredTasks, _wfGetRequiredTasks, args("self", "path"))
        .def("addInput", addInputRef, _wfAddInputDocString, args("self", "input"))
        .def("addTask", &CWorkflowWrap::addTaskWrap, _wfAddTaskDocString, args("self", "task"))
        .def("connect", &CWorkflowWrap::connectWrap, _wfConnectDocString, args("self", "source", "target", "source_index", "target_index"))
        .def("removeInput", &CWorkflowWrap::removeInput, _wfRemoveInputDocString, args("self", "index"))
        .def("clearInputs", &CWorkflowWrap::clearInputs, _wfClearInputsDocString, args("self"))
        .def("clearOutputData", &CWorkflowWrap::clearAllOutputData, _wfClearOutputDataDocString, args("self"))
        .def("clear", &CWorkflowWrap::clearWrap, _wfClearDocString, args("self"))
        .def("deleteTask", &CWorkflowWrap::deleteTaskWrap, _wfDeleteTaskDocString, args("self", "id"))
        .def("deleteEdge", &CWorkflowWrap::deleteEdgeWrap, _wfDeleteEdgeDocString, args("self", "id"))
        .def("run", &CWorkflowWrap::run, &CWorkflowWrap::default_run, _wfRunDocString, args("self"))
        .def("stop", &CWorkflowWrap::stop, _wfStopDocString, args("self"))
        .def("updateStartTime", &CWorkflowWrap::updateStartTime, _wfUpdateStartTimeDocString, args("self"))
        .def("load", &CWorkflowWrap::loadWrap, _wfLoadDocString, args("self", "path"))
        .def("save", &CWorkflowWrap::save, _wfSaveDocString, args("self", "path"))
        .def("exportGraphviz", &CWorkflowWrap::writeGraphviz, _wfExportGraphvizDocString, args("self", "path"))
    ;
}
