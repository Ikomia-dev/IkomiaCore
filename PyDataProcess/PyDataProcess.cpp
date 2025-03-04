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
#include "VectorConverter.hpp"
#include "MapConverter.hpp"
#include "PyDataProcessDocString.hpp"
#include "Task/CTaskFactoryWrap.h"
#include "Task/CTaskParamFactoryWrap.h"
#include "Task/C2dImageTaskWrap.h"
#include "Task/C2dImageInteractiveTaskWrap.h"
#include "Task/CVideoTaskWrap.h"
#include "Task/CVideoOFTaskWrap.h"
#include "Task/CVideoTrackingTaskWrap.h"
#include "Task/CDnnTrainTaskWrap.h"
#include "Task/CClassifTaskWrap.h"
#include "Task/CObjDetectTaskWrap.h"
#include "Task/CSemanticSegTaskWrap.h"
#include "Task/CInstanceSegTaskWrap.h"
#include "Task/CKeyptsDetectTaskWrap.h"
#include "CWidgetFactoryWrap.h"
#include "CPluginProcessInterfaceWrap.h"
#include "IO/CNumericIOWrap.hpp"
#include "IO/CGraphicsInputWrap.h"
#include "IO/CGraphicsOutputWrap.h"
#include "IO/CImageIOWrap.h"
#include "IO/CVideoIOWrap.h"
#include "IO/CWidgetOutputWrap.h"
#include "IO/CDatasetIOWrap.h"
#include "IO/CPathIOWrap.h"
#include "IO/CArrayIOWrap.h"
#include "IO/CObjectDetectionIOWrap.h"
#include "IO/CInstanceSegIOWrap.h"
#include "IO/CSemanticSegIOWrap.h"
#include "IO/CTextIOWrap.h"
#include "IO/CKeyptsIOWrap.h"
#include "CIkomiaRegistryWrap.h"
#include "CWorkflowWrap.h"

void translateCException(const CException& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
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
    std::string (CNumericIO<Type>::*numIOToJsonNoOpt)() const = &CNumericIO<Type>::toJson;
    std::string (CNumericIO<Type>::*numIOToJson)(const std::vector<std::string>&) const = &CNumericIO<Type>::toJson;

    class_<CNumericIOWrap<Type>, bases<CWorkflowTaskIO>, std::shared_ptr<CNumericIOWrap<Type>>>(className.c_str(), _featureProcessIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorFeatureIODocString, args("self", "name")))
        .def(init<const CNumericIO<Type>&>("Copy constructor"))
        .def("__repr__", &CNumericIO<Type>::repr)
        .def("set_output_type", &CNumericIO<Type>::setOutputType, _setOutputTypeDocString, args("self", "type"))
        .def("set_plot_type", &CNumericIO<Type>::setPlotType, _setPlotTypeDocString, args("self", "type"))
        .def("add_value_list", addValueList1, _addValueList1DocString, args("self", "values"))
        .def("add_value_list", addValueList2, _addValueList2DocString, args("self", "values", "header_label"))
        .def("add_value_list", addValueList3, _addValueList3DocString, args("self", "values", "labels"))
        .def("add_value_list", addValueList4, _addValueList4DocString, args("self", "values", "header_label", "labels"))
        .def("get_output_type", &CNumericIO<Type>::getOutputType, _getOutputTypeDocString, args("self"))
        .def("get_plot_type", &CNumericIO<Type>::getPlotType, _getPlotTypeDocString, args("self"))
        .def("get_value_list", &CNumericIO<Type>::getValueList, _getValueListDocString, args("self", "index"))
        .def("get_all_value_list", &CNumericIO<Type>::getAllValues, _getAllValueListDocString, args("self"))
        .def("get_all_label_list", &CNumericIO<Type>::getAllValueLabels, _getAllLabelListDocString, args("self"))
        .def("get_all_header_labels", &CNumericIO<Type>::getAllHeaderLabels, _getAllHeaderLabelsDocString, args("self"))
        .def("get_unit_element_count", &CNumericIO<Type>::getUnitElementCount, &CNumericIOWrap<Type>::default_getUnitElementCount, _getUnitEltCountDerivedDocString, args("self"))
        .def("is_data_available", &CNumericIO<Type>::isDataAvailable, &CNumericIOWrap<Type>::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clear_data", &CNumericIO<Type>::clearData, &CNumericIOWrap<Type>::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("copy_static_data", &CNumericIO<Type>::copyStaticData, &CNumericIOWrap<Type>::default_copyStaticData, _copyStaticDataDerivedDocString, args("self", "io"))
        .def("load", &CNumericIO<Type>::load, &CNumericIOWrap<Type>::default_load, _numericIOLoadDocString, args("self", "path"))
        .def("save", saveNumeric, &CNumericIOWrap<Type>::default_save, _numericIOSaveDocString, args("self", "path"))
        .def("to_json", numIOToJsonNoOpt, &CNumericIOWrap<Type>::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", numIOToJson, &CNumericIOWrap<Type>::default_toJson, _blobIOToJsonDocString, args("self", "options"))
        .def("from_json", &CNumericIO<Type>::fromJson, &CNumericIOWrap<Type>::default_fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    register_ptr_to_python<std::shared_ptr<CNumericIO<Type>>>();
}

BOOST_PYTHON_MODULE(pydataprocess)
{
    // Enable user-defined docstrings and python signatures, while disabling the C++ signatures
    docstring_options local_docstring_options(true, true, false);

    // Set the docstring of the current module scope
    scope().attr("__doc__") = _moduleDocString;

    //Numpy initialization
    CvMatNumpyArrayConverter::init_numpy();

    //Register smart pointers
    register_ptr_to_python<std::shared_ptr<CTaskFactory>>();
    register_ptr_to_python<std::shared_ptr<CTaskParamFactory>>();
    register_ptr_to_python<std::shared_ptr<CWidgetFactory>>();
    register_ptr_to_python<std::shared_ptr<CGraphicsInput>>();
    register_ptr_to_python<std::shared_ptr<CGraphicsOutput>>();
    register_ptr_to_python<std::shared_ptr<CImageIO>>();
    register_ptr_to_python<std::shared_ptr<CVideoIO>>();
    register_ptr_to_python<std::shared_ptr<CWidgetOutput>>();
    register_ptr_to_python<std::shared_ptr<CPathIO>>();
    register_ptr_to_python<std::shared_ptr<CDatasetIO>>();
    register_ptr_to_python<std::shared_ptr<CArrayIO>>();
    register_ptr_to_python<std::shared_ptr<CObjectDetectionIO>>();
    register_ptr_to_python<std::shared_ptr<CSemanticSegIO>>();
    register_ptr_to_python<std::shared_ptr<CInstanceSegIO>>();
    register_ptr_to_python<std::shared_ptr<CKeypointsIO>>();
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
    registerStdVector<CMat>();
    registerStdVector<std::vector<cv::Point>>();
    registerStdVector<CObjectKeypoints>();
    registerStdVector<CKeypointLink>();
    registerStdVector<CTextField>();

    // Register std::map<T>
    registerStdMap<int, std::string>();

    //Register exceptions
    register_exception_translator<CException>(&translateCException);

    //---------------------//
    //----- CTaskInfo -----//
    //---------------------//
    class_<CTaskInfo>("CTaskInfo", _processInfoDocString, init<>("Default constructor"))
        .add_property("name", &CTaskInfo::getName, &CTaskInfo::setName, "Name of the plugin (mandatory - must be unique)")
        .add_property("path", &CTaskInfo::getPath, &CTaskInfo::setPath, "Path in the library tree view of Ikomia")
        .add_property("short_description", &CTaskInfo::getShortDescription, &CTaskInfo::setShortDescription, "Short description of the plugin (mandatory)")
        .add_property("description", &CTaskInfo::getDescription, &CTaskInfo::setDescription, "Full description of the plugin (mandatory)")
        .add_property("documentation_link", &CTaskInfo::getDocumentationLink, &CTaskInfo::setDocumentationLink, "Address (URL) of online documentation")
        .add_property("icon_path", &CTaskInfo::getIconPath, &CTaskInfo::setIconPath, "Relative path to the plugin icon")
        .add_property("keywords", &CTaskInfo::getKeywords, &CTaskInfo::setKeywords, "Keywords associated with the plugin (Used for Ikomia search engine)")
        .add_property("authors", &CTaskInfo::getAuthors, &CTaskInfo::setAuthors, "Authors of the plugin and/or corresponding paper (mandatory)")
        .add_property("article", &CTaskInfo::getArticle, &CTaskInfo::setArticle, "Title of the corresponding paper")
        .add_property("article_url", &CTaskInfo::getArticleUrl, &CTaskInfo::setArticleUrl, "Link (URL) to the corresponding paper")
        .add_property("journal", &CTaskInfo::getJournal, &CTaskInfo::setJournal, "Paper journal")
        .add_property("version", &CTaskInfo::getVersion, &CTaskInfo::setVersion, "Plugin version (mandatory)")
        .add_property("min_ikomia_version", &CTaskInfo::getMinIkomiaVersion, &CTaskInfo::setMinIkomiaVersion, "Minimum Ikomia API version")
        .add_property("max_ikomia_version", &CTaskInfo::getMaxIkomiaVersion, &CTaskInfo::setMaxIkomiaVersion, "Maximum Ikomia API version")
        .add_property("min_python_version", &CTaskInfo::getMinPythonVersion, &CTaskInfo::setMinPythonVersion, "Minimum Python version")
        .add_property("max_python_version", &CTaskInfo::getMaxPythonVersion, &CTaskInfo::setMaxPythonVersion, "Maximum Python version")
        .add_property("year", &CTaskInfo::getYear, &CTaskInfo::setYear, "Year of paper publication")
        .add_property("language", &CTaskInfo::getLanguage, &CTaskInfo::setLanguage, "Language: Python")
        .add_property("license", &CTaskInfo::getLicense, &CTaskInfo::setLicense, "License of the plugin")
        .add_property("repository", &CTaskInfo::getRepository, &CTaskInfo::setRepository, "URL of code repository (GitHub, GitLab, BitBucket...)")
        .add_property("original_repository", &CTaskInfo::getOriginalRepository, &CTaskInfo::setOriginalRepository, "URL of original code repository (GitHub, GitLab, BitBucket...)")
        .add_property("os", &CTaskInfo::getOS, &CTaskInfo::setOS, "Operating system")
        .add_property("algo_type", &CTaskInfo::getAlgoType, &CTaskInfo::setAlgoType, "Algorithm type: train, inference, dataset...")
        .add_property("algo_tasks", &CTaskInfo::getAlgoTasks, &CTaskInfo::setAlgoTasks, "Algorithm tasks: classification, object detection, segmentation...")
        .add_property("internal", &CTaskInfo::isInternal, "Indicate a built-in algorithm.")
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

    //-----------------------------//
    //----- CTaskParamFactory -----//
    //-----------------------------//
    class_<CTaskParamFactoryWrap, std::shared_ptr<CTaskParamFactoryWrap>, boost::noncopyable>("CTaskParamFactory", _taskParamFactoryDocString)
        .add_property("name", &CTaskParamFactory::getName, &CTaskParamFactory::setName, _taskParamFactoryNameDocString)
        .def("create", pure_virtual(&CTaskParamFactory::create), _createTaskParamDocString, args("self"))
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
        .def("get_process_factory", pure_virtual(&CPluginProcessInterface::getProcessFactory), _getProcessFactoryDocString, args("self"))
        .def("get_widget_factory", pure_virtual(&CPluginProcessInterface::getWidgetFactory), _getWidgetFactoryDocString, args("self"))
        .def("get_param_factory", &CPluginProcessInterface::getParamFactory, &CPluginProcessInterfaceWrap::default_getParamFactory, _getParamFactoryDocString, args("self"))
    ;

    //--------------------------//
    //----- CObjectMeasure -----//
    //--------------------------//
    class_<CObjectMeasure>("CObjectMeasure", _objectMeasureDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CMeasure&, double, size_t, const std::string&>(_ctor1ObjMeasureDocString, args("self", "measure", "value", "graphicsId", "label")))
        .def(init<const CMeasure&, const std::vector<double>&, size_t, const std::string&>(_ctor2ObjMeasureDocString))
        .def("get_measure_info", &CObjectMeasure::getMeasureInfo, _getMeasureInfoDocString, args("self"))
        .add_property("values", &CObjectMeasure::getValues, &CObjectMeasure::setValues, "Values of the measure")
        .def_readwrite("graphics_id", &CObjectMeasure::m_graphicsId, "Identifier of the associated graphics item")
        .def_readwrite("label", &CObjectMeasure::m_label, "Label of the measure")
    ;

    //--------------------------//
    //----- CBlobMeasureIO -----//
    //--------------------------//
    void (CBlobMeasureIO::*saveBlob)(const std::string&) = &CBlobMeasureIO::save;
    std::string (CBlobMeasureIO::*blobIOToJsonNoOpt)() const = &CBlobMeasureIO::toJson;
    std::string (CBlobMeasureIO::*blobIOToJson)(const std::vector<std::string>&) const = &CBlobMeasureIO::toJson;

    class_<CBlobMeasureIO, bases<CWorkflowTaskIO>, std::shared_ptr<CBlobMeasureIO>>("CBlobMeasureIO", _measureIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorMeasureIODocString, args("self", "name")))
        .def(init<const CBlobMeasureIO&>("Copy constructor"))
        .def("__repr__", &CBlobMeasureIO::repr)
        .def("set_object_measure", &CBlobMeasureIO::setObjectMeasure, _setObjMeasureDocString, args("self", "index", "measure"))
        .def("get_measures", &CBlobMeasureIO::getMeasures, _getMeasuresDocString, args("self"))
        .def("is_data_available", &CBlobMeasureIO::isDataAvailable, _isMeasureDataAvailableDocString, args("self"))
        .def("add_object_measure", &CBlobMeasureIO::addObjectMeasure, _addObjMeasureDocString, args("self", "measure"))
        .def("add_object_measures", &CBlobMeasureIO::addObjectMeasures, _addObjMeasuresDocString, args("self", "measures"))
        .def("clear_data", &CBlobMeasureIO::clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CBlobMeasureIO::load, _blobMeasureIOLoadDocString, args("self", "path"))
        .def("save", saveBlob, _blobMeasureIOSaveDocString, args("self", "path"))
        .def("to_json", blobIOToJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", blobIOToJson, _blobIOToJsonDocString, args("self", "options"))
        .def("from_json", &CBlobMeasureIO::fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //--------------------------//
    //----- CGraphicsInput -----//
    //--------------------------//
    void (CGraphicsInputWrap::*saveGraphicsIn)(const std::string&) = &CGraphicsInputWrap::save;
    std::string (CGraphicsInput::*graphicsInToJsonNoOpt)() const = &CGraphicsInput::toJson;
    std::string (CGraphicsInput::*graphicsInToJson)(const std::vector<std::string>&) const = &CGraphicsInput::toJson;

    class_<CGraphicsInputWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CGraphicsInputWrap>>("CGraphicsInput", _graphicsInputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorGraphicsInDocString, args("self", "name")))
        .def(init<const CGraphicsInput&>("Copy constructor"))
        .def("__repr__", &CGraphicsInput::repr)
        .def("set_items", &CGraphicsInput::setItems, _setItemsDocString, args("self", "items"))
        .def("get_items", &CGraphicsInput::getItems, _getItemsDocString, args("self"))
        .def("is_data_available", &CGraphicsInput::isDataAvailable, &CGraphicsInputWrap::default_isDataAvailable, _isGraphicsDataAvailableDocString, args("self"))
        .def("clear_data", &CGraphicsInput::clearData, &CGraphicsInputWrap::default_clearData, _clearGraphicsDataDocString, args("self"))
        .def("load", &CGraphicsInput::load, &CGraphicsInputWrap::default_load, _graphicsInputLoadDocString, args("self", "path"))
        .def("save", saveGraphicsIn, &CGraphicsInputWrap::default_save, _graphicsInputSaveDocString, args("self", "path"))
        .def("to_json", graphicsInToJsonNoOpt, &CGraphicsInputWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", graphicsInToJson, &CGraphicsInputWrap::default_toJson, _blobIOToJsonDocString, args("self", "options"))
        .def("from_json", &CGraphicsInput::fromJson, &CGraphicsInputWrap::default_fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
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
    std::string (CGraphicsOutput::*graphicsOutToJsonNoOpt)() const = &CGraphicsOutput::toJson;
    std::string (CGraphicsOutput::*graphicsOutToJson)(const std::vector<std::string>&) const = &CGraphicsOutput::toJson;

    class_<CGraphicsOutputWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CGraphicsOutputWrap>>("CGraphicsOutput", _graphicsOutputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorGraphicsOutDocString, args("self", "name")))
        .def(init<const CGraphicsOutput&>("Copy constructor"))
        .def("__repr__", &CGraphicsOutput::repr)
        .def("set_new_layer", &CGraphicsOutput::setNewLayer, _setNewLayerDocString, args("self", "name"))
        .def("set_image_index", &CGraphicsOutput::setImageIndex, _setImageIndexDocString, args("self", "index"))
        .def("set_items", &CGraphicsOutput::setItems, _setItemsDocString, args("self", "items"))
        .def("get_items", &CGraphicsOutput::getItems, _getItemsDocString, args("self"))
        .def("get_image_index", &CGraphicsOutput::getImageIndex, _getImageIndexDocString, args("self"))
        .def("add_item", &CGraphicsOutput::addItem, _addItemDocString, args("self", "item"))
        .def("add_point", addPoint1, _addPoint1DocString, args("self", "point"))
        .def("add_point", addPoint2, _addPoint2DocString, args("self", "point", "properties"))
        .def("add_rectangle", addRectangle1, _addRectangle1DocString, args("self", "x", "y", "width", "height"))
        .def("add_rectangle", addRectangle2, _addRectangle2DocString, args("self", "x", "y", "width", "height", "properties"))
        .def("add_ellipse", addEllipse1, _addEllipse1DocString, args("self", "x", "y", "width", "height"))
        .def("add_ellipse", addEllipse2, _addEllipse2DocString, args("self", "x", "y", "width", "height", "properties"))
        .def("add_polygon", addPolygon1, _addPolygon1DocString, args("self", "points"))
        .def("add_polygon", addPolygon2, _addPolygon2DocString, args("self", "points", "properties"))
        .def("add_polyline", addPolyline1, _addPolyline1DocString, args("self", "points"))
        .def("add_polyline", addPolyline2, _addPolyline2DocString, args("self", "points", "properties"))
        .def("add_complex_polygon", addComplexPolygon1, _addComplexPolygon1DocString, args("self", "outer", "inners"))
        .def("add_complex_polygon", addComplexPolygon2, _addComplexPolygon2DocString, args("self", "outer", "inners", "properties"))
        .def("add_text", addText1, _addText1DocString, args("self", "text", "x", "y"))
        .def("add_text", addText2, _addText2DocString, args("self", "text", "x", "y", "properties"))
        .def("is_data_available", &CGraphicsOutput::isDataAvailable, &CGraphicsOutputWrap::default_isDataAvailable, _isGraphicsDataAvailableDocString, args("self"))
        .def("clear_data", &CGraphicsOutput::clearData, &CGraphicsOutputWrap::default_clearData, _clearGraphicsDataDocString, args("self"))
        .def("load", &CGraphicsOutput::load, &CGraphicsOutputWrap::default_load, _graphicsOutputLoadDocString, args("self", "path"))
        .def("save", saveGraphicsOut, &CGraphicsOutputWrap::default_save, _graphicsOutputSaveDocString, args("self", "path"))
        .def("to_json", graphicsOutToJsonNoOpt, &CGraphicsOutputWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", graphicsOutToJson, &CGraphicsOutputWrap::default_toJson, _blobIOToJsonDocString, args("self", "options"))
        .def("from_json", &CGraphicsOutput::fromJson, &CGraphicsOutputWrap::default_fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
    ;

    //--------------------//
    //----- CImageIO -----//
    //--------------------//
    void (CImageIO::*drawGraphicsIn)(const GraphicsInputPtr&) = &CImageIO::drawGraphics;
    void (CImageIO::*drawGraphicsOut)(const GraphicsOutputPtr&) = &CImageIO::drawGraphics;
    CMat (CImageIO::*getImageWithGraphicsIO)(const WorkflowTaskIOPtr&) = &CImageIO::getImageWithGraphics;
    CMat (CImageIO::*getImageWithMaskIO)(const WorkflowTaskIOPtr&) = &CImageIO::getImageWithMask;
    CMat (CImageIO::*getImageWithMaskAndGraphicsIO)(const WorkflowTaskIOPtr&) = &CImageIO::getImageWithMaskAndGraphics;
    void (CImageIO::*saveImageIO)(const std::string&) = &CImageIO::save;
    std::string (CImageIO::*imgIOToJsonNoOpt)() const = &CImageIO::toJson;
    std::string (CImageIO::*imgIOToJson)(const std::vector<std::string>&) const = &CImageIO::toJson;

    class_<CImageIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CImageIOWrap>>("CImageIO", _imageProcessIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1imageProcessIODocString, args("self", "data_type")))
        .def(init<IODataType, const CMat&>(_ctor2imageProcessIODocString, args("self", "data_type", "image")))
        .def(init<IODataType, const CMat&, const std::string&>(_ctor3imageProcessIODocString, args("self", "data_type", "image", "name")))
        .def(init<IODataType, const std::string&>(_ctor4imageProcessIODocString, args("self", "data_type", "name")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor5imageProcessIODocString, args("self", "data_type", "name", "path")))
        .def(init<const CImageIO&>("Copy constructor"))
        .def("__repr__", &CImageIO::repr)
        .def("set_image", &CImageIO::setImage, _setImageDocString, args("self", "image"))
        .def("set_overlay_mask", &CImageIO::setOverlayMask, _setOverlayMaskDocString, args("self", "mask"))
        .def("set_channel_count", &CImageIO::setChannelCount, _setChannelCountDocString, args("self", "nb"))
        .def("set_current_image_index", &CImageIO::setCurrentImageIndex, _setCurrentImageIndexDocString, args("self", "index"))
        .def("get_channel_count", &CImageIO::getChannelCount, _getChannelCountDocString, args("self"))
        .def("get_current_image_index", &CImageIO::getCurrentImageIndex, _getCurrentImageIndexDocString, args("self"))
        .def("get_data", &CImageIO::getData, _getDataDocString, args("self"))
        .def("get_image", &CImageIO::getImage, &CImageIOWrap::default_getImage, _getImageDocString, args("self"))
        .def("get_image_with_graphics", getImageWithGraphicsIO, _getImageWithGraphicsDocString, args("self", "io"))
        .def("get_image_with_mask", getImageWithMaskIO, _getImageWithMaskDocString, args("self", "io"))
        .def("get_image_with_mask_and_graphics", getImageWithMaskAndGraphicsIO, _getImageWithMaskAndGraphicsDocString, args("self", "io"))
        .def("get_overlay_mask", &CImageIO::getOverlayMask, _getOverlayMaskDocString, args("self"))
        .def("get_unit_element_count", &CImageIO::getUnitElementCount, &CImageIOWrap::default_getUnitElementCount, _getImageUnitElementCountDocString, args("self"))
        .def("is_data_available", &CImageIO::isDataAvailable, &CImageIOWrap::default_isDataAvailable, _isImageDataAvailableDocString, args("self"))
        .def("is_overlay_available", &CImageIO::isOverlayAvailable, _isOverlayAvailableDocString, args("self"))
        .def("clear_data", &CImageIO::clearData, &CImageIOWrap::default_clearData, _clearImageDataDocString, args("self"))
        .def("copy_static_data", &CImageIO::copyStaticData, &CImageIOWrap::default_copyStaticData, _copyImageStaticDataDocString, args("self", "io"))
        .def("draw_graphics", drawGraphicsIn, _drawGraphicsInDocString, args("self", "graphics"))
        .def("draw_graphics", drawGraphicsOut, _drawGraphicsOutDocString, args("self", "graphics"))
        .def("load", &CImageIO::load, &CImageIOWrap::default_load, _imageIOLoadDocString, args("self", "path"))
        .def("save", saveImageIO, &CImageIOWrap::default_save, _imageIOSaveDocString, args("self", "path"))
        .def("to_json", imgIOToJsonNoOpt, &CImageIOWrap::default_toJsonNoOpt, _imageIOToJsonNoOptDocString, args("self"))
        .def("to_json", imgIOToJson, &CImageIOWrap::default_toJson, _imageIOToJsonDocString, args("self", "options"))
        .def("from_json", &CImageIO::fromJson, &CImageIOWrap::default_fromJson, _imageIOFromJsonIDocString, args("self", "jsonStr"))
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
        .def(init<IODataType>(_ctor1VideoProcessIODocString, args("self", "data_type")))
        .def(init<IODataType, const CMat&>(_ctor2VideoProcessIODocString, args("self", "data_type", "frame")))
        .def(init<IODataType, const CMat&, const std::string&>(_ctor3VideoProcessIODocString, args("self", "data_type", "frame", "name")))
        .def(init<IODataType, const std::string&>(_ctor4VideoProcessIODocString, args("self", "data_type", "name")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor5VideoProcessIODocString, args("self", "data_type", "name", "path")))
        .def(init<const CVideoIO&>("Copy constructor"))
        .def("__repr__", &CVideoIO::repr)
        .def("set_video_path", &CVideoIO::setVideoPath, _setVideoPathDocString, args("self", "path"))
        .def("set_video_pos", &CVideoIO::setVideoPos, _setVideoPosDocString, args("self", "position"))
        .def("get_video_frame_count", &CVideoIO::getVideoFrameCount, _getVideoFrameCountDocString, args("self"))
        .def("get_video_images", &CVideoIO::getVideoImages, _getVideoImagesDocString, args("self"))
        .def("get_video_path", &CVideoIO::getVideoPath, _getVideoPathDocString, args("self"))
        .def("get_snapshot", &CVideoIO::getSnapshot, _getSnapshotDocString, args("self", "position"))
        .def("get_current_pos", &CVideoIO::getCurrentPos, _getCurrentPosDocString, args("self"))
        .def("start_video", &CVideoIO::startVideo, _startVideoDocString, args("self", "timeout"))
        .def("stop_video", &CVideoIO::stopVideo, _stopVideoDocString, args("self"))
        .def("start_video_write", &CVideoIO::startVideoWrite, _startVideoWriteDocString, args("self", "width", "height", "frames", "fps", "fourcc", "timeout"))
        .def("stop_video_write", &CVideoIO::stopVideoWrite, _stopVideoWriteDocString, args("self", "timeout"))
        .def("add_video_image", &CVideoIO::addVideoImage, _addVideoImageDocString, args("self", "image"))
        .def("write_image", &CVideoIO::writeImage, _writeImageDocString, args("self", "image"))
        .def("has_video", &CVideoIO::hasVideo, _hasVideoDocString, args("self"))
        .def("get_image", &CVideoIO::getImage, &CVideoIOWrap::default_getImage, _getVideoImageDocString, args("self"))
        .def("get_unit_element_count", &CVideoIO::getUnitElementCount, &CVideoIOWrap::default_getUnitElementCount, _getVideoUnitElementCountDocString, args("self"))
        .def("is_data_available", &CVideoIO::isDataAvailable, &CVideoIOWrap::default_isDataAvailable, _isVideoDataAvailableDocString, args("self"))
        .def("clear_data", &CVideoIO::clearData, &CVideoIOWrap::default_clearData, _clearVideoDataDocString, args("self"))
        .def("copy_static_data", &CVideoIO::copyStaticData, &CVideoIOWrap::default_copyStaticData, _copyStaticDataDerivedDocString, args("self"))
    ;

    //-------------------------//
    //----- CWidgetOutput -----//
    //-------------------------//
    class_<CWidgetOutputWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CWidgetOutputWrap>>("CWidgetOutput", _widgetOutputDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1WidgetOutputDocString, args("self", "data_type")))
        .def(init<IODataType, const std::string&>(_ctor2WidgetOutputDocString, args("self", "data_type", "name")))
        .def("set_widget", &CWidgetOutputWrap::setWidget, _setWidgetDocString, args("self", "widget"))
        .def("is_data_available", &CWidgetOutput::isDataAvailable, &CWidgetOutputWrap::default_isDataAvailable, _isWidgetDataAvailableDocString, args("self"))
        .def("clear_data", &CWidgetOutput::clearData, &CWidgetOutputWrap::default_clearData, _clearWidgetDataDocString, args("self"))
    ;

    //-------------------//
    //----- CPathIO -----//
    //-------------------//
    class_<CPathIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CPathIOWrap>>("CPathIO", _pathIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1PathIODocString, args("self", "data_type")))
        .def(init<IODataType, const std::string&>(_ctor2PathIODocString, args("self", "data_type", "path")))
        .def(init<IODataType, const std::string&, const std::string&>(_ctor3PathIODocString, args("self", "data_type", "path", "name")))
        .def("__repr__", &CPathIO::repr)
        .def("set_path", &CPathIO::setPath, _setPathDocString, args("self", "path"))
        .def("get_path", &CPathIO::getPath, _getPathDocString, args("self"))
        .def("is_data_available", &CPathIO::isDataAvailable, &CPathIOWrap::default_isDataAvailable, _isVideoDataAvailableDocString, args("self"))
        .def("clear_data", &CPathIO::clearData, &CPathIOWrap::default_clearData, _clearDataDocString, args("self"))
    ;

    //----------------------//
    //----- CDatasetIO -----//
    //----------------------//
    std::string (CDatasetIO::*datasetToJsonNoOpt)() const = &CDatasetIO::toJson;
    std::string (CDatasetIO::*datasetToJson)(const std::vector<std::string>&) const = &CDatasetIO::toJson;

    class_<CDatasetIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CDatasetIOWrap>, boost::noncopyable>("CDatasetIO", _datasetIODocString)
        .def(init<>("Default constructor"))
        .def(init<const std::string&>(_ctor1DatasetIODocString, args("self", "name")))
        .def(init<const std::string&, const std::string&>(_ctor2DatasetIODocString, args("self", "name", "source_format")))
        .def("__repr__", &CDatasetIO::repr)
        .def("get_image_paths", &CDatasetIO::getImagePaths, &CDatasetIOWrap::default_getImagePaths, _getImagePathsDocStr)
        .def("get_categories", &CDatasetIO::getCategories, &CDatasetIOWrap::default_getCategories, _getCategoriesDocStr)
        .def("get_category_count", &CDatasetIO::getCategoryCount, &CDatasetIOWrap::default_getCategoryCount, _getCategoryCountDocStr)
        .def("get_mask_path", &CDatasetIO::getMaskPath, &CDatasetIOWrap::default_getMaskPath, _getMaskPathDocStr, args("self", "image_path"))
        .def("get_graphics_annotations", &CDatasetIO::getGraphicsAnnotations, &CDatasetIOWrap::default_getGraphicsAnnotations, _getGraphicsAnnotationsDocStr, args("self", "image_path"))
        .def("get_source_format", &CDatasetIO::getSourceFormat, _getSourceFormatDocStr)
        .def("is_data_available", &CDatasetIO::isDataAvailable, &CDatasetIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clear_data", &CDatasetIO::clearData, &CDatasetIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("save", &CDatasetIO::save, &CDatasetIOWrap::default_save, _saveDocStr)
        .def("load", &CDatasetIO::load, &CDatasetIOWrap::default_load, _loadDocStr)
        .def("to_json", datasetToJsonNoOpt, &CDatasetIOWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", datasetToJson, &CDatasetIOWrap::default_toJson, _datasetIOToJsonDocStr, args("self", "options"))
        .def("from_json", &CDatasetIO::fromJson, &CDatasetIOWrap::default_fromJson, _datasetIOFromJsonDocStr, args("self", "json_str"))
    ;

    //--------------------//
    //----- CArrayIO -----//
    //--------------------//
    class_<CArrayIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CArrayIOWrap>>("CArrayIO", _arrayIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1ArrayIODocString, args("self", "name")))
        .def(init<const CMat&, const std::string&>(_ctor2ArrayIODocString, args("self", "array", "name")))
        .def(init<const CArrayIO&>("Copy constructor"))
        .def("__repr__", &CArrayIO::repr)
        .def("set_array", &CArrayIO::setArray, _setArrayDocString, args("self", "array"))
        .def("get_array", &CArrayIO::getArray, _getArrayDocString, args("self"))
        .def("get_unit_element_count", &CArrayIO::getUnitElementCount, &CArrayIOWrap::default_getUnitElementCount, _getArrayUnitElementCountDocString, args("self"))
        .def("is_data_available", &CArrayIO::isDataAvailable, &CArrayIOWrap::default_isDataAvailable, _isArrayDataAvailableDocString, args("self"))
        .def("clear_data", &CArrayIO::clearData, &CArrayIOWrap::default_clearData, _clearArrayDataDocString, args("self"))
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
        .def("__repr__", &CObjectDetection::repr)
        .def(self_ns::str(self_ns::self))
    ;

    void (CObjectDetectionIO::*addObjectBox1)(int, const std::string&, double, double, double, double, double, const CColor&) = &CObjectDetectionIO::addObject;
    void (CObjectDetectionIO::*addObjectRotateBox1)(int, const std::string&, double, double, double, double, double, double, const CColor&) = &CObjectDetectionIO::addObject;
    std::string (CObjectDetectionIO::*objDetectToJsonNoOpt)() const = &CObjectDetectionIO::toJson;
    std::string (CObjectDetectionIO::*objDetectToJson)(const std::vector<std::string>&) const = &CObjectDetectionIO::toJson;

    class_<CObjectDetectionIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CObjectDetectionIOWrap>>("CObjectDetectionIO", _objDetectionIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CObjectDetectionIO&>("Copy constructor"))
        .def("__repr__", &CObjectDetectionIO::repr)
        .def("get_object_count", &CObjectDetectionIO::getObjectCount, _getObjectCountDocString, args("self"))
        .def("get_object", &CObjectDetectionIO::getObject, _getObjectDocString, args("self", "index"))
        .def("get_objects", &CObjectDetectionIO::getObjects, _getObjectsDocString, args("self"))
        .def("get_graphics_io", &CObjectDetectionIO::getGraphicsIO, _getGraphicsIODocString, args("self"))
        .def("get_blob_measure_io", &CObjectDetectionIO::getBlobMeasureIO, _getBlobMeasureIODocString, args("self"))
        .def("is_data_available", &CObjectDetectionIO::isDataAvailable, &CObjectDetectionIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("init", &CObjectDetectionIO::init, _initObjDetectIODocString, args("self", "task_name", "ref_image_index"))
        .def("add_object", addObjectBox1, _addObjectDocString, args("self", "id", "label", "confidence", "box_x", "box_y", "box_width", "box_height", "color"))
        .def("add_object", addObjectRotateBox1, _addObject2DocString, args("self", "id", "label", "confidence", "cx", "cy", "width", "height", "angle", "color"))
        .def("clear_data", &CObjectDetectionIO::clearData, &CObjectDetectionIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CObjectDetectionIO::load, &CObjectDetectionIOWrap::default_load, _objDetectLoadDocString, args("self", "path"))
        .def("save", &CObjectDetectionIO::save, &CObjectDetectionIOWrap::default_save, _objDetectSaveDocString, args("self", "path"))
        .def("to_json", objDetectToJsonNoOpt, &CObjectDetectionIOWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", objDetectToJson, &CObjectDetectionIOWrap::default_toJson, _objDetectToJsonDocString, args("self", "options"))
        .def("from_json", &CObjectDetectionIO::fromJson, &CObjectDetectionIOWrap::default_fromJson, _objDetectFromJsonDocString, args("self", "json_str"))
    ;

    //-----------------------------------//
    //----- CInstanceSegmentationIO -----//
    //-----------------------------------//
    class_<CInstanceSegmentation>("CInstanceSegmentation", _instanceSegDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("id", &CInstanceSegmentation::getId, &CInstanceSegmentation::setId, "Object ID (int)")
        .add_property("type", &CInstanceSegmentation::getType, &CInstanceSegmentation::setType, "Object type (int 0=THING or 1=STUFF)")
        .add_property("class_index", &CInstanceSegmentation::getClassIndex, &CInstanceSegmentation::setClassIndex, "Object class index (int)")
        .add_property("label", &CInstanceSegmentation::getLabel, &CInstanceSegmentation::setLabel, "Object label (str)")
        .add_property("confidence", &CInstanceSegmentation::getConfidence, &CInstanceSegmentation::setConfidence, "Prediction confidence (double)")
        .add_property("box", &CInstanceSegmentation::getBox, &CInstanceSegmentation::setBox, "Object bounding box [x, y, width, height]")
        .add_property("mask", &CInstanceSegmentation::getMask, &CInstanceSegmentation::setMask, "Object mask (numpy array)")
        .add_property("color", &CInstanceSegmentation::getColor, &CInstanceSegmentation::setColor, "Object display color [r, g, b, a]")
        .def_readonly("polygons", &CInstanceSegmentation::m_polygons)
        .def("__repr__", &CInstanceSegmentation::repr)
        .def(self_ns::str(self_ns::self))
    ;

    std::string (CInstanceSegIO::*instSegToJsonNoOpt)() const = &CInstanceSegIO::toJson;
    std::string (CInstanceSegIO::*instSegToJson)(const std::vector<std::string>&) const = &CInstanceSegIO::toJson;

    class_<CInstanceSegIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CInstanceSegIOWrap>>("CInstanceSegmentationIO", _instanceSegIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CInstanceSegIO&>("Copy constructor"))
        .def("__repr__", &CInstanceSegIO::repr)
        .def("get_object_count", &CInstanceSegIO::getObjectCount, _getInstanceCountDocString, args("self"))
        .def("get_object", &CInstanceSegIO::getObject, _getInstanceDocString, args("self", "index"))
        .def("get_objects", &CInstanceSegIO::getObjects, _getInstancesDocString, args("self"))
        .def("get_graphics_io", &CInstanceSegIO::getGraphicsIO, _getGraphicsIODocString, args("self"))
        .def("get_blob_measure_io", &CInstanceSegIO::getBlobMeasureIO, _getBlobMeasureIODocString, args("self"))
        .def("get_merge_mask", &CInstanceSegIO::getMergeMask, _getMergeMaskDocString, args("self"))
        .def("is_data_available", &CInstanceSegIO::isDataAvailable, &CInstanceSegIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("init", &CInstanceSegIO::init, _initInstanceSegIODocString, args("self", "task_name", "ref_image_index", "width", "heigh"))
        .def("add_object", &CInstanceSegIO::addObject, _addInstanceDocString, args("self", "id", "type", "class_index", "label", "confidence", "box_x", "box_y", "box_width", "box_height", "mask", "color"))
        .def("clear_data", &CInstanceSegIO::clearData, &CInstanceSegIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CInstanceSegIO::load, &CInstanceSegIOWrap::default_load, _instanceSegLoadDocString, args("self", "path"))
        .def("save", &CInstanceSegIO::save, &CInstanceSegIOWrap::default_save, _instanceSegSaveDocString, args("self", "path"))
        .def("to_json", instSegToJsonNoOpt, &CInstanceSegIOWrap::default_toJsonNoOpt, _imageIOToJsonNoOptDocString, args("self"))
        .def("to_json", instSegToJson, &CInstanceSegIOWrap::default_toJson, _instanceSegToJsonDocString, args("self", "options"))
        .def("from_json", &CInstanceSegIO::fromJson, &CInstanceSegIOWrap::default_fromJson, _instanceSegFromJsonDocString, args("self", "json_str"))
    ;

    //--------------------------//
    //----- CSemanticSegIO -----//
    //--------------------------//
    std::string (CSemanticSegIO::*semSegToJsonNoOpt)() const = &CSemanticSegIO::toJson;
    std::string (CSemanticSegIO::*semSegToJson)(const std::vector<std::string>&) const = &CSemanticSegIO::toJson;

    class_<CSemanticSegIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CSemanticSegIOWrap>>("CSemanticSegmentationIO", _semanticSegIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CSemanticSegIO&>("Copy constructor"))
        .def("__repr__", &CSemanticSegIO::repr)
        .def("get_legend", &CSemanticSegIO::getLegend, _getLegendDocString, args("self"))
        .def("get_mask", &CSemanticSegIO::getMask, _getMaskDocString, args("self"))
        .def("get_polygons", &CSemanticSegIO::getPolygons, _getPolygonsDocString, args("self"))
        .def("get_class_names", &CSemanticSegIO::getClassNames, _getClassNamesDocString, args("self"))
        .def("get_colors", &CSemanticSegIO::getColors, _getColorsDocString, args("self"))
        .def("set_mask", &CSemanticSegIO::setMask, _setMaskDocString, args("self", "mask"))
        .def("set_class_names", &CSemanticSegIO::setClassNames, _setClassNamesDocString, args("self", "names"))
        .def("set_class_colors", &CSemanticSegIO::setClassColors, _setClassColorsDocString, args("self", "colors"))
        .def("is_data_available", &CSemanticSegIO::isDataAvailable, &CSemanticSegIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("clear_data", &CSemanticSegIO::clearData, &CSemanticSegIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CSemanticSegIO::load, &CSemanticSegIOWrap::default_load, _instanceSegLoadDocString, args("self", "path"))
        .def("save", &CSemanticSegIO::save, &CSemanticSegIOWrap::default_save, _instanceSegSaveDocString, args("self", "path"))
        .def("to_json", semSegToJsonNoOpt, &CSemanticSegIOWrap::default_toJsonNoOpt, _imageIOToJsonNoOptDocString, args("self"))
        .def("to_json", semSegToJson, &CSemanticSegIOWrap::default_toJson, _instanceSegToJsonDocString, args("self", "options"))
        .def("from_json", &CSemanticSegIO::fromJson, &CSemanticSegIOWrap::default_fromJson, _instanceSegFromJsonDocString, args("self", "jsonStr"))
    ;

    //------------------------//
    //----- CKeypointsIO -----//
    //------------------------//
    class_<CObjectKeypoints>("CObjectKeypoints", _objkeyptsDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("id", &CObjectKeypoints::getId, &CObjectKeypoints::setId, "Object ID (int)")
        .add_property("label", &CObjectKeypoints::getLabel, &CObjectKeypoints::setLabel, "Object label (str)")
        .add_property("confidence", &CObjectKeypoints::getConfidence, &CObjectKeypoints::setConfidence, "Prediction confidence (double)")
        .add_property("box", &CObjectKeypoints::getBox, &CObjectKeypoints::setBox, "Object bounding box [x, y, width, height]")
        .add_property("color", &CObjectKeypoints::getColor, &CObjectKeypoints::setColor, "Object display color [r, g, b, a]")
        .add_property("points", &CObjectKeypoints::getKeypoints, &CObjectKeypoints::setKeypoints, "Keypoints list (:py:class:`~ikomia.core.pycore.CPointF`)")
        .def("__repr__", &CObjectKeypoints::repr)
        .def(self_ns::str(self_ns::self))
    ;

    class_<CKeypointLink>("CKeypointLink", _keyptLinkDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("start_point_index", &CKeypointLink::getStartPointIndex, &CKeypointLink::setStartPointIndex, "Starting point index (int)")
        .add_property("end_point_index", &CKeypointLink::getEndPointIndex, &CKeypointLink::setEndPointIndex, "Ending point index (int)")
        .add_property("label", &CKeypointLink::getLabel, &CKeypointLink::setLabel, "Link label (str)")
        .add_property("color", &CKeypointLink::getColor, &CKeypointLink::setColor, "Link color [r, g, b]")
        .def("__repr__", &CKeypointLink::repr)
        .def(self_ns::str(self_ns::self))
    ;

    std::string (CKeypointsIO::*keyptsToJsonNoOpt)() const = &CKeypointsIO::toJson;
    std::string (CKeypointsIO::*keyptsToJson)(const std::vector<std::string>&) const = &CKeypointsIO::toJson;

    class_<CKeyptsIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CKeyptsIOWrap>>("CKeypointsIO", _keyptsIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CKeypointsIO&>("Copy constructor"))
        .def("__repr__", &CKeypointsIO::repr)
        .def("add_object", &CKeypointsIO::addObject, _keyptsAddObjDocString, args("self", "id", "label", "confidence", "box_x", "box_y", "box_width", "box_height", "keypoints", "color"))
        .def("clear_data", &CKeypointsIO::clearData, &CKeyptsIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("get_object_count", &CKeypointsIO::getObjectCount, _keyptsGetObjCountDocString, args("self"))
        .def("get_object", &CKeypointsIO::getObject, _keyptsGetObjDocString, args("self", "index"))
        .def("get_objects", &CKeypointsIO::getObjects, _keyptsGetObjectsDocString, args("self"))
        .def("get_graphics_io", &CKeypointsIO::getGraphicsIO, _keyptsGetGraphicsIODocString, args("self"))
        .def("get_blob_measure_io", &CKeypointsIO::getBlobMeasureIO, _getBlobMeasureIODocString, args("self"))
        .def("get_data_string_io", &CKeypointsIO::getDataStringIO, _keyptsGetDataStringIODocString, args("self"))
        .def("get_keypoint_links", &CKeypointsIO::getKeypointLinks, _getKeyptsLinksDocString, args("self"))
        .def("get_keypoint_names", &CKeypointsIO::getKeypointNames, _getKeyptsNamesDocString, args("self"))
        .def("init", &CKeypointsIO::init, _keyptsInitDocString, args("self", "task_name", "ref_image_index"))
        .def("is_data_available", &CKeypointsIO::isDataAvailable, &CKeyptsIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("set_keypoint_names", &CKeypointsIO::setKeypointNames, _setKeyptNamesDocString, args("self", "names"))
        .def("set_keypoint_links", &CKeypointsIO::setKeypointLinks, _setKeyptLinksDocString, args("self", "links"))
        .def("load", &CKeypointsIO::load, &CKeyptsIOWrap::default_load, _keyptsLoadDocString, args("self", "path"))
        .def("save", &CKeypointsIO::save, &CKeyptsIOWrap::default_save, _keyptsSaveDocString, args("self", "path"))
        .def("to_json", keyptsToJsonNoOpt, &CKeyptsIOWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", keyptsToJson, &CKeyptsIOWrap::default_toJson, _objDetectToJsonDocString, args("self", "options"))
        .def("from_json", &CKeypointsIO::fromJson, &CKeyptsIOWrap::default_fromJson, _objDetectFromJsonDocString, args("self", "json_str"))
    ;

    //-------------------//
    //----- CTextIO -----//
    //-------------------//
    class_<CTextField>("CTextField", _textFieldDocString)
        .def(init<>("Default constructor", args("self")))
        .add_property("id", &CTextField::getId, &CTextField::setId, "Text field ID (int)")
        .add_property("label", &CTextField::getLabel, &CTextField::setLabel, "Text field label (str)")
        .add_property("text", &CTextField::getText, &CTextField::setText, "Text field value (str)")
        .add_property("confidence", &CTextField::getConfidence, &CTextField::setConfidence, "Prediction confidence (double)")
        .add_property("polygon", &CTextField::getPolygon, &CTextField::setPolygon, "Text field polygon: list of :py:class:`~ikomia.core.pycore.CPointF`")
        .add_property("color", &CTextField::getColor, &CTextField::setColor, "Text field display color [r, g, b, a]")
        .def("__repr__", &CTextField::repr)
        .def(self_ns::str(self_ns::self))
    ;

    void (CTextIO::*addFieldBox)(int, const std::string&, const std::string&, double, double, double, double, double, const CColor&) = &CTextIO::addTextField;
    void (CTextIO::*addFieldPoly)(int, const std::string&, const std::string&, double, const PolygonF&, const CColor&) = &CTextIO::addTextField;
    std::string (CTextIO::*textToJsonNoOpt)() const = &CTextIO::toJson;
    std::string (CTextIO::*textToJson)(const std::vector<std::string>&) const = &CTextIO::toJson;

    class_<CTextIOWrap, bases<CWorkflowTaskIO>, std::shared_ptr<CTextIOWrap>>("CTextIO", _textIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CTextIO&>("Copy constructor"))
        .def("__repr__", &CTextIO::repr)
        .def("get_text_field_count", &CTextIO::getTextFieldCount, _getTextFieldCountDocString, args("self"))
        .def("get_text_field", &CTextIO::getTextField, _getTextFieldDocString, args("self", "index"))
        .def("get_text_fields", &CTextIO::getTextFields, _getTextFieldsDocString, args("self"))
        .def("get_graphics_io", &CTextIO::getGraphicsIO, _textIOGetGraphicsIODocString, args("self"))
        .def("get_data_string_io", &CTextIO::getDataStringIO, _textIOGetDataStringIODocString, args("self"))
        .def("is_data_available", &CTextIO::isDataAvailable, &CTextIOWrap::default_isDataAvailable, _isDataAvailableDerivedDocString, args("self"))
        .def("init", &CTextIO::init, _initObjDetectIODocString, args("self", "task_name", "ref_image_index"))
        .def("finalize", &CTextIO::finalize, _textIOFinalizeIODocString, args("self"))
        .def("add_text_field", addFieldBox, _addTextFieldBoxDocString, args("self", "id", "label", "text", "confidence", "box_x", "box_y", "box_width", "box_height", "color"))
        .def("add_text_field", addFieldPoly, _addTextFieldPolyDocString, args("self", "id", "label", "text", "confidence", "polygon", "color"))
        .def("clear_data", &CTextIO::clearData, &CTextIOWrap::default_clearData, _clearDataDerivedDocString, args("self"))
        .def("load", &CTextIO::load, &CTextIOWrap::default_load, _textLoadDocString, args("self", "path"))
        .def("save", &CTextIO::save, &CTextIOWrap::default_save, _textSaveDocString, args("self", "path"))
        .def("to_json", textToJsonNoOpt, &CTextIOWrap::default_toJsonNoOpt, _blobIOToJsonNoOptDocString, args("self"))
        .def("to_json", textToJson, &CTextIOWrap::default_toJson, _objDetectToJsonDocString, args("self", "options"))
        .def("from_json", &CTextIO::fromJson, &CTextIOWrap::default_fromJson, _objDetectFromJsonDocString, args("self", "json_str"))
    ;

    //------------------------//
    //----- C2dImageTask -----//
    //------------------------//
    class_<C2dImageTaskWrap, bases<CWorkflowTask>, std::shared_ptr<C2dImageTaskWrap>>("C2dImageTask", _imageProcess2dDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<bool>(_ctor1ImageProcess2dDocString, args("self", "has_graphics_input")))
        .def(init<const std::string&>(_ctor2ImageProcess2dDocString, args("self", "name")))
        .def(init<const std::string&, bool>(_ctor3ImageProcess2dDocString, args("self", "name", "has_graphics_input")))
        .def(init<const C2dImageTaskWrap&>("Copy constructor"))
        .def("__repr__", &C2dImageTask::repr, &C2dImageTaskWrap::default_repr)
        .def("set_active", &C2dImageTask::setActive, &C2dImageTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("set_output_color_map", &C2dImageTask::setOutputColorMap, _setOutputColorMapDocString, args("self", "index", "mask_index", "colors", "reserve_zero"))
        .def("update_static_outputs", &C2dImageTask::updateStaticOutputs, &C2dImageTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &C2dImageTask::beginTaskRun, &C2dImageTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &C2dImageTask::endTaskRun, &C2dImageTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &C2dImageTask::graphicsChanged, &C2dImageTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &C2dImageTask::globalInputChanged, &C2dImageTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("create_input_graphics_mask", &C2dImageTask::createInputGraphicsMask, _createInputGraphicsMaskDocString, args("self", "index", "width", "height"))
        .def("create_graphics_mask", &C2dImageTask::createGraphicsMask, _createGraphicsMaskDocString, args("self", "width", "height", "graphics"))
        .def("apply_graphics_mask", &C2dImageTaskWrap::applyGraphicsMask, _applyGraphicsMaskDocString, args("self", "origin", "processed", "index"))
        .def("apply_graphics_mask_to_binary", &C2dImageTaskWrap::applyGraphicsMaskToBinary, _applyGraphicsMaskToBinaryDocString, args("self", "origin", "processed", "index"))
        .def("get_progress_steps", &C2dImageTask::getProgressSteps, &C2dImageTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("get_graphics_mask", &C2dImageTask::getGraphicsMask, _getGraphicsMaskDocString, args("self", "index"))
        .def("is_mask_available", &C2dImageTask::isMaskAvailable, _isMaskAvailableDocString, args("self", "index"))
        .def("run", &C2dImageTask::run, &C2dImageTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &C2dImageTask::stop, &C2dImageTaskWrap::default_stop, _stopDocString, args("self"))
        .def("forward_input_image", &C2dImageTask::forwardInputImage, _forwardInputImageDocString, args("self", "input_index", "output_index"))
        .def("emit_add_sub_progress_steps", &C2dImageTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &C2dImageTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &C2dImageTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &C2dImageTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &C2dImageTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &C2dImageTask::executeActions, &C2dImageTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
    ;

    //-----------------------------------//
    //----- C2dImageInteractiveTask -----//
    //-----------------------------------//
    class_<C2dImageInteractiveTaskWrap, bases<C2dImageTask>, std::shared_ptr<C2dImageInteractiveTaskWrap>>("C2dImageInteractiveTask", _interactiveImageProcess2d)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorInteractiveImageProcessDocString, args("self", "name")))
        .def(init<const C2dImageInteractiveTask&>("Copy constructor"))
        .def("__repr__", &C2dImageInteractiveTask::repr, &C2dImageInteractiveTaskWrap::default_repr)
        .def("set_active", &C2dImageInteractiveTask::setActive, &C2dImageInteractiveTaskWrap::default_setActive, _setActiveInteractiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &C2dImageInteractiveTask::updateStaticOutputs, &C2dImageInteractiveTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &C2dImageInteractiveTask::beginTaskRun, &C2dImageInteractiveTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &C2dImageInteractiveTask::endTaskRun, &C2dImageInteractiveTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &C2dImageInteractiveTask::graphicsChanged, &C2dImageInteractiveTaskWrap::default_graphicsChanged, _graphicsChangedInteractiveDocString, args("self"))
        .def("global_input_changed", &C2dImageInteractiveTask::globalInputChanged, &C2dImageInteractiveTaskWrap::default_globalInputChanged, _globalInputChangedInteractiveDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &C2dImageInteractiveTask::getProgressSteps, &C2dImageInteractiveTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("get_interaction_mask", &C2dImageInteractiveTask::getInteractionMask, _getInteractionMaskDocString, args("self"))
        .def("get_blobs", &C2dImageInteractiveTask::getBlobs, _getBlobsDocString, args("self"))
        .def("create_interaction_mask", &C2dImageInteractiveTask::createInteractionMask, _createInteractionMaskDocString, args("self", "width", "height"))
        .def("compute_blobs", &C2dImageInteractiveTask::computeBlobs, _computeBlobsDocString, args("self"))
        .def("clear_interaction_layer", &C2dImageInteractiveTask::clearInteractionLayer, _clearInteractionLayerDocString, args("self"))
        .def("run", &C2dImageInteractiveTask::run, &C2dImageInteractiveTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &C2dImageInteractiveTask::stop, &C2dImageInteractiveTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &C2dImageInteractiveTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &C2dImageInteractiveTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &C2dImageInteractiveTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &C2dImageInteractiveTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &C2dImageInteractiveTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &C2dImageInteractiveTask::executeActions, &C2dImageInteractiveTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
    ;

    //----------------------//
    //----- CVideoTask -----//
    //----------------------//
    class_<CVideoTaskWrap, bases<C2dImageTask>, std::shared_ptr<CVideoTaskWrap>>("CVideoTask", _videoProcessDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoProcessDocString, args("self", "name")))
        .def("__repr__", &CVideoTask::repr, &CVideoTaskWrap::default_repr)
        .def("set_active", &CVideoTask::setActive, &CVideoTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CVideoTask::updateStaticOutputs, &CVideoTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CVideoTask::beginTaskRun, &CVideoTaskWrap::default_beginTaskRun, _beginTaskRunVideoDocString, args("self"))
        .def("end_task_run", &CVideoTask::endTaskRun, &CVideoTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("notify_video_start", &CVideoTask::notifyVideoStart, &CVideoTaskWrap::default_notifyVideoStart, _notifyVideoStartDocString, args("self", "frame_count"))
        .def("notify_video_end", &CVideoTask::notifyVideoEnd, &CVideoTaskWrap::default_notifyVideoEnd, _notifyVideoEndDocString, args("self"))
        .def("graphics_changed", &CVideoTask::graphicsChanged, &CVideoTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CVideoTask::globalInputChanged, &CVideoTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CVideoTask::getProgressSteps, &CVideoTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoTask::run, &CVideoTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoTask::stop, &CVideoTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CVideoTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CVideoTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CVideoTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CVideoTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CVideoTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CVideoTask::executeActions, &CVideoTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
    ;

    //------------------------//
    //----- CVideoOFTask -----//
    //------------------------//
    class_<CVideoOFTaskWrap, bases<CVideoTask>, std::shared_ptr<CVideoOFTaskWrap>>("CVideoOFTask", _videoProcessOFDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoProcessOFDocString, args("self", "name")))
        .def("set_active", &CVideoOFTask::setActive, &CVideoOFTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CVideoOFTask::updateStaticOutputs, &CVideoOFTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CVideoOFTask::beginTaskRun, &CVideoOFTaskWrap::default_beginTaskRun, _beginTaskRunVideoOFDocString, args("self"))
        .def("end_task_run", &CVideoOFTask::endTaskRun, &CVideoOFTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CVideoOFTask::graphicsChanged, &CVideoOFTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CVideoOFTask::globalInputChanged, &CVideoOFTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self" "is_new_sequence"))
        .def("get_progress_steps", &CVideoOFTask::getProgressSteps, &CVideoOFTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoOFTask::run, &CVideoOFTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoOFTask::stop, &CVideoOFTaskWrap::default_stop, _stopDocString, args("self"))
        .def("notify_video_start", &CVideoOFTask::notifyVideoStart, &CVideoOFTaskWrap::default_notifyVideoStart, _notifyVideoStartDocString, args("self", "frame_count"))
        .def("notify_video_end", &CVideoOFTask::notifyVideoEnd, &CVideoOFTaskWrap::default_notifyVideoEnd, _notifyVideoEndDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CVideoOFTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CVideoOFTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CVideoOFTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CVideoOFTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CVideoOFTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("draw_opt_flow_map", &CVideoOFTaskWrap::drawOptFlowMapWrap, _drawOptFlowMapDocString, args("self", "flow", "vectors", "step"))
        .def("flow_to_display", &CVideoOFTask::flowToDisplay, _flowToDisplayDocString, args("self", "flow"))
        .def("execute_actions", &CVideoOFTask::executeActions, &CVideoOFTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
    ;

    //------------------------------//
    //----- CVideoTrackingTask -----//
    //------------------------------//
    class_<CVideoTrackingTaskWrap, bases<CVideoTask>, std::shared_ptr<CVideoTrackingTaskWrap>>("CVideoTrackingTask", _videoProcessTrackingDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorVideoTrackingDocString, args("self", "name")))
        .def("set_active", &CVideoTrackingTask::setActive, &CVideoTrackingTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CVideoTrackingTask::updateStaticOutputs, &CVideoTrackingTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CVideoTrackingTask::beginTaskRun, &CVideoTrackingTaskWrap::default_beginTaskRun, _beginTaskRunVideoOFDocString, args("self"))
        .def("end_task_run", &CVideoTrackingTask::endTaskRun, &CVideoTrackingTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CVideoTrackingTask::graphicsChanged, &CVideoTrackingTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CVideoTrackingTask::globalInputChanged, &CVideoTrackingTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CVideoTrackingTask::getProgressSteps, &CVideoTrackingTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CVideoTrackingTask::run, &CVideoTrackingTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CVideoTrackingTask::stop, &CVideoTrackingTaskWrap::default_stop, _stopDocString, args("self"))
        .def("notify_video_start", &CVideoTrackingTask::notifyVideoStart, &CVideoTrackingTaskWrap::default_notifyVideoStart, _notifyVideoStartDocString, args("self", "frame_count"))
        .def("notify_video_end", &CVideoTrackingTask::notifyVideoEnd, &CVideoTrackingTaskWrap::default_notifyVideoEnd, _notifyVideoEndDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CVideoTrackingTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CVideoTrackingTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CVideoTrackingTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CVideoTrackingTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CVideoTrackingTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("set_roi_to_track", &CVideoTrackingTask::setRoiToTrack, _setRoiToTrackDocString, args("self"))
        .def("manage_outputs", &CVideoTrackingTask::manageOutputs, _manageOutputsDocString, args("self"))
        .def("execute_actions", &CVideoTrackingTask::executeActions, &CVideoTrackingTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
    ;

    //-------------------------//
    //----- CDnnTrainTask -----//
    //-------------------------//
    class_<CDnnTrainTaskWrap, bases<CWorkflowTask>, std::shared_ptr<CDnnTrainTaskWrap>>("CDnnTrainTask", _dnnTrainProcessDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1DnnTrainProcessDocString, args("self", "name")))
        .def(init<const std::string&, const std::shared_ptr<CWorkflowTaskParam>&>(_ctor2DnnTrainProcessDocString, args("self", "name", "param")))
        .def("__repr__", &CDnnTrainTask::repr, &CDnnTrainTaskWrap::default_repr)
        .def("begin_task_run", &CDnnTrainTask::beginTaskRun, &CDnnTrainTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CDnnTrainTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CDnnTrainTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_output_changed", &CDnnTrainTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CDnnTrainTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("enable_mlflow", &CDnnTrainTask::enableMlflow, _enableMlflowDocString, args("self", "enable"))
        .def("enable_tensorboard", &CDnnTrainTask::enableTensorboard, _enableTensorboardDocString, args("self", "enable"))
        .def("end_task_run", &CDnnTrainTask::endTaskRun, &CDnnTrainTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("execute_actions", &CDnnTrainTask::executeActions, &CDnnTrainTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("get_progress_steps", &CDnnTrainTask::getProgressSteps, &CDnnTrainTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CDnnTrainTask::run, &CDnnTrainTaskWrap::default_run, _runDocString, args("self"))
        .def("set_active", &CDnnTrainTask::setActive, &CDnnTrainTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("stop", &CDnnTrainTask::stop, &CDnnTrainTaskWrap::default_stop, _stopDocString, args("self"))
    ;

    //-------------------------------//
    //----- CClassificationTask -----//
    //-------------------------------//
    class_<CClassifTaskWrap, bases<C2dImageTask>, std::shared_ptr<CClassifTaskWrap>>("CClassificationTask", _classifTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorClassifDocString, args("self", "name")))
        .def("__repr__", &CClassificationTask::repr, &CClassifTaskWrap::default_repr)
        .def("set_active", &CClassificationTask::setActive, &CClassifTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CClassificationTask::updateStaticOutputs, &CClassifTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CClassificationTask::beginTaskRun, &CClassifTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CClassificationTask::endTaskRun, &CClassifTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CClassificationTask::graphicsChanged, &CClassifTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CClassificationTask::globalInputChanged, &CClassifTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CClassificationTask::getProgressSteps, &CClassifTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CClassificationTask::run, &CClassifTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CClassificationTask::stop, &CClassifTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CClassifTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CClassifTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CClassifTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CClassifTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CClassifTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CClassificationTask::executeActions, &CClassifTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("add_object", &CClassificationTask::addObject, _classifAddObjectDocString, args("self", "graphics_item", "class_index", "confidence"))
        .def("get_names", &CClassificationTask::getNames, _classifGetNamesDocString, args("self"))
        .def("get_input_objects", &CClassificationTask::getInputObjects, _classifGetInputObjectsDocString, args("self"))
        .def("get_object_sub_image", &CClassificationTask::getObjectSubImage, _classifGetObjectSubImageDocString, args("self", "item"))
        .def("get_objects_results", &CClassificationTask::getObjectsResults, _classifGetObjectsResultsDocString, args("self"))
        .def("get_whole_image_results", &CClassificationTask::getWholeImageResults, _classifGetWholeImageResultsDocString, args("self"))
        .def("get_image_with_graphics", &CClassificationTask::getImageWithGraphics, _classifGetImgWithGraphicsDocString, args("self"))
        .def("is_whole_image_classification", &CClassificationTask::isWholeImageClassification, _classifIsWholeImageDocString, args("self"))
        .def("read_class_names", &CClassificationTask::readClassNames, _classifReadClassNamesDocString, args("self", "path"))
        .def("set_colors", &CClassificationTask::setColors, _classifSetColorsDocString, args("self", "colors"))
        .def("set_names", &CClassificationTask::setNames, _classifSetNamesDocString, args("self", "names"))
        .def("set_whole_image_results", &CClassificationTask::setWholeImageResults, _classifSetWholeImageResultsDocString, args("self", "names", "confidences"))
    ;

    //--------------------------------//
    //----- CObjectDetectionTask -----//
    //--------------------------------//
    void (CObjectDetectionTask::*addObjectBox2)(int, int, double, double, double, double, double) = &CObjectDetectionTask::addObject;
    void (CObjectDetectionTask::*addObjectRotateBox2)(int, int, double, double, double, double, double, double) = &CObjectDetectionTask::addObject;

    class_<CObjDetectTaskWrap, bases<C2dImageTask>, std::shared_ptr<CObjDetectTaskWrap>>("CObjectDetectionTask", _objDetTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorObjDetectDocString, args("self", "name")))
        .def("__repr__", &CObjectDetectionTask::repr, &CObjDetectTaskWrap::default_repr)
        .def("set_active", &CObjectDetectionTask::setActive, &CObjDetectTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CObjectDetectionTask::updateStaticOutputs, &CObjDetectTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CObjectDetectionTask::beginTaskRun, &CObjDetectTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CObjectDetectionTask::endTaskRun, &CObjDetectTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CObjectDetectionTask::graphicsChanged, &CObjDetectTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CObjectDetectionTask::globalInputChanged, &CObjDetectTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CObjectDetectionTask::getProgressSteps, &CObjDetectTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CObjectDetectionTask::run, &CObjDetectTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CObjectDetectionTask::stop, &CObjDetectTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CObjDetectTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CObjDetectTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CObjDetectTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CObjDetectTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CObjDetectTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CObjectDetectionTask::executeActions, &CObjDetectTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("add_object", addObjectBox2, _objDetAddObject1DocString, args("self", "id", "class_index", "confidence", "x", "y", "width", "height"))
        .def("add_object", addObjectRotateBox2, _objDetAddObject2DocString, args("self", "id", "class_index", "confidence", "cx", "cy", "width", "height", "angle"))
        .def("get_names", &CObjectDetectionTask::getNames, _classifGetNamesDocString, args("self"))
        .def("get_results", &CObjectDetectionTask::getResults, _objDetectGetResultsDocString, args("self"))
        .def("get_image_with_graphics", &CObjectDetectionTask::getImageWithGraphics, _classifGetImgWithGraphicsDocString, args("self"))
        .def("read_class_names", &CObjectDetectionTask::readClassNames, _classifReadClassNamesDocString, args("self", "path"))
        .def("set_colors", &CObjectDetectionTask::setColors, _classifSetColorsDocString, args("self", "colors"))
        .def("set_names", &CObjectDetectionTask::setNames, _classifSetNamesDocString, args("self", "names"))
    ;

    //-------------------------------------//
    //----- CSemanticSegmentationTask -----//
    //-------------------------------------//
    class_<CSemanticSegTaskWrap, bases<C2dImageTask>, std::shared_ptr<CSemanticSegTaskWrap>>("CSemanticSegmentationTask", _semSegTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorSemSegDocString, args("self", "name")))
        .def("__repr__", &CSemanticSegTask::repr, &CSemanticSegTaskWrap::default_repr)
        .def("set_active", &CSemanticSegTask::setActive, &CSemanticSegTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CSemanticSegTask::updateStaticOutputs, &CSemanticSegTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CSemanticSegTask::beginTaskRun, &CSemanticSegTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CSemanticSegTask::endTaskRun, &CSemanticSegTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CSemanticSegTask::graphicsChanged, &CSemanticSegTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CSemanticSegTask::globalInputChanged, &CSemanticSegTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CSemanticSegTask::getProgressSteps, &CSemanticSegTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CSemanticSegTask::run, &CSemanticSegTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CSemanticSegTask::stop, &CSemanticSegTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CSemanticSegTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CSemanticSegTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CSemanticSegTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CSemanticSegTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CSemanticSegTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CSemanticSegTask::executeActions, &CSemanticSegTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("get_names", &CSemanticSegTask::getNames, _classifGetNamesDocString, args("self"))
        .def("get_results", &CSemanticSegTask::getResults, _semSegGetResultsDocString, args("self"))
        .def("get_image_with_graphics", &CSemanticSegTask::getImageWithGraphics, _classifGetImgWithGraphicsDocString, args("self"))
        .def("get_image_with_mask", &CSemanticSegTask::getImageWithMask, _semSegGetImgWithMaskDocString, args("self"))
        .def("get_image_with_mask_and_graphics", &CSemanticSegTask::getImageWithMaskAndGraphics, _semSegGetImgWithMaskAndGraphicsDocString, args("self"))
        .def("read_class_names", &CSemanticSegTask::readClassNames, _classifReadClassNamesDocString, args("self", "path"))
        .def("set_colors", &CSemanticSegTask::setColors, _classifSetColorsDocString, args("self", "colors"))
        .def("set_names", &CSemanticSegTask::setNames, _classifSetNamesDocString, args("self", "names"))
        .def("set_mask", &CSemanticSegTask::setMask, _semSegGetMaskDocString, args("self", "mask"))
    ;

    //-------------------------------------//
    //----- CInstanceSegmentationTask -----//
    //-------------------------------------//
    class_<CInstanceSegTaskWrap, bases<C2dImageTask>, std::shared_ptr<CInstanceSegTaskWrap>>("CInstanceSegmentationTask", _instanceSegTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorInstanceSegDocString, args("self", "name")))
        .def("__repr__", &CInstanceSegTask::repr, &CInstanceSegTaskWrap::default_repr)
        .def("set_active", &CInstanceSegTask::setActive, &CInstanceSegTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CInstanceSegTask::updateStaticOutputs, &CInstanceSegTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CInstanceSegTask::beginTaskRun, &CInstanceSegTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CInstanceSegTask::endTaskRun, &CInstanceSegTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CInstanceSegTask::graphicsChanged, &CInstanceSegTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CInstanceSegTask::globalInputChanged, &CInstanceSegTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CInstanceSegTask::getProgressSteps, &CInstanceSegTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CInstanceSegTask::run, &CInstanceSegTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CInstanceSegTask::stop, &CInstanceSegTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CInstanceSegTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CInstanceSegTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CInstanceSegTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CInstanceSegTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CInstanceSegTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CInstanceSegTask::executeActions, &CInstanceSegTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("add_object", &CInstanceSegTask::addObject, _instanceSegAddInstanceDocString, args("self", "id", "type", "class_index", "confidence", "x", "y", "width", "height", "mask"))
        .def("get_names", &CInstanceSegTask::getNames, _classifGetNamesDocString, args("self"))
        .def("get_results", &CInstanceSegTask::getResults, _instanceSegGetResultsDocString, args("self"))
        .def("get_image_with_graphics", &CInstanceSegTask::getImageWithGraphics, _classifGetImgWithGraphicsDocString, args("self"))
        .def("get_image_with_mask", &CInstanceSegTask::getImageWithMask, _semSegGetImgWithMaskDocString, args("self"))
        .def("get_image_with_mask_and_graphics", &CInstanceSegTask::getImageWithMaskAndGraphics, _instanceSegGetImgWithMaskAndGraphicsDocString, args("self"))
        .def("read_class_names", &CInstanceSegTask::readClassNames, _classifReadClassNamesDocString, args("self", "path"))
        .def("set_colors", &CInstanceSegTask::setColors, _classifSetColorsDocString, args("self", "colors"))
        .def("set_names", &CInstanceSegTask::setNames, _classifSetNamesDocString, args("self", "names"))
    ;

    //----------------------------------//
    //----- CKeypointDetectionTask -----//
    //----------------------------------//
    class_<CKeyptsDetectTaskWrap, bases<C2dImageTask>, std::shared_ptr<CKeyptsDetectTaskWrap>>("CKeypointDetectionTask", _keyDetTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorKeyDetDocString, args("self", "name")))
        .def("__repr__", &CKeypointDetectionTask::repr, &CKeyptsDetectTaskWrap::default_repr)
        .def("set_active", &CKeypointDetectionTask::setActive, &CKeyptsDetectTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("update_static_outputs", &CKeypointDetectionTask::updateStaticOutputs, &CKeyptsDetectTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CKeypointDetectionTask::beginTaskRun, &CKeyptsDetectTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CKeypointDetectionTask::endTaskRun, &CKeyptsDetectTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("graphics_changed", &CKeypointDetectionTask::graphicsChanged, &CKeyptsDetectTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CKeypointDetectionTask::globalInputChanged, &CKeyptsDetectTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("get_progress_steps", &CKeypointDetectionTask::getProgressSteps, &CKeyptsDetectTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("run", &CKeypointDetectionTask::run, &CKeyptsDetectTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CKeypointDetectionTask::stop, &CKeyptsDetectTaskWrap::default_stop, _stopDocString, args("self"))
        .def("emit_add_sub_progress_steps", &CKeyptsDetectTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressSteps, args("self", "count"))
        .def("emit_step_progress", &CKeyptsDetectTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CKeyptsDetectTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CKeyptsDetectTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CKeyptsDetectTaskWrap::emitParametersChanged, _emitParamsModifiedDocString, args("self"))
        .def("execute_actions", &CKeypointDetectionTask::executeActions, &CKeyptsDetectTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("add_object", &CKeypointDetectionTask::addObject, _keyDetAddObjectDocString, args("self", "id", "class_index", "confidence", "x", "y", "width", "height", "keypoints"))
        .def("get_keypoint_links", &CKeypointDetectionTask::getKeypointLinks, _getKeyptsLinksDocString, args("self"))
        .def("get_keypoint_names", &CKeypointDetectionTask::getKeypointNames, _getKeyptsNamesDocString, args("self"))
        .def("get_object_names", &CKeypointDetectionTask::getObjectNames, _classifGetNamesDocString, args("self"))
        .def("get_results", &CKeypointDetectionTask::getResults, _keyDetGetResultsDocString, args("self"))
        .def("get_image_with_graphics", &CKeypointDetectionTask::getImageWithGraphics, _classifGetImgWithGraphicsDocString, args("self"))
        .def("read_class_names", &CKeypointDetectionTask::readClassNames, _classifReadClassNamesDocString, args("self", "path"))
        .def("set_keypoint_links", &CKeypointDetectionTask::setKeypointLinks, _setKeyptLinksDocString, args("self", "links"))
        .def("set_keypoint_names", &CKeypointDetectionTask::setKeypointNames, _setKeyptNamesDocString, args("self", "names"))
        .def("set_object_colors", &CKeypointDetectionTask::setObjectColors, _classifSetColorsDocString, args("self", "colors"))
        .def("set_object_names", &CKeypointDetectionTask::setObjectNames, _classifSetNamesDocString, args("self", "names"))
    ;

    //---------------------------//
    //----- CIkomiaRegistry -----//
    //---------------------------//
    WorkflowTaskPtr (CIkomiaRegistry::*createInstance1)(const std::string&) = &CIkomiaRegistry::createInstance;
    WorkflowTaskPtr (CIkomiaRegistry::*createInstance2)(const std::string&, const WorkflowTaskParamPtr&) = &CIkomiaRegistry::createInstance;
    WorkflowTaskPtr (CIkomiaRegistry::*createInstance3)(const std::string&, const UMapString&) = &CIkomiaRegistry::createInstance;

    class_<CIkomiaRegistryWrap, std::shared_ptr<CIkomiaRegistryWrap>, boost::noncopyable>("CIkomiaRegistry", _ikomiaRegistryDocString)
        .def(init<>("Default constructor", args("self")))
        .def("set_plugins_directory", &CIkomiaRegistry::setPluginsDirectory, _setPluginsDirDocString, args("self", "directory"))
        .def("get_plugins_directory", &CIkomiaRegistry::getPluginsDirectory, _getPluginsDirDocString, args("self"))
        .def("get_algorithms", &CIkomiaRegistry::getAlgorithms, _getAlgorithmsDocString, args("self)"))
        .def("get_algorithm_info", &CIkomiaRegistryWrap::getAlgorithmInfo, _getAlgorithmInfoDocString, args("self", "name"))
        .def("is_all_loaded", &CIkomiaRegistry::isAllLoaded, _isAllLoadedDocString, args("self"))
        .def("create_instance", createInstance1, _createInstance1DocString, args("self", "name"))
        .def("create_instance", createInstance2, _createInstance2DocString, args("self", "name", "parameters"))
        .def("create_instance", createInstance3, _createInstance3DocString, args("self", "name", "parameters"))
        .def("register_task", &CIkomiaRegistry::registerTask, _registerTaskDocString, args("self", "factory"))
        .def("register_io", &CIkomiaRegistry::registerIO, _registerIODocString, args("self", "factory"))
        .def("load_algorithms", &CIkomiaRegistry::loadPlugins, _loadPluginsDocString, args("self"))
        .def("load_cpp_algorithms", &CIkomiaRegistry::loadCppPlugins, _loadCppPluginsDocString, args("self"))
        .def("load_python_algorithms", &CIkomiaRegistry::loadPythonPlugins, _loadPythonPluginsDocString, args("self"))
        .def("load_cpp_algorithm", &CIkomiaRegistry::loadCppPlugin, _loadCppPluginDocString, args("self", "path"))
        .def("load_python_algorithm", &CIkomiaRegistry::loadPythonPlugin, _loadPythonPluginDocString, args("self", "path"))
        .def("get_black_listed_packages", &CIkomiaRegistry::getBlackListedPackages)
        .staticmethod("get_black_listed_packages")
    ;

    //---------------------//
    //----- CWorkflow -----//
    //---------------------//
    registerStdVector<std::intptr_t>();

    void (CWorkflowWrap::*addInputRef)(const WorkflowTaskIOPtr&) = &CWorkflowWrap::addInput;
    void (CWorkflowWrap::*setInputNewSeq)(const WorkflowTaskIOPtr&, size_t, bool) = &CWorkflowWrap::setInput;

    class_<CWorkflowWrap, bases<CWorkflowTask>, std::shared_ptr<CWorkflowWrap>>("CWorkflow", _workflowDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1WorkflowDocString, args("self", "name")))
        .def(init<const std::string&, const std::shared_ptr<CIkomiaRegistry>&>(_ctor2WorkflowDocString, args("self", "name", "registry")))
        .def(init<const CWorkflow&>("Copy constructor"))
        .add_property("description", &CWorkflow::getDescription, &CWorkflow::setDescription, "Workflow description")
        .add_property("keywords", &CWorkflow::getKeywords, &CWorkflow::setKeywords, "Workflow associated keywords")
        .def("__repr__", &CWorkflow::repr, &CWorkflowWrap::default_repr)
        .def("set_input", setInputNewSeq, _wfSetInputDocString, args("self", "input", "index", "new_sequence"))
        .def("set_output_folder", &CWorkflow::setOutputFolder, _wfSetOutputFolderDocString, args("self", "path"))
        .def("set_auto_save", &CWorkflow::setAutoSave, _wfSetAutoSaveDocString, args("self", "enable"))
        .def("set_cfg_entry", &CWorkflow::setCfgEntry, _wfSetCfgEntryDocString, args("self", "key", "value"))
        .def("set_exposed_parameter", &CWorkflow::setExposedParameter, _wfSetExposedParamDocString, args("self", "name", "value"))
        .def("get_task_count", &CWorkflow::getTaskCount, _wfGetTaskCountDocString, args("self"))
        .def("get_root_id", &CWorkflowWrap::getRootID, _wfGetRootIDDocString, args("self"))
        .def("get_task_ids", &CWorkflowWrap::getTaskIDs, _wfGetTaskIDsDocString, args("self"))
        .def("get_last_task_id", &CWorkflowWrap::getLastTaskID, _wfGetLastTaskIDDocString, args("self"))
        .def("get_task", &CWorkflowWrap::getTask, _wfGetTaskDocString, args("self", "id"))
        .def("get_parents", &CWorkflowWrap::getParents, _wfGetParentsDocString, args("self", "id"))
        .def("get_children", &CWorkflowWrap::getChildren, _wfGetChildrenDocString, args("self", "id"))
        .def("get_in_edges", &CWorkflowWrap::getInEdges, _wfGetInEdgesDocString, args("self", "id"))
        .def("get_out_edges", &CWorkflowWrap::getOutEdges, _wfGetOutEdgesDocString, args("self", "id"))
        .def("get_edge_info", &CWorkflowWrap::getEdgeInfo, _wfGetEdgeInfoDocString, args("self", "id"))
        .def("get_edge_source", &CWorkflowWrap::getEdgeSource, _wfGetEdgeSourceDocString, args("self", "id"))
        .def("get_edge_target", &CWorkflowWrap::getEdgeTarget, _wfGetEdgeTargetDocString, args("self", "id"))
        .def("get_final_tasks", &CWorkflow::getFinalTasks, _wfGetFinalTasks, args("self"))
        .def("get_root_target_types", &CWorkflow::getRootTargetTypes, _wfGetRootTargetTypesDocString, args("self"))
        .def("get_total_elapsed_time", &CWorkflow::getTotalElapsedTime, _wfGetTotalElapsedTimeDocString, args("self"))
        .def("get_elapsed_time_to", &CWorkflowWrap::getElapsedTimeTo, _wfGetElapsedTimeToDocString, args("self", "task_id"))
        .def("get_required_tasks", &CWorkflow::getRequiredTasks, _wfGetRequiredTasks, args("self", "path"))
        .def("get_last_run_folder", &CWorkflow::getLastRunFolder, _wfGetLastRunFolder, args("self"))
        .def("get_exposed_parameters", &CWorkflowWrap::getExposedParameters, _wfGetExposedParamsDocString, args("self"))
        .def("get_output_count", &CWorkflow::getOutputCount, _getOutputCountDocString, args("self"))
        .def("get_outputs", &CWorkflow::getOutputs, _getOutputsDocString, args("self"))
        .def("get_output", &CWorkflow::getOutput, _getOutputDocString, args("self", "index"))
        .def("get_output_data_type", &CWorkflow::getOutputDataType, _getOutputDataTypeDocString, args("self", "index"))
        .def("add_input", addInputRef, _wfAddInputDocString, args("self", "input"))
        .def("add_task", &CWorkflowWrap::addTaskWrap, _wfAddTaskDocString, args("self", "task"))
        .def("add_exposed_parameter", &CWorkflowWrap::addExposedParameter, _wfAddExposedParameterDocString, args("self", "name", "description", "task_id", "target_param_name"))
        .def("add_output", &CWorkflowWrap::addOutput, _wfAddOutputDocString, args("self", "description", "task_id", "task_output_index"))
        .def("connect", &CWorkflowWrap::connectWrap, _wfConnectDocString, args("self", "source", "target", "source_index", "target_index"))
        .def("remove_input", &CWorkflow::removeInput, _wfRemoveInputDocString, args("self", "index"))
        .def("clear_inputs", &CWorkflow::clearInputs, _wfClearInputsDocString, args("self"))
        .def("clear_output_data", &CWorkflow::clearAllOutputData, _wfClearOutputDataDocString, args("self"))
        .def("clear", &CWorkflowWrap::clearWrap, _wfClearDocString, args("self"))
        .def("clear_exposed_parameters", &CWorkflow::clearExposedParameters, _wfClearExposedParamsDocString, args("self"))
        .def("clear_outputs", &CWorkflow::clearOutputs, _wfClearOutputsDocString, args("self"))
        .def("delete_task", &CWorkflowWrap::deleteTaskWrap, _wfDeleteTaskDocString, args("self", "id"))
        .def("delete_edge", &CWorkflowWrap::deleteEdgeWrap, _wfDeleteEdgeDocString, args("self", "id"))
        .def("run", &CWorkflow::run, &CWorkflowWrap::default_run, _wfRunDocString, args("self"))
        .def("stop", &CWorkflow::stop, _wfStopDocString, args("self"))
        .def("update_start_time", &CWorkflow::updateStartTime, _wfUpdateStartTimeDocString, args("self"))
        .def("load", &CWorkflowWrap::loadWrap, _wfLoadDocString, args("self", "path"))
        .def("save", &CWorkflow::save, _wfSaveDocString, args("self", "path"))
        .def("export_graphviz", &CWorkflow::writeGraphviz, _wfExportGraphvizDocString, args("self", "path"))
    ;
}
