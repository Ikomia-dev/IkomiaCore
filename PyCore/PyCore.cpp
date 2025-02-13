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

#include "PyCore.h"
#include "PyCoreDocString.hpp"
#include <QString>
#include <QHash>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "CWorkflowTaskParamWrap.h"
#include "CWorkflowTaskIOWrap.h"
#include "CWorkflowTaskWrap.h"
#include "CWorkflowTaskWidgetWrap.h"
#include "CTaskIOFactoryWrap.h"
#include "Data/CvMatNumpyArrayConverter.h"
#include "Data/CMeasure.h"
#include "CGraphicsItemWrap.h"
#include "Graphics/CGraphicsComplexPolygon.h"
#include "Graphics/CGraphicsEllipse.h"
#include "Graphics/CGraphicsPoint.h"
#include "Graphics/CGraphicsPolygon.h"
#include "Graphics/CGraphicsPolyline.h"
#include "Graphics/CGraphicsRectangle.h"
#include "Graphics/CGraphicsText.h"
#include "Graphics/CGraphicsConversion.h"
#include "VectorConverter.hpp"
#include "MapConverter.hpp"
#include "PairConverter.hpp"

template<typename T>
void exposeCPoint(const std::string& className)
{
    class_<CPoint<T>>(className.c_str(), "Generic 2D point class.", init<>("Default constructor"))
        .def(init<T, T>(_ctorCPointDocString, args("self", "x", "y")))
        .def(init<T, T, T>(_ctorCPointDocString, args("self", "x", "y", "z")))
        .add_property("x", &CPoint<T>::getX, &CPoint<T>::setX, "x-coordinate")
        .add_property("y", &CPoint<T>::getY, &CPoint<T>::setY, "y-coordinate")
        .add_property("z", &CPoint<T>::getZ, &CPoint<T>::setZ, "z-coordinate")
    ;
}

void translateCException(const CException& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE(pycore)
{
    // Enable user-defined docstrings and python signatures, while disabling the C++ signatures
    docstring_options core_docstring_options(true, true, false);

    // Set the docstring of the current module scope
    scope().attr("__doc__") = _moduleDocString;

    // Numpy initialization
    CvMatNumpyArrayConverter::init_numpy();

    // CMat <-> Numpy NdArray converters
    to_python_converter<CMat, BoostCvMatToNumpyArrayConverter>();
    BoostNumpyArrayToCvMatConverter();

    // Register smart pointers
    register_ptr_to_python<std::shared_ptr<CProxyGraphicsItem>>();
    register_ptr_to_python<std::shared_ptr<CWorkflowTaskParam>>();
    register_ptr_to_python<std::shared_ptr<CWorkflowTaskIO>>();
    register_ptr_to_python<std::shared_ptr<CWorkflowTask>>();
    register_ptr_to_python<std::shared_ptr<CWorkflowTaskWidget>>();

    // Register std::unordered_map<T>
    registerStdUMap<std::string, std::string>();

    // Register std::pair<T1,T2>
    registerStdPair<int, int>();
    registerStdPair<std::string, std::string>();
    registerStdPair<int, CPointF>();

    // Register std::vector<T>
    registerStdVector<int>();
    registerStdVector<size_t>();
    registerStdVector<double>();
    registerStdVector<float>();
    registerStdVector<std::string>();
    registerStdVector<CPoint<float>>();
    registerStdVector<std::vector<CPoint<float>>>();
    registerStdVector<ProxyGraphicsItemPtr>();
    registerStdVector<std::shared_ptr<CWorkflowTaskIO>>();
    registerStdVector<std::shared_ptr<CWorkflowTask>>();
    registerStdVector<CMeasure>();
    registerStdVector<IODataType>();
    registerStdVector<std::pair<int, int>>();
    registerStdVector<std::pair<std::string, std::string>>();
    registerStdVector<std::pair<int, CPointF>>();

    //Register exceptions
    register_exception_translator<CException>(&translateCException);

    //--------------------//
    //----- Graphics -----//
    //--------------------//
    enum_<GraphicsShape>("GraphicsShape", "Enum - List of available graphics shapes")
        .value("ELLIPSE", GraphicsShape::ELLIPSE)
        .value("FREEHAND_POLYGON", GraphicsShape::FREEHAND_POLYGON)
        .value("FREEHAND_POLYLINE", GraphicsShape::FREEHAND_POLYLINE)
        .value("LINE", GraphicsShape::LINE)
        .value("POINT", GraphicsShape::POINT)
        .value("POLYGON", GraphicsShape::POLYGON)
        .value("POLYLINE", GraphicsShape::POLYLINE)
        .value("RECTANGLE", GraphicsShape::RECTANGLE)
        .value("SELECTION", GraphicsShape::SELECTION)
        .value("TEXT", GraphicsShape::TEXT)
    ;

    enum_<GraphicsItem>("GraphicsItem", "Enum - List of available graphics item types (ie annotations)")
        .value("LAYER", GraphicsItem::LAYER)
        .value("ELLIPSE", GraphicsItem::ELLIPSE)
        .value("RECTANGLE", GraphicsItem::RECTANGLE)
        .value("POINT", GraphicsItem::POINT)
        .value("POLYGON", GraphicsItem::POLYGON)
        .value("COMPLEX_POLYGON", GraphicsItem::COMPLEX_POLYGON)
        .value("POLYLINE", GraphicsItem::POLYLINE)
        .value("TEXT", GraphicsItem::TEXT)
    ;

    exposeCPoint<float>("CPointF");    

    //Base class of all graphics items
    class_<CGraphicsItemWrap, std::shared_ptr<CGraphicsItemWrap>, boost::noncopyable>("CGraphicsItem", _graphicsItemDocString, init<>("Default constructor", args("self")))
        .def(init<GraphicsItem>("Constructor with item type definition", args("self", "type")))
        .def("get_bounding_rect", pure_virtual(&CProxyGraphicsItem::getBoundingRect), _getGraphicsBoundRectDocString, args("self"))
        .def("get_category", pure_virtual(&CProxyGraphicsItem::getCategory), _getGraphicsCategoryDocString, args("self"))
        .def("get_id", &CProxyGraphicsItem::getId, _getGraphicsIdDocString, args("self"))
        .def("get_type", &CProxyGraphicsItem::getType, _getGraphicsTypeDocString, args("self"))
        .def("insert_to_image", pure_virtual(&CGraphicsItemWrap::insertToImage), "**Internal use only.**")
        .def("is_text_item", &CProxyGraphicsItem::isTextItem, &CGraphicsItemWrap::default_isTextItem, _isGraphicsTextItemDocString, args("self"))
        .def("set_category", pure_virtual(&CProxyGraphicsItem::setCategory), _setGraphicsCategoryDocString, args("self", "category"))
    ;

    //Complex polygon
    class_<CProxyGraphicsComplexPoly, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsComplexPoly>>("CGraphicsComplexPolygon", _graphicsComplexPolyDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::vector<CPoint<float>>, const std::vector<PolygonF>>(_ctor1GraphicsComplexPoly, args("self", "outer", "inners")))
        .def(init<const std::vector<CPoint<float>>, const std::vector<PolygonF>, const CGraphicsPolygonProperty&>(_ctor2GraphicsComplexPoly, args("self", "outer", "inners", "property")))
        .add_property("outer", &CProxyGraphicsComplexPoly::getOuter, &CProxyGraphicsComplexPoly::setOuter, "Outer polygon (list of vertices)")
        .add_property("inners", &CProxyGraphicsComplexPoly::getInners, &CProxyGraphicsComplexPoly::setInners, "Inner polygons (list of inner polygons corresponding to holes)")
        .add_property("property", &CProxyGraphicsComplexPoly::getProperty, &CProxyGraphicsComplexPoly::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsPolygonProperty`")
    ;

    //Ellipse
    class_<CGraphicsEllipseProperty>("GraphicsEllipseProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsEllipse` item.")
        .add_property("pen_color", &CGraphicsEllipseProperty::getPenColor, &CGraphicsEllipseProperty::setPenColor, "Outline color (list - rgba)")
        .add_property("brush_color", &CGraphicsEllipseProperty::getBrushColor, &CGraphicsEllipseProperty::setBrushColor, "Fill color (list - rgba)")
        .def_readwrite("line_size", &CGraphicsEllipseProperty::m_lineSize, "Outline size")
        .def_readwrite("category", &CGraphicsEllipseProperty::m_category, "Graphics category")
    ;

    class_<CProxyGraphicsEllipse, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsEllipse>>("CGraphicsEllipse", _graphicsEllipseDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<float, float, float, float>(_ctor1GraphicsEllipse, args("self", "left", "top", "width", "height")))
        .def(init<float, float, float, float, const CGraphicsEllipseProperty&>(_ctor2GraphicsEllipse, args("self", "left", "top", "width", "height", "property")))
        .add_property("x", &CProxyGraphicsEllipse::getX, &CProxyGraphicsEllipse::setX, "x coordinate of top-left point")
        .add_property("y", &CProxyGraphicsEllipse::getY, &CProxyGraphicsEllipse::setY, "y coordinate of top-left point")
        .add_property("width", &CProxyGraphicsEllipse::getWidth, &CProxyGraphicsEllipse::setWidth, "Ellipse width")
        .add_property("height", &CProxyGraphicsEllipse::getHeight, &CProxyGraphicsEllipse::setHeight, "Ellipse height")
        .add_property("property", &CProxyGraphicsEllipse::getProperty, &CProxyGraphicsEllipse::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsEllipseProperty`")
    ;

    //Point
    class_<CGraphicsPointProperty>("GraphicsPointProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsPoint` item.")
        .add_property("pen_color", &CGraphicsPointProperty::getPenColor, &CGraphicsPointProperty::setPenColor, "Outline color (list - rgba)")
        .add_property("brush_color", &CGraphicsPointProperty::getBrushColor, &CGraphicsPointProperty::setBrushColor, "Fill color (list - rgba)")
        .def_readwrite("size", &CGraphicsPointProperty::m_size, "Size")
        .def_readwrite("category", &CGraphicsPointProperty::m_category, "Graphics category")
    ;

    class_<CProxyGraphicsPoint, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsPoint>>("CGraphicsPoint", _graphicsPointDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CPoint<float>&>(_ctor1GraphicsPoint, args("self", "point")))
        .def(init<const CPoint<float>&, const CGraphicsPointProperty&>(_ctor2GraphicsPoint, args("self", "point", "property")))
        .add_property("point", &CProxyGraphicsPoint::getPoint, &CProxyGraphicsPoint::setPoint, "2D point coordinates (:py:class:`CPointF`)")
        .add_property("property", &CProxyGraphicsPoint::getProperty, &CProxyGraphicsPoint::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsPointProperty`")
    ;

    //Polygon
    class_<CGraphicsPolygonProperty>("GraphicsPolygonProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsPolygon` item.")
        .add_property("pen_color", &CGraphicsPolygonProperty::getPenColor, &CGraphicsPolygonProperty::setPenColor, "Outline color (list - rgba)")
        .add_property("brush_color", &CGraphicsPolygonProperty::getBrushColor, &CGraphicsPolygonProperty::setBrushColor, "Fill color (list - rgba)")
        .def_readwrite("line_size", &CGraphicsPolygonProperty::m_lineSize, "Outline size")
        .def_readwrite("category", &CGraphicsPolygonProperty::m_category, "Graphics category")
    ;

    class_<CProxyGraphicsPolygon, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsPolygon>>("CGraphicsPolygon", _graphicsPolygonDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::vector<CPoint<float>>>(_ctor1GraphicsPolygon, args("self", "points")))
        .def(init<const std::vector<CPoint<float>>, const CGraphicsPolygonProperty&>(_ctor2GraphicsPolygon, args("self", "points", "property")))
        .add_property("points", &CProxyGraphicsPolygon::getPoints, &CProxyGraphicsPolygon::setPoints, "List of polygon vertices (:py:class:`CPointF`)")
        .add_property("property", &CProxyGraphicsPolygon::getProperty, &CProxyGraphicsPolygon::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsPolygonProperty`")
    ;

    //Polyline
    class_<CGraphicsPolylineProperty>("GraphicsPolylineProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsPolyline` item.")
        .add_property("pen_color", &CGraphicsPolylineProperty::getColor, &CGraphicsPolylineProperty::setColor, "Outline color (list - rgba)")
        .def_readwrite("line_size", &CGraphicsPolylineProperty::m_lineSize, "Outline size")
        .def_readwrite("category", &CGraphicsPolylineProperty::m_category, "Graphics category")
    ;

    class_<CProxyGraphicsPolyline, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsPolyline>>("CGraphicsPolyline", _graphicsPolylineDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::vector<CPoint<float>>>(_ctor1GraphicsPolyline, args("self", "points")))
        .def(init<const std::vector<CPoint<float>>, const CGraphicsPolylineProperty&>(_ctor2GraphicsPolyline, args("self", "points", "property")))
        .add_property("points", &CProxyGraphicsPolyline::getPoints, &CProxyGraphicsPolyline::setPoints, "List of polyline vertices (:py:class:`~ikomia.core.pycore.CPointF`)")
        .add_property("property", &CProxyGraphicsPolyline::getProperty, &CProxyGraphicsPolyline::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsPolylineProperty`")
    ;

    //Rectangle
    class_<CGraphicsRectProperty>("GraphicsRectProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsRectangle` item.")
        .add_property("pen_color", &CGraphicsRectProperty::getPenColor, &CGraphicsRectProperty::setPenColor, "Outline color (list - rgba)")
        .add_property("brush_color", &CGraphicsRectProperty::getBrushColor, &CGraphicsRectProperty::setBrushColor, "Fill color (list - rgba)")
        .def_readwrite("line_size", &CGraphicsRectProperty::m_lineSize, "Outline size")
        .def_readwrite("category", &CGraphicsRectProperty::m_category, "Graphics category")
    ;

    class_<CProxyGraphicsRect, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsRect>>("CGraphicsRectangle", _graphicsRectangleDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<float, float, float, float>(_ctor1GraphicsRectangle, args("self", "left", "top", "width", "height")))
        .def(init<float, float, float, float, const CGraphicsRectProperty&>(_ctor2GraphicsRectangle, args("self", "left", "top", "width", "height", "property")))
        .add_property("x", &CProxyGraphicsRect::getX, &CProxyGraphicsRect::setX, "x coordinate of top-left point")
        .add_property("y", &CProxyGraphicsRect::getY, &CProxyGraphicsRect::setY, "y coordinate of top-left point")
        .add_property("width", &CProxyGraphicsRect::getWidth, &CProxyGraphicsRect::setWidth, "Rectangle width")
        .add_property("height", &CProxyGraphicsRect::getHeight, &CProxyGraphicsRect::setHeight, "Rectangle height")
        .add_property("property", &CProxyGraphicsRect::getProperty, &CProxyGraphicsRect::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsRectProperty`")
    ;

    //Text
    class_<CGraphicsTextProperty>("GraphicsTextProperty", "Visual properties for :py:class:`~ikomia.core.pycore.CGraphicsText` item.")
        .add_property("color", &CGraphicsTextProperty::getColor, &CGraphicsTextProperty::setColor, "Text color (list - rgba)")
        .def_readwrite("font_name", &CGraphicsTextProperty::m_fontName, "Font family name")
        .def_readwrite("font_size", &CGraphicsTextProperty::m_fontSize, "Font size")
        .def_readwrite("bold", &CGraphicsTextProperty::m_bBold, "Bold (boolean)")
        .def_readwrite("italic", &CGraphicsTextProperty::m_bItalic, "Italic (boolean)")
        .def_readwrite("underline", &CGraphicsTextProperty::m_bUnderline, "Underline (boolean)")
        .def_readwrite("strikeout", &CGraphicsTextProperty::m_bStrikeOut, "Strikeout (boolean)")
    ;

    class_<CProxyGraphicsText, bases<CProxyGraphicsItem>, std::shared_ptr<CProxyGraphicsText>>("CGraphicsText", _graphicsTextDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctor1GraphicsText, args("self", "text")))
        .def(init<const std::string&, float, float>(_ctor2GraphicsText, args("self", "text", "x", "y")))
        .def(init<const std::string&, float, float, const CGraphicsTextProperty&>(_ctor3GraphicsText, args("self", "text", "x", "y", "property")))
        .add_property("x", &CProxyGraphicsText::getX, &CProxyGraphicsText::setX, "x coordinate of top-left point")
        .add_property("y", &CProxyGraphicsText::getY, &CProxyGraphicsText::setY, "y coordinate of top-left point")
        .add_property("text", &CProxyGraphicsText::getText, &CProxyGraphicsText::setText, "Text string")
        .add_property("property", &CProxyGraphicsText::getProperty, &CProxyGraphicsText::setProperty, "Visual properties :py:class:`~ikomia.core.pycore.GraphicsTextProperty`")
    ;

    //-------------------------------//
    //----- CGraphicsConversion -----//
    //-------------------------------//
    CMat (CGraphicsConversion::*proxyGraphicsToBinaryMask)(const std::vector<std::shared_ptr<CProxyGraphicsItem>>&) = &CGraphicsConversion::graphicsToBinaryMask;

    class_<CGraphicsConversion>("CGraphicsConversion", _graphicsConvDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<int, int>(_ctorGraphicsConv, args("self", "width", "height")))
        .def("graphics_to_binary_mask", proxyGraphicsToBinaryMask, _graphicsToBinaryMaskDocString, args("self", "graphics"))
    ;

    //------------------------------//
    //----- CWorkflowTaskParam -----//
    //------------------------------//
    class_<CWorkflowTaskParamWrap, std::shared_ptr<CWorkflowTaskParamWrap>>("CWorkflowTaskParam", _WorkflowTaskParamDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const CWorkflowTaskParam&>("Copy constructor", args("self")))
        .def(self_ns::str(self_ns::self))
        .def_pickle(TaskParamPickleSuite())
        .def("set_values", &CWorkflowTaskParam::setParamMap, &CWorkflowTaskParamWrap::default_setParamMap, _setParamMapDocString, args("self", "params"))
        .def("get_values", &CWorkflowTaskParam::getParamMap, &CWorkflowTaskParamWrap::default_getParamMap, _getParamMapDocString, args("self"))
        .def("get_hash_value", &CWorkflowTaskParam::getHashValue, &CWorkflowTaskParamWrap::default_getHashValue, _getHashValueDocString, args("self"))
    ;

    //---------------------------//
    //----- CWorkflowTaskIO -----//
    //---------------------------//
    enum_<IODataType>("IODataType", "Enum - List of available input/output data types")
        .value("NONE", IODataType::NONE)
        .value("IMAGE", IODataType::IMAGE)
        .value("IMAGE_BINARY", IODataType::IMAGE_BINARY)
        .value("IMAGE_LABEL", IODataType::IMAGE_LABEL)
        .value("VOLUME", IODataType::VOLUME)
        .value("VOLUME_BINARY", IODataType::VOLUME_BINARY)
        .value("VOLUME_LABEL", IODataType::VOLUME_LABEL)
        .value("VIDEO", IODataType::VIDEO)
        .value("VIDEO_BINARY", IODataType::VIDEO_BINARY)
        .value("VIDEO_LABEL", IODataType::VIDEO_LABEL)
        .value("LIVE_STREAM", IODataType::LIVE_STREAM)
        .value("LIVE_STREAM_BINARY", IODataType::LIVE_STREAM_BINARY)
        .value("LIVE_STREAM_LABEL", IODataType::LIVE_STREAM_LABEL)
        .value("INPUT_GRAPHICS", IODataType::INPUT_GRAPHICS)
        .value("OUTPUT_GRAPHICS", IODataType::OUTPUT_GRAPHICS)
        .value("NUMERIC_VALUES", IODataType::NUMERIC_VALUES)
        .value("BLOB_VALUES", IODataType::BLOB_VALUES)
        .value("DESCRIPTORS", IODataType::DESCRIPTORS)
        .value("WIDGET", IODataType::WIDGET)
        .value("FOLDER_PATH", IODataType::FOLDER_PATH)
        .value("FILE_PATH", IODataType::FILE_PATH)
        .value("DNN_DATASET", IODataType::DNN_DATASET)
        .value("ARRAY", IODataType::ARRAY)
        .value("DATA_DICT", IODataType::DATA_DICT)
        .value("OBJECT_DETECTION", IODataType::OBJECT_DETECTION)
        .value("INSTANCE_SEGMENTATION", IODataType::INSTANCE_SEGMENTATION)
        .value("SEMANTIC_SEGMENTATION", IODataType::SEMANTIC_SEGMENTATION)
        .value("KEYPOINTS", IODataType::KEYPOINTS)
        .value("TEXT", IODataType::TEXT)
    ;

    void (CWorkflowTaskIOWrap::*ioSave)(const std::string&) = &CWorkflowTaskIOWrap::save;
    std::string (CWorkflowTaskIO::*toJsonNoOpt)() const = &CWorkflowTaskIO::toJson;
    std::string (CWorkflowTaskIO::*toJson)(const std::vector<std::string>&) const = &CWorkflowTaskIO::toJson;

    class_<CWorkflowTaskIOWrap, std::shared_ptr<CWorkflowTaskIOWrap>>("CWorkflowTaskIO", _WorkflowTaskIODocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<IODataType>(_ctor1WorkflowTaskIODocString, args("self", "data_type")))
        .def(init<IODataType, const std::string&>(_ctor2WorkflowTaskIODocString, args("self", "data_type", "name")))
        .def(init<const CWorkflowTaskIO&>("Copy constructor"))
        .add_property("name", &CWorkflowTaskIO::getName, &CWorkflowTaskIO::setName, "I/O name")
        .add_property("data_type", &CWorkflowTaskIO::getDataType, &CWorkflowTaskIO::setDataType, "I/O data type")
        .add_property("dim_count", &CWorkflowTaskIO::getDimensionCount, &CWorkflowTaskIO::setDimensionCount, "Number of dimensions")
        .add_property("description", &CWorkflowTaskIO::getDescription, &CWorkflowTaskIO::setDescription, "Custom description to explain input/output type and use")
        .add_property("displayable", &CWorkflowTaskIO::isDisplayable, &CWorkflowTaskIO::setDisplayable, "Displayable status (Ikomia Studio)")
        .add_property("auto_save", &CWorkflowTaskIO::isAutoSave, &CWorkflowTaskIO::setAutoSave, "Auto-save status")
        .def_readonly("source_file_path", &CWorkflowTaskIO::getSourceFilePath, "Path to the source file used as workflow input (if any)")
        .def_readonly("file_path", &CWorkflowTaskIO::getSavePath, "Path where the output is saved.")
        .def(self_ns::str(self_ns::self))
        .def("__repr__", &CWorkflowTaskIO::repr)
        .def("get_unit_element_count", &CWorkflowTaskIO::getUnitElementCount, &CWorkflowTaskIOWrap::default_getUnitElementCount, _getUnitElementCountDocString, args("self"))
        .def("get_sub_io_list", &CWorkflowTaskIO::getSubIOList, &CWorkflowTaskIOWrap::default_getSubIOList, _getSubIOListDocString, args("self", "types"))
        .def("is_data_available", &CWorkflowTaskIO::isDataAvailable, &CWorkflowTaskIOWrap::default_isDataAvailable, _isDataAvailableDocString, args("self"))
        .def("is_composite", &CWorkflowTaskIO::isComposite, &CWorkflowTaskIOWrap::default_isComposite, _isCompositeDocString, args("self"))
        .def("clear_data", &CWorkflowTaskIO::clearData, &CWorkflowTaskIOWrap::default_clearData, _clearDataDocString, args("self"))
        .def("copy_static_data", &CWorkflowTaskIO::copyStaticData, &CWorkflowTaskIOWrap::default_copyStaticData, _copyStaticDataDocString, args("self", "io"))
        .def("load", &CWorkflowTaskIO::load, &CWorkflowTaskIOWrap::default_load, _loadDocString, args("self", "path"))
        .def("save", ioSave, &CWorkflowTaskIOWrap::default_save, _saveDocString, args("self", "path"))
        .def("to_json", toJsonNoOpt, &CWorkflowTaskIOWrap::default_toJsonNoOpt, _toJsonNoOptDocString, args("self"))
        .def("to_json", toJson, &CWorkflowTaskIOWrap::default_toJson, _toJsonDocString, args("self", "options"))
        .def("from_json", &CWorkflowTaskIO::fromJson, &CWorkflowTaskIOWrap::default_fromJson, _fromJsonDocString, args("self", "jsonStr"))
    ;

    //----------------------------------//
    //----- CWorkflowTaskIOFactory -----//
    //----------------------------------//
    class_<CTaskIOFactoryWrap, std::shared_ptr<CTaskIOFactoryWrap>, boost::noncopyable>("CWorkflowTaskIOFactory", _ioFactoryDocString)
        .def("create", pure_virtual(&CTaskIOFactoryWrap::create), _ioFactoryCreateDocString, args("self", "datatype"))
    ;

    //-------------------------//
    //----- CWorkflowTask -----//
    //-------------------------//
    enum_<CWorkflowTask::Type>("TaskType", "Enum - List of available process or task types")
        .value("GENERIC", CWorkflowTask::Type::GENERIC)
        .value("IMAGE_PROCESS_2D", CWorkflowTask::Type::IMAGE_PROCESS_2D)
        .value("IMAGE_PROCESS_3D", CWorkflowTask::Type::IMAGE_PROCESS_3D)
        .value("VIDEO", CWorkflowTask::Type::VIDEO)
        .value("DNN_TRAIN", CWorkflowTask::Type::DNN_TRAIN)
    ;

    enum_<CWorkflowTask::ActionFlag>("ActionFlag", "Enum - List of specific behaviors or actions that can be enabled/disabled for a task")
        .value("APPLY_VOLUME", CWorkflowTask::ActionFlag::APPLY_VOLUME)
        .value("OUTPUT_AUTO_EXPORT", CWorkflowTask::ActionFlag::OUTPUT_AUTO_EXPORT)
    ;

    enum_<AlgoType>("AlgoType", "Enum - List of algorithms general type")
        .value("INFER", AlgoType::INFER)
        .value("TRAIN", AlgoType::TRAIN)
        .value("DATASET", AlgoType::DATASET)
        .value("OTHER", AlgoType::OTHER)
    ;

    //Overload member functions
    InputOutputVect (CWorkflowTask::*getInputs)() const = &CWorkflowTask::getInputs;
    InputOutputVect (CWorkflowTask::*getOutputs)() const = &CWorkflowTask::getOutputs;
    void (CWorkflowTask::*addInputRef)(const WorkflowTaskIOPtr&) = &CWorkflowTask::addInput;
    void (CWorkflowTask::*addOutputRef)(const WorkflowTaskIOPtr&) = &CWorkflowTask::addOutput;

    class_<CWorkflowTaskWrap, std::shared_ptr<CWorkflowTaskWrap>>("CWorkflowTask", _WorkflowTaskDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<const std::string&>(_ctorWorkflowTaskDocString, args("self", "name")))
        .def(init<const CWorkflowTask&>("Copy constructor"))
        .add_property("type", &CWorkflowTask::getType, "Main purpose or data type on which the task is dedicated to.")
        .add_property("uuid", &CWorkflowTask::getUUID, "Task unique identifier")
        .add_property("name", &CWorkflowTask::getName, &CWorkflowTask::setName, "Task name (must be unique)")
        .add_property("output_folder", &CWorkflowTask::getOutputFolder, &CWorkflowTask::setOutputFolder, "Output folder when auto-save mode is enabled. Default is the current user home folder.")
        .def(self_ns::str(self_ns::self))
        .def("__repr__", &CWorkflowTaskWrap::repr)
        .def("init_long_process", &CWorkflowTask::initLongProcess, &CWorkflowTaskWrap::default_initLongProcess, _initLongProcessDocString, args("self"))
        .def("set_auto_save", &CWorkflowTask::setAutoSave, _setAutoSaveDocString, args("self", "enable"))
        .def("set_input_data_type", &CWorkflowTask::setInputDataType, &CWorkflowTaskWrap::default_setInputDataType, _setInputDataTypeDocString, args("self", "data_type", "index"))
        .def("set_input", &CWorkflowTask::setInput, &CWorkflowTaskWrap::default_setInput, _setInputDocString, args("self", "input", "index"))
        .def("set_inputs", &CWorkflowTask::setInputs, &CWorkflowTaskWrap::default_setInputs, _setInputsDocString, args("self", "inputs"))
        .def("set_output_data_type", &CWorkflowTask::setOutputDataType, &CWorkflowTaskWrap::default_setOutputDataType, _setOutputDataTypeDocString, args("self", "data_type", "index"))
        .def("set_output", &CWorkflowTask::setOutput, &CWorkflowTaskWrap::default_setOutput, _setOutputDocString, args("self", "output", "index"))
        .def("set_outputs", &CWorkflowTask::setOutputs, &CWorkflowTaskWrap::default_setOutputs, _setOutputsDocString, args("self", "outputs"))
        .def("set_param_object", &CWorkflowTask::setParam, _setParamDocString, args("self", "param"))
        .def("set_parameters", &CWorkflowTask::setParamValues, _setParamValuesDocString, args("self", "values"))
        .def("set_action_flag", &CWorkflowTask::setActionFlag, _setActionFlagDocString, args("self", "action", "is_enable"))
        .def("set_active", &CWorkflowTask::setActive, &CWorkflowTaskWrap::default_setActive, _setActiveDocString, args("self", "is_active"))
        .def("set_enabled", &CWorkflowTask::setEnabled, _setEnabledDocString, args("self", "is_enable"))
        .def("get_input_count", &CWorkflowTask::getInputCount, _getInputCountDocString, args("self"))
        .def("get_inputs", getInputs, _getInputsDocString, args("self"))
        .def("get_input", &CWorkflowTask::getInput, _getInputDocString, args("self", "index"))
        .def("get_input_data_type", &CWorkflowTask::getInputDataType, _getInputDataTypeDocString, args("self", "index"))
        .def("get_output_count", &CWorkflowTask::getOutputCount, _getOutputCountDocString, args("self"))
        .def("get_outputs", getOutputs, _getOutputsDocString, args("self"))
        .def("get_output", &CWorkflowTask::getOutput, _getOutputDocString, args("self", "index"))
        .def("get_output_data_type", &CWorkflowTask::getOutputDataType, _getOutputDataTypeDocString, args("self", "index"))
        .def("get_param_object", &CWorkflowTask::getParam, _getParamDocString, args("self"))
        .def("get_parameters", &CWorkflowTask::getParamValues, _getParamValuesDocString, args("self"))
        .def("get_elapsed_time", &CWorkflowTask::getElapsedTime, _getElapsedTimeDocString, args("self"))
        .def("get_progress_steps", &CWorkflowTask::getProgressSteps, &CWorkflowTaskWrap::default_getProgressSteps, _getProgressStepsDocString, args("self"))
        .def("is_graphics_changed_listening", &CWorkflowTask::isGraphicsChangedListening, &CWorkflowTaskWrap::default_isGraphicsChangedListening, _isGraphicsChangedListeningDocString, args("self"))
        .def("add_input", addInputRef, &CWorkflowTaskWrap::default_addInput, _addInputDocString, args("self", "input"))
        .def("add_output", addOutputRef, &CWorkflowTaskWrap::default_addOutput, _addOutputDocString, args("self", "output"))
        .def("remove_input", &CWorkflowTask::removeInput, _removeInputDocString, args("self", "index"))
        .def("remove_output", &CWorkflowTask::removeOutput, _removeOutputDocString, args("self", "index"))
        .def("clear_inputs", &CWorkflowTask::clearInputs, _clearInputsDocString, args("self"))
        .def("clear_outputs", &CWorkflowTask::clearOutputs, _clearOutputsDocString, args("self"))
        .def("run", &CWorkflowTask::run, &CWorkflowTaskWrap::default_run, _runDocString, args("self"))
        .def("stop", &CWorkflowTask::stop, &CWorkflowTaskWrap::default_stop, _stopDocString, args("self"))
        .def("execute_actions", &CWorkflowTask::executeActions, &CWorkflowTaskWrap::default_executeActions, _executeActionsDocString, args("self", "action"))
        .def("update_static_outputs", &CWorkflowTask::updateStaticOutputs, &CWorkflowTaskWrap::default_updateStaticOutputs, _updateStaticOutputsDocString, args("self"))
        .def("begin_task_run", &CWorkflowTask::beginTaskRun, &CWorkflowTaskWrap::default_beginTaskRun, _beginTaskRunDocString, args("self"))
        .def("end_task_run", &CWorkflowTask::endTaskRun, &CWorkflowTaskWrap::default_endTaskRun, _endTaskRunDocString, args("self"))
        .def("parameters_modified", &CWorkflowTask::parametersModified, &CWorkflowTaskWrap::default_parametersModified, _parametersModifiedDocString, args("self"))
        .def("graphics_changed", &CWorkflowTask::graphicsChanged, &CWorkflowTaskWrap::default_graphicsChanged, _graphicsChangedDocString, args("self"))
        .def("global_input_changed", &CWorkflowTask::globalInputChanged, &CWorkflowTaskWrap::default_globalInputChanged, _globalInputChangedDocString, args("self", "is_new_sequence"))
        .def("emit_add_sub_progress_steps", &CWorkflowTaskWrap::emitAddSubProgressSteps, _emitAddSubProgressStepsDocString, args("self", "count"))
        .def("emit_step_progress", &CWorkflowTaskWrap::emitStepProgress, _emitStepProgressDocString, args("self"))
        .def("emit_graphics_context_changed", &CWorkflowTaskWrap::emitGraphicsContextChanged, _emitGraphicsContextChangedDocString, args("self"))
        .def("emit_output_changed", &CWorkflowTaskWrap::emitOutputChanged, _emitOutputChangedDocString, args("self"))
        .def("emit_parameters_changed", &CWorkflowTaskWrap::emitParametersChanged, _emitParametersModifiedDocString, args("self"))
        .def("workflow_started", &CWorkflowTask::workflowStarted, &CWorkflowTaskWrap::default_workflowStarted, _workflowStartedDocString, args("self"))
        .def("workflow_finished", &CWorkflowTask::workflowFinished, &CWorkflowTaskWrap::default_workflowFinished, _workflowStartedDocString, args("self"))
        .def("download", &CWorkflowTask::download, _downloadDocString, args("self", "url", "to"))
    ;

    //-------------------------------//
    //----- CWorkflowTaskWidget -----//
    //-------------------------------//
    class_<CWorkflowTaskWidgetWrap, std::shared_ptr<CWorkflowTaskWidgetWrap>, boost::noncopyable>("CWorkflowTaskWidget", _WorkflowTaskWidget)
        .def(init<>("Default constructor", args("self")))
        .def(init<QWidget*>("Construct with parent window.", args("self", "parent")))
        .def("set_layout", &CWorkflowTaskWidgetWrap::setLayout, _setLayoutDocString, args("self", "layout"))
        .def("set_apply_btn_hidden", &CWorkflowTaskWidgetWrap::setApplyBtnHidden, _setApplyBtnHiddenDocString, args("self", "is_hidden"))
        .def("on_apply", pure_virtual(&CWorkflowTaskWidget::onApply), _applyDocString, args("self"))
        .def("on_parameters_changed", &CWorkflowTaskWidget::onParametersChanged, &CWorkflowTaskWidgetWrap::default_onParametersModified, _onParamsModifiedDocString, args("self"))
        .def("emit_apply", &CWorkflowTaskWidgetWrap::emitApply, _emitApplyDocString, args("self"))
        .def("emit_send_process_action", &CWorkflowTaskWidgetWrap::emitSendProcessAction, _emitSendProcessActionDocString, args("self", "action"))
        .def("emit_set_graphics_tool", &CWorkflowTaskWidgetWrap::emitSetGraphicsTool, _emitSetGraphicsToolDocString, args("self", "tool"))
        .def("emit_set_graphics_category", &CWorkflowTaskWidgetWrap::emitSetGraphicsCategory, _emitSetGraphicsCategoryDocString, args("self", "category"))
    ;

    //--------------------//
    //----- CMeasure -----//
    //--------------------//
    enum_<CMeasure::Id>("MeasureId", "Enum - List of available measures")
        .value("SURFACE", CMeasure::Id::SURFACE)
        .value("PERIMETER", CMeasure::Id::PERIMETER)
        .value("CENTROID", CMeasure::Id::CENTROID)
        .value("BBOX", CMeasure::Id::BBOX)
        .value("ORIENTED_BBOX", CMeasure::Id::ORIENTED_BBOX)
        .value("EQUIVALENT_DIAMETER", CMeasure::Id::EQUIVALENT_DIAMETER)
        .value("ELONGATION", CMeasure::Id::ELONGATION)
        .value("CIRCULARITY", CMeasure::Id::CIRCULARITY)
        .value("SOLIDITY", CMeasure::Id::SOLIDITY)
        .value("CUSTOM", CMeasure::Id::CUSTOM)
    ;

    class_<CMeasure>("CMeasure", _measureDocString)
        .def(init<>("Default constructor", args("self")))
        .def(init<int>(_ctor1MeasureDocString, args("self", "id")))
        .def(init<int, std::string>(_ctor2MeasureDocString, args("self", "id", "name")))
        .def("get_available_measures", &CMeasure::getAvailableMeasures, _getAvailableMeasuresDocString)
        .staticmethod("get_available_measures")
        .def("get_name", &CMeasure::getName, _getNameDocString, args("id"))
        .staticmethod("get_name")
        .def_readwrite("id", &CMeasure::m_id, "Measure identifier")
        .def_readwrite("name", &CMeasure::m_name, "Measure name")
    ;
}
