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

#ifndef PYDATAPROCESSDOCSTRING_HPP
#define PYDATAPROCESSDOCSTRING_HPP

constexpr auto _moduleDocString =
        "Module offering implementation of specialization classes to handle **inputs/outputs** and **tasks** involved in Ikomia workflows for concrete use cases. It consists of Python bindings from C++ Ikomia Core.\n\n"
        "In addiction to workflow management, the system uses factory design pattern to allow integration of third-party plugins. "
        "To this end, when you want to add your own plugin, you have to implement (override) three factory classes derived from the following:\n\n"
        "   - :py:class:`~ikomia.dataprocess.pydataprocess.CPluginProcessInterface`: abstract base class exposing the two factories required by the plugin engine (task and widget)\n"
        "   - :py:class:`~ikomia.dataprocess.pydataprocess.CTaskFactory`: abstract base class for process instanciation\n"
        "   - :py:class:`~ikomia.dataprocess.pydataprocess.CWidgetFactory`: abstract base class for widget instanciation\n\n"
        "This module provides class specialization for several types of usual inputs/outputs.\n"
        "It also provides class specialization for common processing task.\n"
        "You will find below details about implementation of such specializations.\n\n";

//---------------------//
//----- CTaskInfo -----//
//---------------------//
constexpr auto _processInfoDocString =
        "Manage metadata associated with a task. "
        "Information are then available for consulting in Ikomia Studio. "
        "These metadata are also used by the system search engine (task library and Ikomia HUB).\n";

//------------------------//
//----- CTaskFactory -----//
//------------------------//
constexpr auto _processFactoryDocString =
        "Abstract class defining the core structure of the task factory. "
        "The system extensibility for the task library is based on the factory design pattern. "
        "Each task must implement a factory class derived from this class. "
        "Then the system is able to instantiate dynamically a task object (even user-defined).\n\n";

constexpr auto _processFactoryInfoDocString =
        "Metadata structure associated with the task. "
        "Some fields are mandatory to allow plugin publication. "
        "See :py:class:`~ikomia.dataprocess.pydataprocess.CTaskInfo` for details.\n\n";

constexpr auto _create1DocString =
        "Pure virtual method to create new task instance with default task parameters. "
        "Must be implemented.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTask` based object: task instance\n\n";

constexpr auto _create2DocString =
        "Pure virtual method to create new task instance with the given task parameters. "
        "Must be implemented.\n\n"
        "Args:\n\n"
        "   param (:py:class:`~ikomia.core.pycore.CWorkflowTaskParam` based object): parameters instance for initial values\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTask` based object: task instance\n\n";

//--------------------------//
//----- CWidgetFactory -----//
//--------------------------//
constexpr auto _widgetFactoryDocString =
        "Abstract class defining the core structure of the widget factory. "
        "The system extensibility for the task library is based on the factory design pattern. "
        "Each task must implement a widget factory class derived from this class. "
        "Then the system is able to instantiate dynamically a task widget object (even user-defined).\n\n";

constexpr auto _widgetFactoryNameDocString =
        "Name of the associated task. This name must be the same of the one set in the task class.\n\n";

constexpr auto _createWidgetDocString =
        "Pure virtual method to create new task widget instance with the given task parameters. "
        "Must be implemented.\n\n"
        "Args:\n\n"
        "   param (:py:class:`~ikomia.core.pycore.CWorkflowTaskParam` based object): parameters instance for initial values\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTaskWidget` based object: task instance\n\n";

//-----------------------------------//
//----- CPluginProcessInterface -----//
//-----------------------------------//
constexpr auto _pluginInterfaceDocString =
        "Abstract class defining the core structure of a plugin interface. "
        "A plugin is a user-defined task that can be integrated automatically into the system. "
        "For such, it must respect this architecture based on factory design pattern. "
        "A plugin must provide and implement two factory classes: one for the task and one for its widget (used in Ikomia Studio). "
        "It should also override this CPluginProcessInterface class and provides an implementation for the two factory getters.\n\n";

constexpr auto _getProcessFactoryDocString =
        "Pure virtual method that returns instance of a task factory object. "
        "Must be implemented.\n\n"
        "Returns:\n\n"
        "   factory: :py:class:`CTaskFactory` based object\n\n";

constexpr auto _getWidgetFactoryDocString =
        "Pure virtual method that returns instance of a task factory object. "
        "Must be implemented.\n\n"
        "Returns:\n\n"
        "   factory: :py:class:`CWidgetFactory` based object\n\n";

//--------------------------//
//----- CObjectMeasure -----//
//--------------------------//
constexpr auto _objectMeasureDocString =
        "Store values of a given measure computed on a single blob of an image. "
        "Available measures are listed here: :py:class:`~ikomia.core.pycore.MeasureId`. "
        "It can also maps this measure with the associated graphics item (by its identifier). "
        "Store graphics item in a :py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput` object "
        "is the classical way to handle graphics items.\n\n";

constexpr auto _ctor1ObjMeasureDocString =
        "Construct a CObjectMeasure instance with a single value.\n\n"
        "Args:\n\n"
        "   measure (:py:class:`~ikomia.core.pycore.CMeasure`): basic information about the computed measure.\n\n"
        "   value (double): value of the measure\n\n"
        "   graphicsId (int): graphics item identifier, -1 if no associated item\n\n"
        "   label (str)\n\n";

constexpr auto _ctor2ObjMeasureDocString =
        "Construct a CObjectMeasure instance with multiple values.\n\n"
        "Args:\n\n"
        "   measure (:py:class:`~ikomia.core.pycore.CMeasure`): basic information about the computed measure.\n\n"
        "   values (list of double): values of the measure (example Bbox needs 4 values)\n\n"
        "   graphicsId (int): graphics item identifier, -1 if no associated item\n\n"
        "   label (str)\n\n";

constexpr auto _getMeasureInfoDocString =
        "Get measure information (identifier + name). See :py:class:`~ikomia.core.pycore.CMeasure`.\n\n";

//--------------------------//
//----- CBlobMeasureIO -----//
//--------------------------//
constexpr auto _measureIODocString =
        "Define input or output for a task consuming or producing measures on blobs (ie connected components). "
        "It is possible to compute and store several measures for a single blob. A CBlobMeasureIO instance "
        "stores a list of measures for each blob of an image, so you have a list of :py:class:`~ikomia.dataprocess.pydataprocess.CObjectMeasure` list. "
        "Please note that it is possible to map each blob with its associated graphics item stored in a :py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput` instance. "
        "You just need to pass the graphics item identifier to the object measure\n. "
        "Blob measures can also be handled by :py:class:`~ikomia.dataprocess.pydataprocess.CNumericIO`. "
        "Although CNumericIO is more generic, it can't map measure values with graphics item, which can be useful "
        "to give visual information from object measures to users.\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`\n\n";

constexpr auto _ctorMeasureIODocString =
        "Construct a CBlobMeasureIO instance with the given name.\n\n"
        "Args:\n\n"
        "   name (str): input or output custom name (to give insigths to end user)\n\n";

constexpr auto _setObjMeasureDocString =
        "Set measure for the blob specified by the given index.\n\n"
        "Args:\n\n"
        "   index (int): zeo-based index in the blob list\n\n"
        "   measure (:py:class:`~ikomia.dataprocess.pydataprocess.CObjectMeasure`)\n\n";

constexpr auto _getMeasuresDocString =
        "Get measures for all blobs.\n\n"
        "Returns:\n\n"
        "   measures (list of :py:class:`~ikomia.dataprocess.pydataprocess.CObjectMeasure` list)\n\n";

constexpr auto _isMeasureDataAvailableDocString =
        "Return True if there is at least one measure for one blob, False otherwise.\n\n";

constexpr auto _addObjMeasureDocString =
        "Add a new blob measure. Use this method if only one measure is computed for a blob.\n\n"
        "Args:\n\n"
        "   measure (:py:class:`~ikomia.dataprocess.pydataprocess.CObjectMeasure`)\n\n";

constexpr auto _addObjMeasuresDocString =
        "Add a new list of measures for a blob.\n\n"
        "Args:\n\n"
        "   measures (list of :py:class:`~ikomia.dataprocess.pydataprocess.CObjectMeasure`)\n\n";

constexpr auto _blobMeasureIOLoadDocString =
        "Load blob measure I/O data from previously exported file. The file must be in CSV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _blobMeasureIOSaveDocString =
        "Save blob measure I/O data to file. The file must be in CSV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _blobIOToJsonNoOptDocString =
        "Return input/output data in JSON formatted string (compact mode).\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _blobIOToJsonDocString =
        "Return input/output data in JSON formatted string. JSON format option can be set, possible values are:\n\n"
        "- ['json_format', 'compact'] (**default**)\n"
        "- ['json_format', 'indented']\n\n"
        "Args:\n\n"
        "   list of str: format-specific options encoded as pairs [option_name, option_value]\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

//----------------------//
//----- CNumericIO -----//
//----------------------//
constexpr auto _featureProcessIODocString =
        "Define input or output for a task consuming or producing data as numeric values. "
        "C++ implementation uses template to handle various data types. In Python, Ikomia API exposes only the double precision (float) and string specializations. "
        "The class is designed to handle row/column data structure, "
        "it consists on a list of values, a list of associated labels and a display type (see :py:class:`NumericOutputType`). "
        "For the specific case of plot display, a plot type property is available (see :py:class:`PlotType`). "
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`\n\n";

constexpr auto _ctorFeatureIODocString =
        "Construct a CNumericIO instance with the given name.\n\n"
        "Args:\n\n"
        "   name (str): input or output custom name (to give insights to end user)\n\n";

constexpr auto _addValueList1DocString =
        "Append a new value list.\n\n"
        "Args:\n\n"
        "   values (list): list of values (float or string)\n\n";

constexpr auto _addValueList2DocString =
        "Append a new value list with the corresponding header label. A header label can be seen as a column name in a table representation.\n\n"
        "Args:\n\n"
        "   values (list): list of values (float or string)\n\n"
        "   header_label (str): header label\n\n";

constexpr auto _addValueList3DocString =
        "Append a new value list with a label for each value.\n\n"
        "Args:\n\n"
        "   values (list): list of values (float or string)\n\n"
        "   labels (list of str): store label for each numeric value\n\n";

constexpr auto _addValueList4DocString =
        "Append a new value list with the corresponding header label and a label for each value.\n\n"
        "Args:\n\n"
        "   values (list): list of values (float or string)\n\n"
        "   header_label (str): header label\n\n"
        "   labels (list of str): store label for each numeric value\n\n";

constexpr auto _clearDataDerivedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTaskIO.clear_data`.\n\n";

constexpr auto _copyStaticDataDerivedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTaskIO.copy_static_data`.\n\n";

constexpr auto _getAllLabelListDocString =
        "Get all label lists associated with value lists.\n\n"
        "Returns:\n\n"
        "   list of list of str: all labels\n\n";

constexpr auto _getAllHeaderLabelsDocString =
        "Get all header labels (column labels).\n\n"
        "Returns:\n\n"
        "   list of str\n\n";

constexpr auto _getAllValueListDocString =
        "Get all value lists.\n\n"
        "Returns:\n\n"
        "   list of list (float or string): all values\n\n";

constexpr auto _getOutputTypeDocString =
        "Get output display type. Can be one of the values defined in :py:class:`NumericOutputType`.\n\n"
        "Returns:\n\n"
        "   int: display type\n\n";

constexpr auto _getPlotTypeDocString =
        "Get plot type. Can be one of the values defined in :py:class:`PlotType`. "
        "Meaningful only if output display type is set to PLOT.\n\n"
        "Returns:\n\n"
        "   PlotType: plot type\n\n";

constexpr auto _getUnitEltCountDerivedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTaskIO.get_unit_elementCount`.\n\n";

constexpr auto _getValueListDocString =
        "Get value list at position index.\n\n"
        "Args:\n\n"
        "   index (int): index of value list\n\n"
        "Returns:\n\n"
        "   list (float or string): values\n\n";

constexpr auto _isDataAvailableDerivedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTaskIO.is_data_available`.\n\n";

constexpr auto _setOutputTypeDocString =
        "Set output display type. See :py:class:`NumericOutputType` for possible values.\n\n"
        "Args:\n\n"
        "   type (int): display type\n\n";

constexpr auto _setPlotTypeDocString =
        "Set plot type. See :py:class:`PlotType` for possible values. "
        "Only used if output display type is set to PLOT (see :py:meth:`set_output_type`).\n\n"
        "Args:\n\n"
        "   type (PlotType): plot type\n\n";

constexpr auto _numericIOLoadDocString =
        "Load data values (double or string) from previsouly exported file. At this time, file must be in CSV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _numericIOSaveDocString =
        "Save data values (double or string) to file. At this time, file must be in CSV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

//--------------------//
//----- CImageIO -----//
//--------------------//
constexpr auto _imageProcessIODocString =
        "Define task input or output for a task dedicated to image processing (2D or 3D). "
        "This class is designed to handle image as input or output. "
        "A CImageIO instance can be added as input or output to a :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived object. "
        "It is the base class to define input or output of an image processing task. "
        "Several image data type can be defined according to the nature of the algorithm:\n\n"
        "- binary image\n"
        "- labelled image\n"
        "- standard image\n\n"
        "The internal image data structure is a numpy array that can be either 2D or 3D.\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctor1imageProcessIODocString =
        "Construct a CImageIO instance with the given data type. The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Please note that internal image structure is empty.\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): image data type\n\n";

constexpr auto _ctor2imageProcessIODocString =
        "Construct a CImageIO instance with the given data type and the given image. The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Please note that internal image structure is empty.\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): image data type\n\n"
        "   image (Numpy array): 2D/3D image\n\n";

constexpr auto _ctor3imageProcessIODocString =
        "Construct a CImageIO instance with the given data type, name and image. "
        "The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): image data type\n\n"
        "   image (Numpy array): 2D/3D image\n\n"
        "   name (str): input or output custom name (give insights to end user)\n\n";

constexpr auto _ctor4imageProcessIODocString =
        "Construct a CImageIO instance with the given data type and identification name. "
        "The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): image data type\n\n"
        "   name (str): input or output custom name (give insights to end user)\n\n";

constexpr auto _ctor5imageProcessIODocString =
        "Construct a CImageIO instance with the given data type, identification name and the image loaded from the given path. "
        "The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): image data type\n\n"
        "   name (str): input or output custom name (give insights to end user)\n\n"
        "   path (str): image path to be loaded. Image is loaded to memory automatically.\n\n";

constexpr auto _clearImageDataDocString =
        "Clear image and overlay mask so that they become empty.\n\n";

constexpr auto _copyImageStaticDataDocString =
        "Set the static information from the given input or ouput. "
        "For this class, the channel count is the only static data.\n\n"
        "Args:\n\n"
        "   io (:py:class:`~ikomia.core.pycore.CWorkflowTaskIO`): input or ouput instance from which data is copied.\n\n";

constexpr auto _getChannelCountDocString =
        "Get the static channel count information. "
        "The method does not get the channel count property from the image data structure\n\n"
        "Returns:\n\n"
        "   int: number of channel required for the input/output\n\n";

constexpr auto _getDataDocString =
        "Get the image data.\n\n"
        "Returns:\n\n"
        "   Numpy array: either 2D or 3D image buffer\n\n";

constexpr auto _getImageDocString =
        "Get the 2D image data only. "
        "In case of volume, the current image index is used to get the desired 2D plane (see :py:meth:`set_current_image`).\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getImageWithGraphicsDocString =
        "Get a copy of the internal image with graphics items from the given I/O.\n\n"
        "Args:\n\n"
        "   io (:py:class:`~ikomia.dataprocess.pydataprocess.CWorkflowTaskIO`)\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getImageWithMaskDocString =
        "Get a copy of the internal image with mask overlay from the given I/O.\n\n"
        "Args:\n\n"
        "   io (:py:class:`~ikomia.dataprocess.pydataprocess.CWorkflowTaskIO`)\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getImageWithMaskAndGraphicsDocString =
        "Get a copy of the internal image with graphics and mask overlay from the given I/O.\n\n"
        "Args:\n\n"
        "   io (:py:class:`~ikomia.dataprocess.pydataprocess.CWorkflowTaskIO`)\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getOverlayMaskDocString =
        "Get the overlay mask. See :py:meth:`set_overlay_mask` for more information.\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer (8 bits - 1 channel)\n\n";

constexpr auto _getImageUnitElementCountDocString =
        "Get the number of unit elements when the data is processed. "
        "The number of unit elements is used to calculate the number of progress steps needed to perform a task. "
        "In case of 2D image, we won't update progress bar every pixel, so the return value should be 1 (1 update per image). "
        "In case of 3D image, update progress every 2D plane can make sense for task processing volume on 2D planes independently. "
        "So user have to define the number of unit elements through the progress bar update perspective.\n\n"
        "Returns:\n\n"
        "   int: number of unit element to process.\n\n";

constexpr auto _isImageDataAvailableDocString =
        "Check whether the input/output have valid image or not.\n\n"
        "Returns:\n\n"
        "   bool: True if image is not empty, False otherwise.\n\n";

constexpr auto _isOverlayAvailableDocString =
        "Check whether the input/output have valid overlay mask or not\n\n"
        "Returns:\n\n"
        "   bool: True if overlay mask is not empty, False otherwise.\n\n";

constexpr auto _setChannelCountDocString =
        "Set the channel count of the image data."
        "This property holds the required channel count as a static information. "
        "Such information can be useful when one designs a workflow and wants to inform about specific image structure towards future connected tasks.\n\n"
        "Args:\n\n"
        "   nb (int): channel count, 1 for monochrome image, 3 for color image\n\n";

constexpr auto _setCurrentImageDocString =
        "Set the index of the current image (2D plane) from a volume (3D image data structure).\n\n"
        "Args:\n\n"
        "   index (int): zero-based index of the 2D plane\n\n";

constexpr auto _setImageDocString =
        "Set the image data\n\n"
        "Args:\n\n"
        "   image (Numpy array): image buffer\n\n";

constexpr auto _setOverlayMaskDocString =
        "Set the associated overlay mask. Ikomia Studio is able to display overlay mask on top of image. "
        "This method sets this mask, it will be displayed automatically according to a predefined color map. "
        "Zero-value pixels of the mask will be completely transparent, non-zero will be displayed according to the corresponding color in the color map. "
        "The color map must be defined in the task implementation. "
        "See :py:meth:`~C2dImageTask.set_output_color_map` for details.\n\n"
        "Args:\n\n"
        "   image (Numpy array): image buffer (8 bits - 1 channel)\n\n";

constexpr auto _drawGraphicsInDocString =
        "Draw given graphics items input in the image. Warning, this function overwrite the original output image.\n\n"
        "Args:\n\n"
        "   graphics (:py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsInput`)\n\n";

constexpr auto _drawGraphicsOutDocString =
        "Draw given graphics items output in the image. Warning, this function overwrite the original output image.\n\n"
        "Args:\n\n"
        "   graphics (:py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput`)\n\n";

constexpr auto _imageIOLoadDocString =
        "Load image IO data from image file. As we use OpenCV as our image reader backend, the file must be a valid OpenCV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _imageIOSaveDocString =
        "Save image IO data to file. As we use OpenCV as our image writer backend, the file must be a valid OpenCV format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _imageIOToJsonNoOptDocString =
        "Return input/output data in JSON formatted string (compact mode and image encoded as JPEG).\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _imageIOToJsonDocString =
        "Return input/output data in JSON formatted string.\n"
        "Available options:\n\n"
        "- JSON format ['json_format', 'compact', ...] (**default**) or ['json_format', 'indented', ...]\n"
        "- image format ['image_format', 'jpg', ...] or ['image_format', 'png', ...]\n\n"
        "Args:\n\n"
        "   json_str (list of str): format-specific options encoded as pairs [option_name, option_value]\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _imageIOFromJsonIDocString =
        "Set input/output data from JSON formatted string.\n\n"
        "Args:\n\n"
        "   str: data as JSON formatted string\n\n";

//--------------------------//
//----- CGraphicsInput -----//
//--------------------------//
constexpr auto _graphicsInputDocString =
        "Define task input containing graphics items. Consult :py:class:`~ikomia.core.pycore.GraphicsItem` to see the list of possible graphics types. "
        "Instance can be added as input of a :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived. "
        "This kind of input is used to manage user-defined ROI or to forward graphics items generated by a previous task in your workflow. "
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctorGraphicsInDocString =
        "Construct a CGraphicsInput instance with the given name.\n\n"
        "Args:\n\n"
        "   name (str): input custom name (to give insights to end user)\n\n";

constexpr auto _setItemsDocString =
        "Fill input/output with the given graphics item list.\n\n"
        "Args:\n\n"
        "   items (list of :py:class:`~ikomia.core.pycore.CGraphicsItem`): based objects\n\n";

constexpr auto _getItemsDocString =
        "Get list of graphics items.\n\n"
        "Returns:\n\n"
        "   list of :py:class:`~ikomia.core.pycore.CGraphicsItem` based objects\n\n";

constexpr auto _isGraphicsDataAvailableDocString =
        "Check whether the input/output contains data.\n\n"
        "Returns:\n\n"
        "   bool: True if input contains data, False otherwise\n\n";

constexpr auto _clearGraphicsDataDocString =
        "Clear input/output data.\n\n";

constexpr auto _graphicsInputLoadDocString =
        "Load graphics input/output from previsouly exported file. The file must be in JSON format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _graphicsInputSaveDocString =
        "Save graphics input/output data to file. The file must be in JSON format.\n\n"
        "Args:\n\n"
        "   path (str)\n";

//---------------------------//
//----- CGraphicsOutput -----//
//---------------------------//
constexpr auto _graphicsOutputDocString =
        "Define task output containing graphics items. Consult :py:class:`~ikomia.core.pycore.GraphicsItem` to see the list of possible graphics types. "
        "Instance can be added as output of a :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived. "
        "This kind of output is used to manage graphics items generated in a task. "
        "Ikomia software displays it as an overlay layer on top of images or videos. "
        "Graphics items can then be forwarded as input of workflow's following tasks. "
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctorGraphicsOutDocString =
        "Construct a CGraphicsOutput instance with the given name.\n\n"
        "Args:\n\n"
        "   name (str): output custom name (to give insights to end user)\n\n";

constexpr auto _setNewLayerDocString =
        "Initiate a new graphics layer for the output with the given name. The method clears all graphics items that was previously generated.\n\n"
        "Args:\n\n"
        "   name (str): name of the associated graphics layer\n\n";

constexpr auto _setImageIndexDocString =
        "Set the output index on which graphics items will be displayed (in overlay layer). "
        "Index must refer to an existing output managing image data such as :py:class:`CImageIO` or derived.\n\n"
        "Args:\n\n"
        "   index (int): index of image-based output\n\n";

constexpr auto _getImageIndexDocString =
        "Get the output index on which graphics items should be associated. "
        "Index must refer to an existing output managing image data such as :py:class:`CImageIO` or derived.\n\n"
        "Return:\n\n"
        "   int: index of image-based output\n\n";

constexpr auto _addItemDocString =
        "Add graphics item to this output. Object class must be derived from :py:class:`~ikomia.core.pycore.CGraphicsItem`. "
        "Use this method only if you need to add user-defined graphics items. "
        "Consult :py:class:`~ikomia.core.pycore.GraphicsItem` to see the list of built-in graphics items.\n\n"
        "Args:\n\n"
        "   item (:py:class:`~ikomia.core.pycore.CGraphicsItem` based object)\n\n";

constexpr auto _addPoint1DocString =
        "Add point item to this output. Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   point (:py:class:`~ikomia.core.pycore.CPointF`): position in image coordinates\n\n";

constexpr auto _addPoint2DocString =
        "Add point item to this output with the given display properties (color, size...)\n\n"
        "Args:\n\n"
        "   point (:py:class:`~ikomia.core.pycore.CPointF`): position in image coordinates\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsPointProperty`): display properties\n\n";

constexpr auto _addRectangle1DocString =
        "Add rectangle or square item to this output. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n"
        "   width (float): width of the rectangle\n\n"
        "   height (float): height of the rectangle\n\n";

constexpr auto _addRectangle2DocString =
        "Add rectangle or square item to this output with the given display properties (color, size...).\n\n"
        "Args:\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n"
        "   width (float): width of the rectangle\n\n"
        "   height (float): height of the rectangle\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsRectProperty`): display properties\n\n";

constexpr auto _addEllipse1DocString =
        "Add ellipse or circle item to this output. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n"
        "   width (float): width of the ellipse\n\n"
        "   height (float): height of the ellipse\n\n";

constexpr auto _addEllipse2DocString =
        "Add ellipse or circle item to this output with the given display properties (color, size...).\n\n"
        "Args:\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n"
        "   width (float): width of the ellipse\n\n"
        "   height (float): height of the ellipse\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsEllipseProperty`): display properties\n\n";

constexpr auto _addPolygon1DocString =
        "Add polygon item to this output. The polygon is closed automatically. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   points (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n";

constexpr auto _addPolygon2DocString =
        "Add polygon item to this output with the given display properties (color, size...).\n\n"
        "Args:\n\n"
        "   points (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsPolygonProperty`): display properties\n\n";

constexpr auto _addPolyline1DocString =
        "Add polyline item to this output. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   points (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n";

constexpr auto _addPolyline2DocString =
        "Add polyline item to this output with the given display properties (color, size...).\n\n"
        "Args:\n\n"
        "   points (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsPolylineProperty`): display properties\n\n";

constexpr auto _addComplexPolygon1DocString =
        "Add complex polygon item to this output. A complex polygon means a polygon with one or several holes inside. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   outer (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n"
        "   inners (list of :py:class:`~ikomia.core.pycore.CPointF` list): list of inner polygons (holes)\n\n";

constexpr auto _addComplexPolygon2DocString =
        "Add complex polygon item to this output with the given display properties (color, size...). "
        "A complex polygon means a polygon with one or several holes inside.\n\n"
        "Args:\n\n"
        "   outer (:py:class:`~ikomia.core.pycore.CPointF` list): list of polygon vertices (x,y)\n\n"
        "   inners (list of :py:class:`~ikomia.core.pycore.CPointF` list): list of inner polygons (holes)\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsPolygonProperty`): display properties\n\n";

constexpr auto _addText1DocString =
        "Add text item to this output. "
        "Item properties (color, size...) come from the global preferences of Ikomia software. "
        "They can be adjusted interactively from the option button of the graphics toolbar.\n\n"
        "Args:\n\n"
        "   text (str): text to display\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n";

constexpr auto _addText2DocString =
        "Add text item to this output with the given display properties (color, size...).\n\n"
        "Args:\n\n"
        "   text (str): text to display\n\n"
        "   x (float): x-coordinate of the top-left point.\n\n"
        "   y (float): y-coordinate of the top-left point.\n\n"
        "   properties (:py:class:`~ikomia.core.pycore.GraphicsTextProperty`): display properties\n\n";

constexpr auto _graphicsOutputLoadDocString =
        "Load graphics output from previsouly exported file. The file must be in JSON format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

constexpr auto _graphicsOutputSaveDocString =
        "Save graphics output data to file. The file must be in JSON format.\n\n"
        "Args:\n\n"
        "   path (str)\n\n";

//--------------------//
//----- CVideoIO -----//
//--------------------//
constexpr auto _videoProcessIODocString =
        "Define an input or output for a task dedicated to video management. "
        "This class is designed to handle video and instance can be added as input or output to a :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived object. "
        "It is the base class to define input or output of a video processing task. "
        "Several video output type can be defined according to the nature of the algorithm:\n\n"
        "- binary video\n"
        "- labelled video (graylevel connected components)\n"
        "- standard video\n\n"
        "Source video can be either a file, an image sequence or a stream. Image data is stored as a numpy array.\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.CImageIO`.\n\n";

constexpr auto _ctor1VideoProcessIODocString =
        "Construct a CVideoIO instance with the given data type. The data type must be one of these values:\n\n"
        "- IODataType.VIDEO\n"
        "- IODataType.VIDEO_BINARY\n"
        "- IODataType.VIDEO_LABEL\n"
        "- IODataType.LIVE_STREAM\n"
        "- IODataType.LIVE_STREAM_BINARY\n"
        "- IODataType.LIVE_STREAM_LABEL\n\n"
        "Please note that internal image structure is empty.\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): video data type\n\n";

constexpr auto _ctor2VideoProcessIODocString =
        "Construct a CVideoIO instance with the given data type. The data type must be one of these values:\n\n"
        "- IODataType.VIDEO\n"
        "- IODataType.VIDEO_BINARY\n"
        "- IODataType.VIDEO_LABEL\n"
        "- IODataType.LIVE_STREAM\n"
        "- IODataType.LIVE_STREAM_BINARY\n"
        "- IODataType.LIVE_STREAM_LABEL\n\n"
        "Please note that internal image structure is empty.\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): video data type\n\n"
        "   frame (Numpy array): 2D image (first frame of the video)\n\n";

constexpr auto _ctor3VideoProcessIODocString =
        "Construct a CVideoIO instance with the given data type and the given image. "
        "The data type must be one of these values:\n\n"
        "- IODataType.VIDEO\n"
        "- IODataType.VIDEO_BINARY\n"
        "- IODataType.VIDEO_LABEL\n"
        "- IODataType.LIVE_STREAM\n"
        "- IODataType.LIVE_STREAM_BINARY\n"
        "- IODataType.LIVE_STREAM_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): video data type\n\n"
        "   frame (Numpy array): 2D image\n\n"
        "   name (str): input or output custom name (to give insights to end user)\n\n";

constexpr auto _ctor4VideoProcessIODocString =
        "Construct a CVideoIO instance with the given data type and identification name. "
        "The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): video data type\n\n"
        "   name (str): input or output custom name (to give insights to end user)\n\n";

constexpr auto _ctor5VideoProcessIODocString =
        "Construct a CImageIO instance with the given data type, identification name and video path . "
        "The data type must be one of these values:\n\n"
        "- IODataType.IMAGE\n"
        "- IODataType.IMAGE_BINARY\n"
        "- IODataType.IMAGE_LABEL\n"
        "- IODataType.VOLUME\n"
        "- IODataType.VOLUME_BINARY\n"
        "- IODataType.VOLUME_LABEL\n\n"
        "Args:\n\n"
        "   data_type (:py:class:`~ikomia.core.pycore.IODataType`): video data type\n\n"
        "   name (str): input or output name\n\n"
        "   path (str): video path\n\n";

constexpr auto _setVideoPathDocString =
        "Set the source path of the video.\n\n"
        "Args:\n\n"
        "   path (str): can be a file path, a device ID (internal webcam) or an url (IP camera)\n\n";

constexpr auto _setVideoPosDocString =
        "Set the current frame of the video.\n\n"
        "Args:\n\n"
        "   position (int): index of the frame\n\n";

constexpr auto _getVideoFrameCountDocString =
        "Get the total frames number of the video.\n\n"
        "Returns:\n\n"
        "   int: frames number\n\n";

constexpr auto _getVideoImagesDocString =
        "Get all image frames extracted from the video.\n\n"
        "Returns:\n\n"
        "   Numpy array list: list of 2D images\n\n";

constexpr auto _getVideoPathDocString =
        "Get the path to the source video. It can be a file path, a device ID or an url.\n\n"
        "Returns:\n\n"
        "   str: source video path\n\n";

constexpr auto _getSnapshotDocString =
        "Get image of a single frame.\n\n"
        "Args:\n\n"
        "   position (int): index of the frame\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getCurrentPosDocString =
        "Get the current frame index of the video or stream.\n\n"
        "Returns:\n\n"
        "   int: current frame index\n\n";

constexpr auto _startVideoDocString =
        "Start playing the video.\n\n"
        "Args:\n\n"
        "   timeout (int): maximum time between 2 read operations (in ms)\n\n";

constexpr auto _stopVideoDocString =
        "Stop playing the video.\n\n";

constexpr auto _startVideoWriteDocString =
        "Write the video to disk.\n\n"
        "Args:\n\n"
        "   width (int): video width\n\n"
        "   height (int): video height\n\n"
        "   frames (int): number of frames\n\n"
        "   fps (int): frames per second\n\n"
        "   fourcc (int): codec code (-1 for default)\n\n"
        "   timeout (int): maximum time between 2 write opertions (in ms)\n\n";

constexpr auto _stopVideoWriteDocString =
        "Stop writting to disk.\n\n"
        "Args:\n\n"
        "   timeout (int): time for reading to end (in ms)\n\n";

constexpr auto _addVideoImageDocString =
        "Append image frame to the video.\n\n"
        "Args:\n\n"
        "   image (Numpy array): 2D image\n\n";

constexpr auto _writeImageDocString =
        "Append a new image to the list of images to write to disk. The write process is launched using :py:meth:`start_video_write`.\n\n"
        "Args:\n\n"
        "   image (Numpy array): 2D image\n\n";

constexpr auto _hasVideoDocString =
        "Check whether the input or output has a video source.\n\n"
        "Returns:\n\n"
        "   bool: True if the source is valid, False otherwise\n\n";

constexpr auto _getVideoImageDocString =
        "Get the image at the current frame index.\n\n"
        "Returns:\n\n"
        "   Numpy array: 2D image buffer\n\n";

constexpr auto _getVideoUnitElementCountDocString =
        "Get the number of unit elements to process, ie the number of frames for a video. "
        "Used to determine the number of steps for progress bar.\n\n"
        "Returns:\n\n"
        "   int: frames number of the video\n\n";

constexpr auto _isVideoDataAvailableDocString =
        "Check whether the video contains valid data.\n\n"
        "Returns:\n\n"
        "   bool: True if there is some valid data, False otherwise\n\n";

constexpr auto _clearVideoDataDocString =
        "Clear the current image at the current frame index.\n\n";

//-------------------------//
//----- CWidgetOutput -----//
//-------------------------//
constexpr auto _widgetOutputDocString =
        "Define a widget output for a task. "
        "This class is designed to handle widget as output of a workflow task. "
        "A widget must be a derived object from QWidget of the Qt framework (PyQt5 or PySide2).\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctor1WidgetOutputDocString =
        "Construct a CWidgetOutput instance with the given data type."
        "By default, the widget data type is IODataType.WIDGET (see :py:class:`~ikomia.core.pycore.IODataType`).\n\n";

constexpr auto _ctor2WidgetOutputDocString =
        "Construct a CWidgetOutput instance with the given data type and name."
        "By default, the widget data type is IODataType.WIDGET (see :py:class:`~ikomia.core.pycore.IODataType`).\n\n";

constexpr auto _setWidgetDocString =
        "Set the widget instance: use :py:mod:`~ikomia.utils.qtconversion` module to get C++ handle from Python Qt-based framework.\n\n"
        "Args:\n\n"
        "   widget: C++ compatible widget handle\n\n";

constexpr auto _isWidgetDataAvailableDocString =
        "Check whether the output contains a valid widget.\n\n"
        "Returns:\n\n"
        "   bool: True if the widget handle is valid, False otherwise\n\n";

constexpr auto _clearWidgetDataDocString =
        "Clear the output data: sets the widget handle to null.\n\n";

//-------------------//
//----- CPathIO -----//
//-------------------//
constexpr auto _pathIODocString =
        "Define input or output for a task to handle file or folder path. "
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctor1PathIODocString =
        "Construct a CPathIO instance with the given data type. "
        "Data type can be either a file or folder path (see :py:class:`~ikomia.core.pycore.IODataType`).\n\n";

constexpr auto _ctor2PathIODocString =
        "Construct a CPathIO instance with the given data type and path. "
        "Data type can be either a file or folder path (see :py:class:`~ikomia.core.pycore.IODataType`).\n\n";

constexpr auto _ctor3PathIODocString =
        "Construct a CPathIO instance with the given data type, path and name. "
        "Data type can be either a file or folder path (see :py:class:`~ikomia.core.pycore.IODataType`).\n\n";

constexpr auto _setPathDocString =
        "Set the path of the input/output. The path must exist.\n\n";

constexpr auto _getPathDocString =
        "Get the path of the input/output.\n\n";

constexpr auto _clearDataDocString =
        "Clear the data stored in the object. Should be overriden for custom input or output.\n\n";

//----------------------//
//----- CDatasetIO -----//
//----------------------//
constexpr auto _datasetIODocString =
        "Virtual base classe to define task input or output containing deep learning dataset structure. "
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n"
        "Instances can be added as input or output of a :py:class:`~ikomia.core.pycore.CWorkflowTask` or "
        "derived object. Such input or output is required for deep learning training "
        "task. Custom dataset loader must inherit this class and implements required virtual methods. \n\n"
        "Dataset structure is composed of a dict for each image and some common "
        "metadata with the following specifications (mandatory fields may vary "
        "depending on the training goal).\n\n"
        "   - images (list[dict]): image information and corresponding annotations:\n"
        "       - filename (str): full path of the image file.\n"
        "       - height, width (int): size of the image.\n"
        "       - image_id (int): unique image identifier.\n"
        "       - annotations (list[dict]): each dict corresponds to annotations of one instance in this image.\n"
        "           - bbox (list[float]): x, y, width, height of the bounding box.\n"
        "           - category_id (int): integer representing the category label.\n"
        "           - segmentation_poly (list[list[float]]): list of polygons, one for each connected component.\n"
        "           - keypoints (list[float]).\n"
        "           - iscrowd (boolean): whether the instance is labelled as a crowd region (COCO).\n"
        "       - segmentation_masks (numpy array [N, H, W]).\n"
        "       - instance_seg_masks_file: full path of the ground truth instance segmentation image file.\n"
        "       - semantic_seg_masks_file: full path of the ground truth semantic segmentation image file.\n"
        "   - metadata (dict): key-value mapping that contains information that's shared among the entire dataset:\n"
        "       - category_names (dict{id (int): name (str)]).\n"
        "       - category_colors (list[tuple(r,g,b)]).\n"
        "       - keypoint_names (list[str]).\n"
        "       - keypoint_connection_rules (list[tuple(str, str, (r,g,b))]): each tuple specifies a pair of keypoints that "
        "are connected and the color to use for the line between them.\n\n";

constexpr auto _ctor1DatasetIODocString =
        "Construct a CDatasetIO object specifying the input or output name.\n\n"
        "Args:\n\n"
        "   name (str): custom name\n\n";

constexpr auto _ctor2DatasetIODocString =
        "Construct a CDatasetIO object specifying the name and the source format.\n\n"
        "Args:\n\n"
        "   name (str): custom name\n\n"
        "   source_format (str): unique string identifier\n\n";

constexpr auto _getImagePathsDocStr =
        "Virtual method to reimplement, return the file path list of all images contained in the dataset.\n\n"
        "Returns:\n\n"
        "   str[]: path list\n\n";

constexpr auto _getCategoriesDocStr =
        "Virtual method to reimplement, return the categories of the dataset.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.MapIntStr`: list of categories\n\n";

constexpr auto _getCategoryCountDocStr =
        "Virtual method to reimplement, return the number of instance categories (ie classes) in the dataset.\n\n"
        "Returns:\n\n"
        "   int: number of categories\n\n";

constexpr auto _getMaskPathDocStr =
        "Virtual method to reimplement, return the path of the segmentation mask associated with the given image contained in the dataset.\n\n"
        "Args:\n\n"
        "   image_path (str): path of the associated image in the dataset\n\n"
        "Returns:\n\n"
        "   str: mask path or empty string if image does not exist\n\n";

constexpr auto _getGraphicsAnnotationsDocStr =
        "Virtual method to reimplement, return the list of graphics items corresponding to dataset annotations for the given image.\n\n"
        "Args:\n\n"
        "   image_path (str): path of the associated image in the dataset\n\n"
        "Returns:\n\n"
        "   list of :py:class:`~ikomia.core.pycore.CGraphicsItem` or derived: graphics items\n\n";

constexpr auto _getSourceFormatDocStr =
        "Get the source format of the dataset.\n\n"
        "Return:\n\n"
        "   str: source format string identifier (lowercase)\n\n";

constexpr auto _saveDocStr =
        "Virtual method to reimplement, save dataset structure as JSON.\n\n"
        "Args:\n\n"
        "   str: file path\n\n";

constexpr auto _loadDocStr =
        "Virtual method to reimplement, load dataset structure from JSON.\n\n"
        "Args:\n\n"
        "   str: file path\n\n";

constexpr auto _datasetIOToJsonDocStr =
        "Return input/output data in JSON formatted string. Must be reimplemented and should manage"
        "the common option to set the JSON format. It can be ['json_format', 'compact'] or ['json_format', 'indented'].\n\n"
        "Args:\n\n"
        "   list of str: format-specific options encoded as pairs (option_name, option_value)\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _datasetIOFromJsonDocStr =
        "Set input/output data from JSON formatted string. Must be reimplemented\n\n"
        "Args:\n\n"
        "   str: data as JSON formatted string\n\n";

//--------------------//
//----- CArrayIO -----//
//--------------------//
constexpr auto _arrayIODocString =
        "Define multi-dimensional array as input or output of a task. "
        "A CArrayIO instance can be added as input or output to a :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived object. "
        "The internal image data structure is a numpy array.\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTaskIO`.\n\n";

constexpr auto _ctor1ArrayIODocString =
        "Construct a CArrayIO instance with the given array.\n\n"
        "Args:\n\n"
        "   name (str): custom name (to give insights to end user)\n\n";

constexpr auto _ctor2ArrayIODocString =
        "Construct a CArrayIO instance with the given array.\n\n"
        "Args:\n\n"
        "   array (Numpy array): nd array\n\n"
        "   name (str): custom name (to give insights to end user)\n\n";

constexpr auto _clearArrayDataDocString =
        "Clear array so that it becomes empty.\n\n";

constexpr auto _getArrayDocString =
        "Get the array data.\n\n"
        "Returns:\n\n"
        "   Numpy array: nd array\n\n";

constexpr auto _getArrayUnitElementCountDocString =
        "Get the number of unit elements when the data is processed. "
        "The number of unit elements is used to calculate the number of progress steps needed to perform a task. "
        "Ikomia Studio consider array as a whole entity.\n\n"
        "Returns:\n\n"
        "   int: number of unit element to process (always 1).\n\n";

constexpr auto _isArrayDataAvailableDocString =
        "Check whether the input/output has valid array.\n\n"
        "Returns:\n\n"
        "   bool: True if array is not empty, False otherwise.\n\n";

constexpr auto _setArrayDocString =
        "Set the array data\n\n"
        "Args:\n\n"
        "   array (Numpy array): nd array\n\n";

//------------------------------//
//----- CObjectDetectionIO -----//
//------------------------------//
constexpr auto _objDetectionDocString =
        "Store single object detection information (class properties): label, confidence, box and color. "
        "It is used within workflow input/output of type :py:class:`~ikomia.dataprocess.pydataprocess.CObjectDetectionIO`.\n\n";

constexpr auto _objDetectionIODocString =
        "Define input or output managing common information extracted by object detection task. "
        "Such task are able to automatically detect objects in image and get class label, confidence and "
        "bounding box for each one. For each object, information are stored in a "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CObjectDetection` instance. Among others, algorithms "
        "like FasterRCNN, RetinaNet and YOLO series are object detection tasks.\n\n";

constexpr auto _getObjectCountDocString =
        "Get the number of detected objects.\n\n"
        "Returns:\n\n"
        "   int: object count\n\n";

constexpr auto _getObjectDocString =
        "Get object information at a given index.\n\n"
        "Args:\n\n"
        "   index (int): object index\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CObjectDetection`: object information instance\n\n";

constexpr auto _getObjectsDocString =
        "Get all detected objects.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CObjectDetection` list: detected objets\n\n";

constexpr auto _getGraphicsIODocString =
        "Get internal graphics output instance. It stores graphics items representing boxes and labels of "
        "detected objects in image.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput`: graphics output instance\n\n";

constexpr auto _getBlobMeasureIODocString =
        "Get internal blob measure output instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CBlobMeasureIO`: blob measure output instance\n\n";

constexpr auto _initObjDetectIODocString =
        "Initialisation step to set associated task (name) and reference image. The reference image is the task output index "
        "where the graphics information (label, box) will be displayed as an overlay layer.\n\n"
        "Args:\n\n"
        "   task_name (str): task that contains the output\n\n"
        "   ref_image_index (int): zero-based index of the output containing the reference image\n\n";

constexpr auto _addObjectDocString =
        "Add detected object with bounding box.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): class label\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   box_x (double): left coordinate of object bounding box\n\n"
        "   box_y (double): top coordinate of object bounding box\n\n"
        "   box_width (double): width of object bounding box\n\n"
        "   box_height (double): height of object bounding box\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _addObject2DocString =
        "Add detected object with oriented bounding box.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): class label\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   cx (double): x-coordinate of object bounding box center\n\n"
        "   cy (double): y-coordinate of object bounding box center\n\n"
        "   width (double): width of object bounding box\n\n"
        "   height (double): height of object bounding box\n\n"
        "   angle (double): angle w.r.t horizontal axis of object bounding box\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _objDetectLoadDocString =
        "Load object detection input/output from JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _objDetectSaveDocString =
        "Save object detection input/output to JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _objDetectToJsonDocString =
        "Return input/output data in JSON formatted string. JSON format options can be set, possible values are:\n\n"
        "- ['json_format', 'compact'] (**default**)\n"
        "- ['json_format', 'indented']\n\n"
        "Args:\n\n"
        "   list of str: format-specific options encoded as pairs [option_name, option_value]\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _objDetectFromJsonDocString =
        "Set input/output data from JSON formatted string.\n\n"
        "Args:\n\n"
        "   str: data as JSON formatted string\n\n";

//--------------------------//
//----- CInstanceSegIO -----//
//--------------------------//
constexpr auto _instanceSegDocString =
        "Store single instance segmentation information (class properties): class index, label, confidence, box, mask and color. "
        "It is used within workflow input/output of type :py:class:`~ikomia.dataprocess.pydataprocess.CInstanceSegIO`.\n\n";

constexpr auto _instanceSegIODocString =
        "Define input or output managing common information extracted by instance segmentation task. "
        "Such task are able to automatically detect objects and their shapes in image. "
        "It gets class label, confidence, bounding box and mask for each one. "
        "For each object, information is stored in a :py:class:`~ikomia.dataprocess.pydataprocess.CInstanceSegmentation` instance. "
        "Among others, algorithms like MaskRCNN, Yolact, SparseInst are instance segmentation tasks.\n\n";

constexpr auto _getInstanceCountDocString =
        "Get the number of segmented instances.\n\n"
        "Returns:\n\n"
        "   int: object count\n\n";

constexpr auto _getInstanceDocString =
        "Get segmented instance information at a given index.\n\n"
        "Args:\n\n"
        "   index (int): object index\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CInstanceSegmentation`: segmented instance information\n\n";

constexpr auto _getInstancesDocString =
        "Get all segmented instances.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CInstanceSegmentation` list: segmented instances\n\n";

constexpr auto _getMergeMaskDocString =
        "Get an image array where all segmented masks are merged into a single grayscale mask: "
        "one graylevel value for each class. This image output is typically used for display, Ikomia Studio "
        "uses it to create an overlay color mask on top of the reference image.\n\n"
        "Returns:\n\n"
        "   2D numpy array (uint8): grayscale mask\n\n";

constexpr auto _initInstanceSegIODocString =
        "Initialisation step to set associated task (name), reference image and mask size. "
        "The reference image is the task output index where the graphics information (label, box) will be displayed "
        "as an overlay layer.\n\n"
        "Args:\n\n"
        "   taskName (str): task that contains the output\n\n"
        "   refImageIndex (int): zero-based index of the output containing the reference image\n\n"
        "   width (int): mask width\n\n"
        "   height (int): mask height\n\n";

constexpr auto _addInstanceDocString =
        "Add segmented instance to the input/output.\n\n"
        "Args:\n\n"
        "   id (int): instance identifier\n\n"
        "   type (int): segmentation instance type (0:THING - 1:STUFF)\n\n"
        "   classIndex (int): index of the associated class\n\n"
        "   label (str): class label\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   boxX (double): left coordinate of object bounding box\n\n"
        "   boxY (double): top coordinate of object bounding box\n\n"
        "   boxWidth (double): width of object bounding box\n\n"
        "   boxHeight (double): height of object bounding box\n\n"
        "   mask (numpy array): binary mask\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _instanceSegLoadDocString =
        "Load instance segmentation input/output for JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _instanceSegSaveDocString =
        "Save instance segmentation input/output to JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _instanceSegToJsonDocString =
        "Return input/output data in JSON formatted string. JSON format options can be set, possible values are:\n\n"
        "- ['json_format', 'compact'] (**default**)\n"
        "- ['json_format', 'indented']\n\n"
        "- ['image_format', 'jpg'](for the mask - **default**)\n\n"
        "- ['image_format', 'png']\n\n"
        "Args:\n\n"
        "   list of str: format-specific options encoded as pairs [option_name, option_value]\n\n"
        "Returns:\n\n"
        "   string: JSON formatted string\n\n";

constexpr auto _instanceSegFromJsonDocString =
        "Set input/output data from JSON formatted string.\n\n"
        "Args:\n\n"
        "   str: data as JSON formatted string\n\n";

//--------------------------//
//----- CSemanticSegIO -----//
//--------------------------//
constexpr auto _semanticSegIODocString =
        "Define input or output managing common information extracted by semantic segmentation task. "
        "Such task are able to automatically set a class for each pixel of an image. "
        "Thus, this input/output stores labelled image, class names and class colors. "
        "Among others, algorithms like DeepLabV3+, UNet or TransUNet are semantic segmentation tasks.\n\n";

constexpr auto _getLegendDocString =
        "Get legend image with class labels and colors.\n\n"
        "Returns:\n\n"
        "   Numpy array: legend image\n\n";

constexpr auto _getMaskDocString =
        "Get labelled image (ie graylevel) where each pixel has a specific value corresponding to its class. "
        "You can find the correspondence between pixel value and class name with :py:meth:`ikomia.dataprocess.pydataprocess.CSemanticIO.get_class_names`.\n\n"
        "Returns:\n\n"
        "   Numpy array: grayscale mask\n\n";

constexpr auto _getPolygonsDocString =
        "Get polygons of connected components extracted from segmentation mask.\n\n"
        "Returns:\n\n"
        "   list of :py:class:`~ikomia.core.pycore.CGraphicsItem` based objects\n\n";

constexpr auto _getClassNamesDocString =
        "Get class names list associated with the semantic mask. "
        "Class index in the list corresponds to the pixel value in the mask.\n\n"
        "Returns:\n\n"
        "   list of str: class names\n\n";

constexpr auto _getColorsDocString =
        "Get colors associated with class names.\n\n"
        "Returns:\n\n"
        "   list of int list: colors as [r, g, b] list\n\n";

constexpr auto _setMaskDocString =
        "Set the mask of the semantic segmentation output.\n\n"
        "Args:\n\n"
        "   mask (numpy array): segmentation mask\n\n";

constexpr auto _setClassNamesDocString =
        "Set class names associated with segmentation mask. If colors are not defined, "
        "random colors are automatically generated.\n\n"
        "Args:\n\n"
        "   names (list of str): class names, index in the list must match the pixel value in the mask\n\n";

constexpr auto _setClassColorsDocString =
        "Set class colors associated with segmentation mask. Sizes of names and colors must be equal.\n\n"
        "Args:\n\n"
        "   colors (list of list of int): colors as [r, g, b] list\n\n";

//------------------------//
//----- CKeypointsIO -----//
//------------------------//
constexpr auto _objkeyptsDocString =
        "Store single object keypoints information (class properties): label, confidence, box, color and points. "
        "It is used within workflow input/output of type :py:class:`~ikomia.dataprocess.pydataprocess.CKeypointsIO`.\n\n";

constexpr auto _keyptLinkDocString =
        "Store link information between two keypoints (class properties): starting point index, ending point index, label and color. "
        "It is used within workflow input/output of type :py:class:`~ikomia.dataprocess.pydataprocess.CKeypointsIO`.\n\n";

constexpr auto _keyptsIODocString =
        "Define input or output managing common information extracted by keypoints detection task. "
        "Such task are able to automatically detect objects in image and get keypoints for each one. "
        "For each object, information is stored in a :py:class:`~ikomia.dataprocess.pydataprocess.CObjectKeypoints` instance. "
        "Links between points is also a required information and is stored in a :py:class:`~ikomia.dataprocess.pydataprocess.CKeypointLink` "
        "instance. Among others, algorithms like pose estimation are keypoints detection tasks.\n\n";

constexpr auto _keyptsAddObjDocString =
        "Add detected object and associated keypoints.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): object class label\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   x (double): left coordinate of object bounding box\n\n"
        "   y (double): top coordinate of object bounding box\n\n"
        "   width (double): width of object bounding box\n\n"
        "   height (double): height of object bounding box\n\n"
        "   keypoints (list of tuple): keypoints as list of pairs (index, :py:class:`~ikomia.core.pycore.CPointF`)\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _keyptsGetObjCountDocString =
        "Get the number of detected objects.\n\n"
        "Returns:\n\n"
        "   int: object count\n\n";

constexpr auto _keyptsGetObjDocString =
        "Get object information at a given index.\n\n"
        "Args:\n\n"
        "   index (int): object index\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CObjectKeypoints`: object information instance\n\n";

constexpr auto _keyptsGetObjectsDocString =
        "Get all detected objects.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CObjectKeypoints` list: detected objets\n\n";

constexpr auto _keyptsGetGraphicsIODocString =
        "Get internal graphics output instance. It stores graphics items representing boxes, labels and keypoints of "
        "detected objects in image.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput`: graphics output instance\n\n";

constexpr auto _keyptsGetDataStringIODocString =
        "Get internal data string output instance. It stores keypoint links information.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CDataStringIO`: data string output instance\n\n";

constexpr auto _getKeyptsLinksDocString =
        "Get the global connection scheme of detected keypoints. It consists in a list of links between points with "
        "the following information: starting point index, ending point index, link label and link color. Keypoints list "
        "order is very important: there is a strict equivalence between prediction keypoints index, name index and connection scheme.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CKeypointLink` list\n\n";

constexpr auto _getKeyptsNamesDocString =
        "Get keypoint names. Each keypoint is associated with a label. Order in this list is very important: there is "
        "a strict equivalence between prediction keypoints index, name index and connection scheme.\n\n"
        "Returns:\n\n"
        "   str list: keypoint names\n\n";

constexpr auto _keyptsInitDocString =
        "Initialisation step to set associated task (name) and reference image. The reference image is the task output index "
        "where the graphics information (label, box) will be displayed as an overlay layer.\n\n"
        "Args:\n\n"
        "   task_name (str): task that contains the output\n\n"
        "   ref_image_index (int): zero-based index of the output containing the reference image\n\n";

constexpr auto _setKeyptNamesDocString =
        "Set keypoint names. Each keypoint is associated with a label. Moreover, order in this list is very important as there must "
        "have a strict equivalence between prediction keypoints index, name index and connection scheme.\n\n"
        "Args:\n\n"
        "   names (list of str)\n\n";

constexpr auto _setKeyptLinksDocString =
        "Set the global connection scheme of detected keypoints. It consists in a list of links between points with "
        "the following information: starting point index, ending point index, link label and link color. Keypoints index "
        "are given with respect to prediction model and names structures.\n\n"
        "Args:\n\n"
        "   links (:py:class:`~ikomia.dataprocess.pydataprocess.CKeypointLink` list)\n\n";

constexpr auto _keyptsLoadDocString =
        "Load object keypoints detection input/output from JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _keyptsSaveDocString =
        "Save object keypoints detection input/output to JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

//-------------------//
//----- CTextIO -----//
//-------------------//
constexpr auto _textFieldDocString =
        "Store single text field information (class properties): label, text, confidence, polygon and color. "
        "It is used within workflow input/output of type :py:class:`~ikomia.dataprocess.pydataprocess.CTextIO`.\n\n";

constexpr auto _textIODocString =
        "Define input or output managing common information extracted by OCR task. "
        "Such task are able to automatically detect text fields in image and get class label, text value, confidence and "
        "polygon for each one. For each field, information are stored in a "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CTextField` instance.\n\n";

constexpr auto _getTextFieldCountDocString =
        "Get the number of detected text fields.\n\n"
        "Returns:\n\n"
        "   int: object count\n\n";

constexpr auto _getTextFieldDocString =
        "Get text field information at a given index.\n\n"
        "Args:\n\n"
        "   index (int): field index\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CTextField`: field information instance\n\n";

constexpr auto _getTextFieldsDocString =
        "Get all detected text fields.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CTextField` list: detected fields\n\n";

constexpr auto _textIOGetGraphicsIODocString =
        "Get internal graphics output instance. It stores graphics items representing polygon, labels and "
        "value of detected text fields in image.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsOutput`: graphics output instance\n\n";

constexpr auto _textIOGetDataStringIODocString =
        "Get internal data string output instance. It stores text information.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CDataStringIO`: data string output instance\n\n";

constexpr auto _textIOFinalizeIODocString =
        "Method to call after all fields are added. It will compute data output that will be shown as table "
        "in Ikomia Studio.\n\n";

constexpr auto _addTextFieldBoxDocString =
        "Add detected text fields with bounding box.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): class label\n\n"
        "   text (str): field value\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   box_x (double): left coordinate of object bounding box\n\n"
        "   box_y (double): top coordinate of object bounding box\n\n"
        "   box_width (double): width of object bounding box\n\n"
        "   box_height (double): height of object bounding box\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _addTextFieldPolyDocString =
        "Add detected text fields with polygon.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): class label\n\n"
        "   text (str): field value\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   polygon (list of :py:class:`~ikomia.core.pycore.CPointF`): polygon points coordinates\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _textLoadDocString =
        "Load text field detection input/output from JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

constexpr auto _textSaveDocString =
        "Save text field detection input/output to JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path to JSON file\n\n";

//------------------------//
//----- C2dImageTask -----//
//------------------------//
constexpr auto _imageProcess2dDocString =
        "Implement all basic features used in a task dedicated to 2D image processing. "
        "This class defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image (:py:class:`CImageIO`)\n\n"
        "It is a good starting point to use it as a base class for all task dedicated to 2D images.\n\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTask`.\n\n";

constexpr auto _ctor1ImageProcess2dDocString =
        "Construct C2dImageTask task with or without graphics input.\n\n"
        "Args:\n\n"
        "   has_graphics_input (bool): True, the task manage graphics input. False, the task does not have any graphics input\n";

constexpr auto _ctor2ImageProcess2dDocString =
        "Construct C2dImageTask task with the given name. Default inputs and outputs.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _ctor3ImageProcess2dDocString =
        "Construct C2dImageTask task with the given name, with or without graphics input.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n"
        "   has_graphics_input (bool): True, the task manage graphics input. False, the task does not have any graphics input\n\n";

constexpr auto _setActiveDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.set_active`.\n\n";

constexpr auto _setOutputColorMapDocString =
        "Bind a display color map to an image output. "
        "The color mask is generated from an output mask generated by the task itself (binary or labelled image).\n\n"
        "Args:\n\n"
        "   index (int): zero-based index of the output to be displayed with the color map. The output must be a :py:class:`~ikomia.dataprocess.pydataprocess.CImageIO` or derived.\n\n"
        "   mask_index (int): zero-based index of the output representing the mask used to generate the color overlay.\n\n"
        "   colors (list): list of tuples (r,g,b values) for the color map. If empty, the system generates random colors.\n\n"
        "   reserve_zero (bool): reserve zero pixels of the mask for background so that it will appear transparent in Ikomia Studio.\n\n";

constexpr auto _updateStaticOutputsDocString =
        "Determine output data type automatically from input data types. "
        "Don't forget to call this method in overriden methods. See :py:meth:`~ikomia.core.pycore.CWorkflowTask.update_static_outputs`.\n\n";

constexpr auto _beginTaskRunDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.begin_task_run`.\n\n";

constexpr auto _endTaskRunDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.end_task_run`.\n\n";

constexpr auto _graphicsChangedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.graphics_changed`.\n\n";

constexpr auto _globalInputChangedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.global_input_changed`.\n\n";

constexpr auto _createInputGraphicsMaskDocString =
        "Generate a binary mask image from the task graphics input at the specified index. "
        "The new mask is appended to the internal mask list. "
        "Use :py:meth:`~ikomia.dataprocess.pydataprocess.C2dImageTask.get_graphics_mask` to retrieve the mask.\n\n"
        "Args:\n\n"
        "   index (int): task input index containing graphics information\n\n"
        "   width (int): mask width (should be the width of the source image)\n\n"
        "   height (int): mask height (should be the height of the source image)\n\n";

constexpr auto _createGraphicsMaskDocString =
        "Generate a binary mask image from the given graphics input object. "
        "The new mask is appended to the internal mask list. "
        "Use :py:meth:`~ikomia.dataprocess.pydataprocess.C2dImageTask.get_graphics_mask` to retrieve the mask.\n\n"
        "Args:\n\n"
        "   width (int): mask width (should be the width of the source image)\n\n"
        "   height (int): mask height (should be the height of the source image)\n\n"
        "   graphics (:py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsInput`): graphics items become white blobs in the black background mask\n\n";

constexpr auto _applyGraphicsMaskDocString =
        "Apply the mask generated from graphics to the result image so that only masked areas seems to be processed.\n\n"
        "Args:\n\n"
        "   origin (Numpy array): input image of the task\n\n"
        "   processed (Numpy array): result image of the task\n\n"
        "   index (int): zero-based index of the mask\n\n"
        "Returns:\n\n"
        "   Numpy array: result image\n\n";

constexpr auto _applyGraphicsMaskToBinaryDocString =
        "Apply the mask generated from graphics to the binary source image. "
        "Only white areas on both image and mask are kept in the result image.\n\n"
        "Args:\n\n"
        "   origin (Numpy array): input image of the task\n\n"
        "   processed (Numpy array): result image of the task\n\n"
        "   index (int): zero-based index of the mask\n\n"
        "Returns:\n\n"
        "   Numpy array: result image\n\n";

constexpr auto _getProgressStepsDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.get_progress_steps`.\n\n";

constexpr auto _getGraphicsMaskDocString =
        "Get the binary mask generated from graphics input at position index.\n\n"
        "Args:\n\n"
        "   index (int): zero-based index of the mask\n\n"
        "Returns:\n\n"
        "   Numpy array: binary mask (8 bits - 1 channel)\n\n";

constexpr auto _isMaskAvailableDocString =
        "Check whether a binary mask from graphics input is available at position index.\n\n"
        "Args:\n\n"
        "   index (int): zero-based index of the mask\n\n"
        "Returns:\n\n"
        "   bool: True if mask is available, False otherwise\n\n";

constexpr auto _runDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.run`.\n\n";

constexpr auto _stopDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.stop`.\n\n";

constexpr auto _forwardInputImageDocString =
        "Forward input image at position input_index to output at position output_index.\n\n"
        "Args:\n\n"
        "   input_index (int): zero-based index of the input\n\n"
        "   output_index (int): zero-based index of the output\n\n";

constexpr auto _emitAddSubProgressSteps =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.emit_add_sub_progress_steps`.\n\n";

constexpr auto _emitStepProgressDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.emit_step_progress`.\n\n";

constexpr auto _emitGraphicsContextChangedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.emit_graphics_context_changed`.\n\n";

constexpr auto _emitOutputChangedDocString =
        "See :py:meth:`~ikomia.core.pycore.CWorkflowTask.emit_output_changed`.\n\n";

constexpr auto _executeActionsDocString =
        "Method called when a specific action is requested from the associated widget (see :py:meth:`~ikomia.core.pycore.CWorkflowTaskWidget.emit_send_process_action`).\n\n"
        "Args:\n\n"
        "   action (int): action code\n\n";

//-----------------------------------//
//----- C2dImageInteractiveTask -----//
//-----------------------------------//
constexpr auto _interactiveImageProcess2d =
        "Add user interactions capability to a 2D image process task. "
        "The class implements a user interaction mechanism through the use of dedicated graphics layer. "
        "When a C2dImageInteractiveTask instance is active, the system automatically activates this "
        "internal graphics layer on which the user can interact by drawing items (points, lines, polygons...). "
        "Every changes on this layer are then notified to this class, and actions can be implemented accordingly. "
        "The class could be used for example to handle interactive segmentation with color picker.\n\n"
        "Derived from :py:class:`C2dImageTask`.\n\n";

constexpr auto _ctorInteractiveImageProcessDocString =
        "Construct C2dImageTask task with the given name. Same inputs and outputs as :py:class:`C2dImageTask`.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _setActiveInteractiveDocString =
        "Make the task and its graphics interaction layer active or inactive.\n\n"
        "Args:\n\n"
        "   is_active (bool): True or False\n\n";

constexpr auto _graphicsChangedInteractiveDocString =
        "Notify that the graphics interaction layer has changed.\n\n";

constexpr auto _globalInputChangedInteractiveDocString =
        "Notify that the workflow input has changed. "
        "The graphics interaction layer is updated. "
        "Don't forger to call this method if you override it in a derived class.\n\n"
        "Args:\n\n"
        "   is_new_sequence (bool): indicate if new input is a new sequence (ex: new frame of the same video is not a new sequence).\n\n";

constexpr auto _getInteractionMaskDocString =
        "Get the binary mask generated from the graphics interaction layer.\n\n"
        "Returns:\n\n"
        "   Numpy array: binary mask (8 bits - 1 channel)\n\n";

constexpr auto _getBlobsDocString =
        "Get the list of connected components extracted from the binary interaction mask.\n\n"
        "Returns:\n\n"
        "   List of 2D point list\n\n";

constexpr auto _createInteractionMaskDocString =
        "Generate a binary mask (stored internally) from the graphics interaction layer.\n\n"
        "Args:\n\n"
        "   width (int): mask width (should be the width of the source image)\n\n"
        "   height (int): mask height (should be the height of the source image)\n\n";

constexpr auto _computeBlobsDocString =
        "Generate the list of connected components from the binary mask. "
        "Use :py:meth:`get_blobs` to retrieve it.\n\n";

constexpr auto _clearInteractionLayerDocString =
        "Clear all graphics items in the interaction layer.\n\n";

//----------------------//
//----- CVideoTask -----//
//----------------------//
constexpr auto _videoProcessDocString =
        "Add video specific features. "
        "This class defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- video (:py:class:`CVideoIO`)\n"
        "- graphics layer (not accessible yet in Python API)\n\n"
        "Outputs:\n\n"
        "- video (:py:class:`CVideoIO`)\n\n"
        "It should be the base class for all task dedicated to video processing.\n\n"
        "Derived from :py:class:`C2dImageTask`.\n\n";

constexpr auto _ctorVideoProcessDocString =
        "Construct CVideoTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _notifyVideoStartDocString =
        "Notify that the video has started.\n\n"
        "Args:\n\n"
        "   frame_count (int): total number of frames\n\n";

constexpr auto _notifyVideoEndDocString =
        "Notify that the end of the video is reached.\n\n";

constexpr auto _beginTaskRunVideoDocString =
        "Perform video specific checks before the process is started.\n\n";

//------------------------//
//----- CVideoOFTask -----//
//------------------------//
constexpr auto _videoProcessOFDocString =
        "Add optical flow specific features for methods based on OpenCV framework. "
        "This class handles the persistent data required to compute classical optical flow algorithms. "
        "It also offers a way to display the computed flow as a color map.\n\n"
        "Derived from :py:class:`CVideoTask`.\n";

constexpr auto _ctorVideoProcessOFDocString =
        "Construct CVideoOFTask object with the given name. "
        "Same inputs and outputs as :py:class:`CVideoTask`.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _beginTaskRunVideoOFDocString =
        "See :py:meth:`CVideoTask.begin_task_run`.\n";

constexpr auto _drawOptFlowMapDocString =
        "Draw optical flow vectors in the optical flow map (image).\n\n"
        "Args:\n\n"
        "   flow (Numpy array): source optical flow (store vector coordinates of the optical flow)\n\n"
        "   flow map (Numpy array): color map of the optical flow\n\n"
        "   step (int): sampling step\n\n"
        "Returns:\n\n"
        "   Numpy array: full visualization image (color map + vectors)\n\n";

constexpr auto _flowToDisplayDocString =
        "Generate a displayable image of the optical flow color map.\n\n"
        "Args:\n\n"
        "   flow (Numpy array): source optical flow color map\n\n"
        "Returns:\n\n"
        "   Numpy array: displayable color image (3 channels)\n\n";

//------------------------------//
//----- CVideoTrackingTask -----//
//------------------------------//
constexpr auto _videoProcessTrackingDocString =
        "Add specific features for tracking task. "
        "This class handles the graphics input to extract the region of interest to track. "
        "This region is limited to rectangle region. "
        "It also generates the tracking results as a binary mask and a dedicated graphics layer.\n\n"
        "Derived from :py:class:`CVideoTask`.\n\n";

constexpr auto _ctorVideoTrackingDocString =
        "Construct CVideoTrackingTask object with the given name.\n\n"
        "Inputs: same as :py:class:`CVideoTask`.\n\n"
        "Outputs:\n\n"
        "- binary mask (video frame)\n"
        "- original image (video frame)\n"
        "- graphics layer\n"
        "- dedicated measures\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\\n";

constexpr auto _setRoiToTrackDocString =
        "Extract region of interest to track from graphics input.\n\n";

constexpr auto _manageOutputsDocString =
        "Fill all the outputs generated by tracking algorithm.\n\n"
        "- Output #1: binary mask of the tracked region\n"
        "- Output #2: original frame with tracked graphics item (overlay)\n"
        "- Output #3: graphics layer (tracked item)\n"
        "- Output #4: tracked region coordinates\n\n";

//-------------------------//
//----- CDnnTrainTask -----//
//-------------------------//
constexpr auto _dnnTrainProcessDocString =
        "Internal use only. Use the pure Python implementation :py:class:`~ikomia.dnn.dnntrain.TrainProcess` "
        "to create training algorithms.\n\n";

constexpr auto _ctor1DnnTrainProcessDocString =
        "Internal use only\n\n";

constexpr auto _ctor2DnnTrainProcessDocString =
        "Internal use only\n\n";

constexpr auto _enableMlflowDocString =
        "Enable or disable automatic display of MLflow dashboard when training starts. "
        "Dashboard is opened just once except if a new call to this function is made before "
        "a new training job starts.\n"
        "Default: enable.\n\n"
        "Args:\n\n"
        "   enable (boolean): True or False\n\n";

constexpr auto _enableTensorboardDocString =
        "Enable or disable automatic launch of Tensorboard dashboard when training starts. "
        "Dashboard is opened just once except if a new call to this function is made before "
        "a new training job starts.\n"
        "Default: enable.\n\n"
        "Args:\n\n"
        "   enable (boolean): True or False\n\n";

//-------------------------------//
//----- CClassificationTask -----//
//-------------------------------//
constexpr auto _classifTaskDocString =
        "Base class for classification task in Computer Vision. "
        "It defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image IO (:py:class:`CImageIO`): by default source image is forwarded.\n"
        "- object detection IO (:py:class:`CObjectDetectionIO`): filled if input graphics items are passed. "
        "In this case, classification is computed for each individual object.\n"
        "- graphics output (:py:class:`CGraphicsOutput`): text item with top-1 class if classification is computed on whole image.\n"
        "- data output (:py:class:`CDataStringIO`): sorted list of class scores if classification is computed on whole image.\n\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.C2dImageTask`.\n\n";

constexpr auto _ctorClassifDocString =
        "Construct CClassificationTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _classifAddObjectDocString =
        "Add classification result for individual object. See :py:meth:`get_input_objects` and "
        ":py:meth:`get_object_sub_image` for more information.\n\n"
        "Args:\n\n"
        "   graphics_item (:py:class:`~ikomia.core.pycore.CGraphicsItem` based object)\n\n"
        "   class_index (int): index is used to retrieve class name\n\n"
        "   confidence (float): confidence score of top-1 class\n\n";

constexpr auto _classifGetNamesDocString =
        "Get class names. Call :py:meth:`read_class_names` to populate names from text file.\n\n"
        "Returns:\n\n"
        "   str list: class names\n\n";

constexpr auto _classifGetInputObjectsDocString =
        "Get input graphics items on which classification can be computed individually. One can iterate over this list "
        "to compute classification for each object. Use :py:meth:`get_object_sub_image` to retieve object ROI image and "
        ":py:meth:`add_object` to store classification result.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CGraphicsItem` based objects: graphics items\n\n";

constexpr auto _classifGetObjectSubImageDocString =
        "Get ROI image for the given graphics item. We use the bounding rect property of "
        ":py:class:`~ikomia.core.pycore.CGraphicsItem` to compute ROI. Input graphics items can be retrieved "
        "with :py:meth:`get_input_objects`. Classification can then be computed on the ROI to get individual object class."
        "Args:\n\n"
        "   graphics item (:py:class:`~ikomia.core.pycore.CGraphicsItem` based object)\n\n"
        "Returns:\n\n"
        "   2D Numpy array: ROI image\n\n";

constexpr auto _classifGetObjectsResultsDocString =
        "Get classification results when applied on individual objects (input graphics items). Results are "
        "given as a :py:class:`CObjectDetectionIO` instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`CObjectDetectionIO`: classification results\n\n";

constexpr auto _classifGetWholeImageResultsDocString =
        "Get classification results when applied on whole image (no input graphics items given). It gives "
        "a sorted list of tuple storing class name and confidence.\n\n"
        "Returns:\n\n"
        "   list of tuples (name, confidence): classification results\n\n";

constexpr auto _classifGetImgWithGraphicsDocString =
        "Get visualization image where all extracted information are embedded (graphics items).\n\n"
        "Returns:\n\n"
        "   2D Numpy array: visualization image\n\n";

constexpr auto _classifIsWholeImageDocString =
        "Check whether input graphics items are given for individual classification.\n\n"
        "Returns:\n\n"
        "   bool: True if no input graphics items are given (whole image classification), False otherwise\n\n";

constexpr auto _classifReadClassNamesDocString =
        "Populate class names from the given text file (one line per class).\n\n"
        "Args:\n\n"
        "   path (str): path to class names definition file\n\n";

constexpr auto _classifSetColorsDocString =
        "Set colors associated with class names. The given list must have the same size as names list. "
        "If not provided, random colors are generated while populating the name list "
        "(:py:meth:`read_class_names`).\n\n"
        "Args:\n\n"
        "   colors (list of list: r, g, b integer values in range [0, 255])\n\n";

constexpr auto _classifSetNamesDocString =
        "Set class names. The function generate associated random colors if "
        "none is defined.\n\n"
        "Args:\n\n"
        "   names (list of str)\n\n";

constexpr auto _classifSetWholeImageResultsDocString =
        "Set whole image classification results.\n\n"
        "Args:\n\n"
        "   names (str list): sorted list with respect to confidence score\n\n"
        "   confidences (str list): sorted list (descending)\n\n";

//--------------------------------//
//----- CObjectDetectionTask -----//
//--------------------------------//
constexpr auto _objDetTaskDocString =
        "Base class for object detection task in Computer Vision. "
        "It defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image IO (:py:class:`CImageIO`): by default source image is forwarded.\n"
        "- object detection IO (:py:class:`CObjectDetectionIO`)\n\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.C2dImageTask`.\n\n";

constexpr auto _ctorObjDetectDocString =
        "Construct CObjectDetectionTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _objDetAddObject1DocString =
        "Add detected object result localized in image through regular bounding box.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   class_index (int): index of the associated class\n\n"
        "   confidence (float): confidence of the prediction \n\n"
        "   x (float): left coordinate of object bounding box\n\n"
        "   y (float): top coordinate of object bounding box\n\n"
        "   width (float): width of object bounding box\n\n"
        "   height (float): height of object bounding box\n\n";

constexpr auto _objDetAddObject2DocString =
        "Add detected object result localized in image through oriented bounding box.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   label (str): class label\n\n"
        "   confidence (double): prediction confidence\n\n"
        "   cx (double): x-coordinate of object bounding box center\n\n"
        "   cy (double): y-coordinate of object bounding box center\n\n"
        "   width (double): width of object bounding box\n\n"
        "   height (double): height of object bounding box\n\n"
        "   angle (double): angle w.r.t horizontal axis of object bounding box\n\n"
        "   color (int list - rgba): display color\n\n";

constexpr auto _objDetectGetResultsDocString =
        "Get object detection results as a :py:class:`CObjectDetectionIO` instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`CObjectDetectionIO`: object detection results\n\n";

//-------------------------------------//
//----- CSemanticSegmentationTask -----//
//-------------------------------------//
constexpr auto _semSegTaskDocString =
        "Base class for semantic segmentatio task in Computer Vision. It consists in "
        "labelling each pixel of input image with a class. Common outputs for such task are "
        "graylevel mask and color labelled image for visualization. "
        "It defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image IO (:py:class:`CImageIO`): by default source image is forwarded.\n"
        "- semantic segmentation IO (:py:class:`CSemanticSegmentationIO`)\n\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.C2dImageTask`.\n\n";

constexpr auto _ctorSemSegDocString =
        "Construct CSemanticSegmentationTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _semSegGetResultsDocString =
        "Get semantic segmentation results as a :py:class:`CSemanticSegmentationIO` instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`CSemanticSegmentationIO`: semantic segmentation data\n\n";

constexpr auto _semSegGetImgWithMaskDocString =
        "Get a visualization image composed by original input image and colored segmentation mask. "
        "A transparency factor is applied to see both information.\n\n"
        "Returns:\n\n"
        "   2D numpy array (3 channels): color mask image\n\n";

constexpr auto _semSegGetImgWithGraphicsDocString =
        "Get visualization image where extracted outlines from connected components are embedded (graphics items).\n\n"
        "Returns:\n\n"
        "   2D Numpy array: visualization image\n\n";

constexpr auto _semSegGetImgWithMaskAndGraphicsDocString =
        "Get a visualization image composed by original input image, colored segmentation mask and "
        "embedded graphics (outlines of connected components). A transparency factor is applied between original "
        "image and mask.\n\n"
        "Returns:\n\n"
        "   2D numpy array (3 channels): visualization image\n\n";

constexpr auto _semSegGetMaskDocString =
        "Set the segmentation mask computed from the input image.\n\n"
        "Args:\n\n"
        "   mask (2D - 1 channel numpy array): segmentation mask\n\n";

//-------------------------------------//
//----- CInstanceSegmentationTask -----//
//-------------------------------------//
constexpr auto _instanceSegTaskDocString =
        "Base class for instance segmentation task in Computer Vision. It consists in "
        "detecting object instances of various classes and compute pixel mask of each instance. "
        "Common outputs for such task are bounding boxes and binary masks for each instance "
        "and color labelled image for visualization. "
        "It defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image IO (:py:class:`CImageIO`): by default source image is forwarded.\n"
        "- instance segmentation IO (:py:class:`CInstanceSegmentationIO`)\n\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.C2dImageTask`.\n\n";

constexpr auto _ctorInstanceSegDocString =
        "Construct CInstanceSegmentationTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _instanceSegAddInstanceDocString =
        "Add segmented instance result localized in image through regular bounding box and mask.\n\n"
        "Args:\n\n"
        "   id (int): instance identifier\n\n"
        "   type (int): segmentation instance type (0:THING - 1:STUFF)\n\n"
        "   class_index (int): index of the associated class\n\n"
        "   confidence (float): confidence of the prediction \n\n"
        "   x (float): left coordinate of object bounding box\n\n"
        "   y (float): top coordinate of object bounding box\n\n"
        "   width (float): width of object bounding box\n\n"
        "   height (float): height of object bounding box\n\n"
        "   mask (numpy array): binary mask\n\n";

constexpr auto _instanceSegGetResultsDocString =
        "Get instance segmentation results as a :py:class:`CInstanceSegmentationIO` instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`CInstanceSegmentationIO`: semantic segmentation data\n\n";

constexpr auto _instanceSegGetImgWithMaskAndGraphicsDocString =
        "Get a visualization image composed by original input image, colored segmentation mask and "
        "embedded graphics (object detection). A transparency factor is applied between original "
        "image and mask.\n\n"
        "Returns:\n\n"
        "   2D numpy array (3 channels): visualization image\n\n";

//----------------------------------//
//----- CKeypointDetectionTask -----//
//----------------------------------//
constexpr auto _keyDetTaskDocString =
        "Base class for keypoint detection task in Computer Vision. "
        "It defines a task with the following properties:\n\n"
        "Inputs:\n\n"
        "- image (:py:class:`CImageIO`)\n"
        "- graphics (:py:class:`CGraphicsInput`)\n\n"
        "Outputs:\n\n"
        "- image IO (:py:class:`CImageIO`): by default source image is forwarded.\n"
        "- keypoints detection IO (:py:class:`CKeypointsIO`)\n\n"
        "Derived from :py:class:`~ikomia.dataprocess.pydataprocess.C2dImageTask`.\n\n";

constexpr auto _ctorKeyDetDocString =
        "Construct CKeypointDetectionTask object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): task name, must be unique\n\n";

constexpr auto _keyDetAddObjectDocString =
        "Add detected object with its associated keypoints.\n\n"
        "Args:\n\n"
        "   id (int): object identifier\n\n"
        "   class_index (int): index of the object class\n\n"
        "   confidence (float): confidence of the prediction\n\n"
        "   x (float): left coordinate of object bounding box\n\n"
        "   y (float): top coordinate of object bounding box\n\n"
        "   width (float): width of object bounding box\n\n"
        "   height (float): height of object bounding box\n\n"
        "   keypoints (list of tuple): keypoints as list of pairs (index, :py:class:`~ikomia.core.pycore.CPointF`)\n\n";

constexpr auto _keyDetGetResultsDocString =
        "Get keypoint detection results as a :py:class:`CKeypointsIO` instance.\n\n"
        "Returns:\n\n"
        "   :py:class:`CKeypointsIO`: keypoint detection results\n\n";

//---------------------------//
//----- CIkomiaRegistry -----//
//---------------------------//
constexpr auto _ikomiaRegistryDocString =
        "Algorithms registry of the Ikomia platform.\n\n"
        "Class that enables to instanciate every algorithm from its name. CIkomiaRegistry objects communicate "
        "automatically with the Ikomia HUB to download algorithm package if necessary. It also include "
        "dedicated function to register new algorithms\n\n"
        ".. Note:: A pure Python derived class is also implemented to add higher level features. See "
        ":py:class:`~ikomia.dataprocess.registry.IkomiaRegistry` for more information.\n\n";

constexpr auto _setPluginsDirDocString =
        "Set directory where Ikomia plugins are stored.\n\n"
        "Args:\n\n"
        "   directory (str)\n\n";

constexpr auto _getPluginsDirDocString =
        "Get the current Ikomia plugins directory.\n\n"
        "Returns:\n\n"
        "   str: full path to Ikomia plugins directory\n\n";

constexpr auto _getAlgorithmsDocString =
        "Get all available algorithms from the Ikomia registry.\n"
        "Before using an algorithm, you must instanciate it from its name using "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CIkomiaRegistry.create_instance`."
        "You can only instanciate algorithms whose name is in the returned list.\n\n"
        "Returns:\n\n"
        "   string: list of algorithm names\n\n";

constexpr auto _getAlgorithmInfoDocString =
        "Get algorithm informations such as description, authors, documentation link...\n\n"
        "Args:\n\n"
        "   name (str): algorithm name\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.dataprocess.pydataprocess.CTaskInfo`: algorithm information\n\n";

constexpr auto _isAllLoadedDocString =
        "Return true if all locally installed algorithms have been loaded at least once.\n\n"
        "Returns:\n\n"
        "   bool: True or False\n\n";

constexpr auto _createInstance1DocString =
        "Instanciate algorithm of the Ikomia registry from its name with default parameters.\n"
        "The full list of available algorithms can be retrieved using "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CIkomiaRegistry.get_algorithms`.\n\n"
        "Args:\n\n"
        "   algorithm name(str): unique name\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived: algorithm instance\n\n";

constexpr auto _createInstance2DocString =
        "Instanciate algorithm of the Ikomia registry from its name with the given parameters.\n"
        "The full list of available algorithms can be retrieved using "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CIkomiaRegistry.get_algorithms`.\n\n"
        "Args:\n\n"
        "   algorithm name(str): unique name\n\n"
        "   parameters object(:py:class:`~ikomia.core.pycore.CWorkflowTaskParam`): associated parameters\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTask` or derived: algorithm instance\n\n";

constexpr auto _registerTaskDocString =
        "Add new algorithm factory to Ikomia registry.\n"
        "Once registered, you are able to instanciate algorithm object and use it inside a workflow.\n\n"
        "Args:\n\n"
        "   factory (:py:class:`~ikomia.dataprocess.pydataprocess.CTaskFactory` based object): task factory\n\n";

constexpr auto _registerIODocString =
        "Add new input/output factory to Ikomia registry.\n"
        "Once registered, you are able to instanciate input or ouput and use it inside a task.\n\n"
        "Args:\n\n"
        "   factory (:py:class:`~ikomia.core.pycore.CTaskIOFactory` based object): I/O factory\n\n";

constexpr auto _loadPluginsDocString =
        "Load locally installed algorithms in the registry. After that, algorithms can be instanciated from their unique names.\n\n";

constexpr auto _loadCppPluginsDocString =
        "Load locally installed C++ algorithms in the registry. After that, C++ algorithms can be instanciated from their unique names.\n\n";

constexpr auto _loadPythonPluginsDocString =
        "Load locally installed Python algorithms in the registry. After that, Python algorithms can be instanciated from their unique names.\n\n";

constexpr auto _loadCppPluginDocString =
        "Load C++ algorithm to Ikomia registry.\n\n"
        "Args:\n\n"
        "   path (str): path to shared library of the plugin\n\n";

constexpr auto _loadPythonPluginDocString =
        "Load Python algorithm to Ikomia registry.\n\n"
        "Args:\n\n"
        "   path (str): path module directory of the algorithm\n\n";

//---------------------//
//----- CWorkflow -----//
//---------------------//
constexpr auto _workflowDocString =
        "Workflow management of Computer Vision tasks.\n"
        "Implement features to create, modify and run graph-based pipeline of "
        ":py:class:`~ikomia.core.pycore.CWorkflowTask` objects or derived. Workflows can be created from scratch "
        "by using :py:class:`~ikomia.dataprocess.registry.IkomiaRegistry` to instanciate and connect task objects. "
        "Workflows can also be loaded from JSON file created with the interactive designer of Ikomia Studio.\n"
        "Derived from :py:class:`~ikomia.core.pycore.CWorkflowTask`.\n\n"
        ".. Note:: A pure Python derived class is also implemented to add higher level features. See "
        ":py:class:`~ikomia.dataprocess.workflow.Workflow` for more information.\n\n";

constexpr auto _ctor1WorkflowDocString =
        "Construct a new workflow object with the given name.\n\n"
        "Args:\n\n"
        "   name (str): workflow name\n";

constexpr auto _ctor2WorkflowDocString =
        "Construct a new workflow object with the given name and Ikomia registry. You should use this constructor "
        "if you intend to instanciate tasks from the Ikomia registry (built-in algorithms and Ikomia HUB).\n\n"
        "Args:\n\n"
        "   name (str): workflow name\n\n"
        "   registry (:py:class:`~ikomia.dataprocess.registry.IkomiaRegistry`): algorithms registry\n\n";

constexpr auto _wfSetInputDocString =
        "Set workflow input at position *index*.\n"
        "If *index* is greater than the input count, the function adds the right number of inputs automatically.\n"
        "Derived class that handles common data type already exists: "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CImageIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CVideoIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CNumericIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CGraphicsInput`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CDatasetIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CPathIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CArrayIO`, "
        ":py:class:`~ikomia.dataprocess.pydataprocess.CBlobMeasureIO`.\n\n"
        "Args:\n\n"
        "   input (:py:class:`~ikomia.core.pycore.CWorkflowTaskIO` object or derived): global input of the workflow\n\n"
        "   index (int): zero-based index\n\n"
        "   new_sequence (bool): True if it is a new input sequence, False if it is just a new frame of a video or "
        "camera stream\n\n";

constexpr auto _wfSetOutputFolderDocString =
        "Set workflow output folder.\n"
        "If auto-save mode is activated (see :py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.set_auto_save`) "
        "outputs of each tasks of the workflow will be saved automatically in this folder. Behind the scene, each task "
        "implements a *save()* function that calls sequentially the *save()* function of all these outputs.\n\n"
        "Args:\n\n"
        "   path (str): path to the desired directory\n\n";

constexpr auto _wfSetAutoSaveDocString =
        "Activate/deactivate auto-save mode. If activated, outputs of each tasks of the workflow will be saved in "
        "the workflow output folder (see :py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.set_output_folder` "
        "to set your custom folder). By default, outputs are saved in *user_folder/Ikomia/Workflows*.\n\n"
        "Args:\n\n"
        "   enable (bool): True or False\n\n";

constexpr auto _wfSetCfgEntryDocString =
        "Set special workflow configuration parameters. It consists of key-value string pairs that aim to adjust the "
        "workflow execution behavior. Possible parameters are:\n\n"
        "- Autosave (boolean - default: False): enable/disable saving outputs to disk for all tasks during execution.\n"
        "- BatchMode (boolean - default: False): enable/disable batch processing.\n"
        "- WholeVideo (boolean - default: False): enable/disable processing all frames of a video.\n"
        "- GraphicsEmbedded (boolean - default: False): enable/disable burning of associated graphics items into output images.\n"
        "- VideoReadTimeout (integer - default: 5s): timeout for reading video or camera frames in milliseconds.\n"
        "- VideoWriteTimeout (integer - default: 1min): timeout for writing video or camera frames in milliseconds.\n\n"
        "Args:\n\n"
        "   key (str): parameter name\n\n"
        "   value (str): parameter value. You have to take care of string conversion.\n\n";

constexpr auto _wfSetExposedParamDocString =
    "Set workflow parameter. Actually, workflow parameter is just task parameter exposed at workflow level.\n\n"
    "Args:\n\n"
    "   name (str): workflow parameter name. It could be different from the source task parameter.\n\n"
    "   value (str): parameter value.\n\n";

constexpr auto _wfGetTaskCountDocString =
        "Get the number of tasks in the workflow.\n\n"
        "Returns:\n\n"
        "   int: task count\n\n";

constexpr auto _wfGetRootIDDocString =
        "Get unique identifier of the root node.\n\n"
        "Returns:\n\n"
        "   int: root node ID\n";

constexpr auto _wfGetTaskIDsDocString =
        "Get the list of all task identifiers.\n"
        "You can then retrieve task object from ID with the function "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.get_task`.\n\n";

constexpr auto _wfGetLastTaskIDDocString =
        "Get the last added task of the workflow.\n"
        "You can then retrieve task object from ID with the function "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.get_task`.\n\n"
        "Returns:\n\n"
        "   int: workflow task id\n\n";

constexpr auto _wfGetTaskDocString =
        "Get the task object from the given ID.\n"
        "Unique task identifiers can be retrieved with the functions "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.get_task_ids` or "
        ":py:meth:`~ikomia.dataprocess.workflow.Workflow.find_task`.\n\n"
        "Returns:\n\n"
        "   :py:class:`~ikomia.core.pycore.CWorkflowTask` object or derived\n\n";

constexpr auto _wfGetParentsDocString =
        "Get parent task identifiers of the task specified by the given identifier.\n"
        "Task connected to the inputs of a given task is designated as parent or source.\n\n"
        "Args:\n\n"
        "   id (int): task identifier on which to get parents\n\n"
        "Returns:\n\n"
        "   int list: parent identifiers\n\n";

constexpr auto _wfGetChildrenDocString =
        "Get child task identifiers of the task specified by the given identifier.\n"
        "Task connected to the outputs of a given task is designated as child or target.\n\n"
        "Args:\n\n"
        "   id (int): task identifier on which to get childs\n\n"
        "Returns:\n\n"
        "   int list: child identifiers\n\n";

constexpr auto _wfGetInEdgesDocString =
        "Get input connections (in-edges) of the task specified by the given identifier.\n"
        "Edge information can then be retrieved by the function "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.get_edge_info`.\n\n"
        "Args:\n\n"
        "   id (int): task identifier\n\n"
        "Returns:\n\n"
        "   int list: edge identifiers\n\n";

constexpr auto _wfGetOutEdgesDocString =
        "Get output connections (out-edges) of the task specified by the given identifier.\n"
        "Edge information can then be retrieved by the function "
        ":py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.get_edge_info`.\n\n"
        "Args:\n\n"
        "   id (int): task identifier\n\n"
        "Returns:\n\n"
        "   int list: edge identifiers\n\n";

constexpr auto _wfGetEdgeInfoDocString =
        "Get connection information (edge) from its unique identifier.\n\n"
        "Args:\n\n"
        "   id (int): edge identifier\n\n"
        "Returns:\n\n"
        "   tuple (int): pair composed by the output index of the source/parent task and the input index of the target/child task\n\n";

constexpr auto _wfGetEdgeSourceDocString =
        "Get source/parent task identifier of a given edge.\n\n"
        "Args:\n\n"
        "   id (int): edge identifier\n\n"
        "Returns:\n\n"
        "   id (int): task identifier\n\n";

constexpr auto _wfGetEdgeTargetDocString =
        "Get target/child task identifier of a given edge.\n\n"
        "Args:\n\n"
        "   id (int): edge identifier\n\n"
        "Returns:\n\n"
        "   id (int): task identifier\n\n";

constexpr auto _wfGetFinalTasks =
        "Get all final or leaf tasks of the workflow.\n\n"
        "Returns:\n\n"
        "   int list: leaf task identifiers\n\n";

constexpr auto _wfGetRootTargetTypesDocString =
        "Get all input data types of the tasks connected to the root node.\n\n"
        "Returns:\n\n"
        "   list of :py:class:`~ikomia.core.pycore.IODataType`: data types\n\n";

constexpr auto _wfGetTotalElapsedTimeDocString =
        "Get the total workflow running time in milliseconds\n\n"
        "Returns:\n\n"
        "   float: elapsed time\n\n";

constexpr auto _wfGetElapsedTimeToDocString =
        "Get the workflow running time in milliseconds from the start to the given task.\n\n"
        "Args:\n\n"
        "   id (int): task identifier\n\n"
        "Returns:\n\n"
        "   float: elapsed time\n\n";

constexpr auto _wfGetRequiredTasks =
        "Get task names required to load and execute the given workflow file.\n\n"
        "Args:\n\n"
        "   path (str): path to the workflow file (JSON)\n\n"
        "Returns:\n\n"
        "   list of str: task names\n\n";

constexpr auto _wfGetLastRunFolder =
        "Get the output folder for the last run. Output folder can be set with :py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.set_output_folder`. "
        "It will be used only if auto-save mode is enabled and a timestamp is automatically added. The complete name is returned by this method.\n\n"
        "Returns:\n\n"
        "   str: output folder path (with timestamp added)\n\n";

constexpr auto _wfGetExposedParamsDocString =
    "Get list of available workflow parameters. For each parameter, parameter name and value are provided.\n\n"
    "Returns:\n\n"
    "   parameters (dict): list of name-value pairs.\n\n";

constexpr auto _getOutputDocString =
    "Get output at position index.\n\n"
    "Args:\n\n"
    "   index (int): zero-based output index\n\n"
    "Returns:\n\n"
    "   :py:class:`~ikomia.core.pycore.CWorkflowTaskIO` based object: output object\n\n";

constexpr auto _getOutputCountDocString =
    "Get the number of outputs.\n\n"
    "Returns:\n\n"
    "   int: outputs count\n\n";

constexpr auto _getOutputDataTypeDocString =
    "Get output data type at position index. "
    "This data type can differ from the original type because it can change at runtime according to the data source.\n\n"
    "Args:\n\n"
    "   index (int): zero-based output index\n\n"
    "Returns:\n\n"
    "   :py:class:`~ikomia.core.pycore.IODataType`: output data type\n\n";

constexpr auto _getOutputsDocString =
    "Get the whole list of outputs. Workflow outputs are actually outputs of task exposed at workflow level.\n\n"
    "Returns:\n\n"
    "   list of :py:class:`~ikomia.core.pycore.CWorkflowTaskIO` based objects: output objects\n\n";

constexpr auto _wfAddInputDocString =
        "Add global input to the workflow.\n\n"
        "Args:\n\n"
        "   input (:py:class:`~ikomia.core.pycore.CWorkflowTaskIO` based object): input object\n\n";

constexpr auto _wfAddTaskDocString =
        "Add new task to the workflow.\n\n"
        "Args:\n\n"
        "   task (:py:class:`~ikomia.core.pycore.CWorkflowTask` or derived)\n\n";

constexpr auto _wfAddParameterDocString =
    "Add a workflow parameter from an existing task parameter. It's a convenient way to expose meaningfull parameters at workflow level.\n\n"
    "Args:\n\n"
    "   name (str): name for the workflow parameter, must be unique\n\n"
    "   description (str): small description about the parameter\n\n"
    "   task_id (int): task identifier of the parameter to expose\n\n"
    "   target_param_name (str): name of the source task parameter\n\n";

constexpr auto _wfAddOutputDocString =
    "Add a workflow output from an existing task output. It's a convenient way to expose meaningfull outputs at workflow level.\n\n"
    "Args:\n\n"
    "   description (str): small description about the output\n\n"
    "   task_id (int): task identifier of the output to expose\n\n"
    "   task_output_index (str): index of task output\n\n";

constexpr auto _wfConnectDocString =
        "Connect tasks.\n\n"
        "Args:\n\n"
        "   source (int): source/parent task identifier\n"
        "   target (int): target/child task identifier\n"
        "   source_index (int): output index of the source task\n"
        "   target_index (int): input index of the target task\n\n";

constexpr auto _wfRemoveInputDocString =
        "Remove global workflow input at the given *index*.\n\n"
        "Args:\n\n"
        "   index (int): zero-based index of the input to remove\n\n";

constexpr auto _wfClearInputsDocString =
        "Remove all inputs of the workflow.\n\n";

constexpr auto _wfClearOutputDataDocString =
        "Clear output data for all the tasks of the workflow.\n\n";

constexpr auto _wfClearDocString =
        "Remove all tasks and connections from the workflow. The workflow is thus empty after that.\n\n";

constexpr auto _wfClearExposedParamsDocString =
    "Remove all exposed parameters of the workflow.";

constexpr auto _wfClearOutputsDocString =
    "Remove all exposed outputs of the workflow";

constexpr auto _wfDeleteTaskDocString =
        "Remove the given task from the workflow. The identifier becomes invalid after this operation.\n\n"
        "Args:\n\n"
        "   id (int): task identifier\n\n";

constexpr auto _wfDeleteEdgeDocString =
        "Remove the given connection/edge from the workflow. The identifier becomes invalid after this operation.\n\n"
        "Args:\n\n"
        "   id (int): edge identifier\n\n";

constexpr auto _wfRunDocString =
        "Launch workflow execution. "
        "Each :py:class:`~ikomia.core.pycore.CWorkflowTask` object or derived must "
        "reimplement the *run()* function that will be called in the right order by the workflow. "
        "Please note that global inputs should be set before calling this function.\n\n";

constexpr auto _wfStopDocString =
        "Stop workflow execution."
        "Each :py:class:`~ikomia.core.pycore.CWorkflowTask` object or derived must "
        "reimplement the *stop()* function that will be called by the workflow. "
        "Depending on the process implementation, stop may not be instantaneous.\n\n";

constexpr auto _wfLoadDocString =
        "Load a workflow previously saved as JSON file. Common usage is to use Ikomia Studio to create your workflow "
        "interactively and without code. With this method, it is very easy and fast to build your pipeline, "
        "visualize outputs on test data and save the workflow. But you can also create your workflow from scratch with "
        "this API and use the dedicated function :py:meth:`~ikomia.dataprocess.pydataprocess.CWorkflow.save`.\n\n"
        "Args:\n\n"
        "   path (str): path of the JSON file\n\n";

constexpr auto _wfUpdateStartTimeDocString =
        "Reset the starting point for the computation of the running time.\n"
        "This feature could be interesting if you want to process a list of images and monitor the time per image.\n\n";

constexpr auto _wfSaveDocString =
        "Save the workflow as a JSON file.\n\n"
        "Args:\n\n"
        "   path (str): path where the JSON is saved\n\n";

constexpr auto _wfExportGraphvizDocString =
        "Export the workflow structure as Graphviz *.dot* file.\n"
        "You can then visualize it with the *dot* command or with Graphviz Python package.\n\n"
        "Args:\n\n"
        "   path (str): path where the *.dot* file is saved\n\n";


#endif // PYDATAPROCESSDOCSTRING_HPP
