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

#ifndef COREDEFINE_HPP
#define COREDEFINE_HPP

#include <vector>
#include <map>
#include <unordered_map>
#include <QString>
#include <QMetaType>
#include "Graphics/CPoint.hpp"

#define RANDOM_COLOR_SEED 5

namespace Ikomia
{
    /**
     * @brief
     *
     */
    enum class DataDimension : int
    {
        NONE,
        X,
        Y,
        Z,
        IMAGE,
        TIME,
        POSITION,
        VOLUME,
        MODALITY,
        STUDY,
        SERIE
    };

    /**
     * @enum Data
     * @brief Input/output data types
     */
    enum class IODataType : int
    {
        IMAGE = 0,                  /**< Image data type */
        IMAGE_BINARY = 16,          /**< Binary image data type: 8bits single channel */
        IMAGE_LABEL = 17,           /**< Label image data type: single channel, 1 graylevel per connected component */
        VIDEO = 1,                  /**< Video from file (AVI, MPEG...) */
        VIDEO_BINARY = 18,          /**< Binary video from file (AVI, MPEG...): 8bits single channel */
        VIDEO_LABEL = 19,           /**< Label video from file (AVI, MPEG...): single channel, 1 graylevel per connected component */
        VOLUME = 2,                 /**< Volume data type */
        VOLUME_BINARY = 14,         /**< Binary volume data type: 8bits single channel */
        VOLUME_LABEL = 15,          /**< Label volume data type: single channel, 1 graylevel per connected component */
        LIVE_STREAM = 3,            /**< Video from stream (camera) */
        LIVE_STREAM_BINARY = 20,    /**< Binary video from stream (camera): 8bits single channel */
        LIVE_STREAM_LABEL = 21,     /**< Label video from stream (camera): single channel, 1 graylevel per connected component */
        INPUT_GRAPHICS = 4,         /**< Graphics data type: graphics layer with graphics items (ellipse, rectangle, text...) */
        OUTPUT_GRAPHICS = 5,        /**< Graphics data type: graphics layer with graphics items (ellipse, rectangle, text...) */
        BLOB_VALUES = 6,            /**< Numeric values from measure (surface, diameter...) computed on connected component */
        NUMERIC_VALUES = 7,         /**< Generic numeric values */
        DESCRIPTORS = 8,            /**< Image descriptors (used for classification, registration...) */
        WIDGET = 9,                 /**< User-defined widget */
        PROJECT_FOLDER = 10,        /**< Ikomia project folder: may contain various data type */
        FOLDER_PATH = 11,           /**< Folder path */
        FILE_PATH = 12,             /**< File path */
        DNN_DATASET = 13,           /**< Dataset used for deep learning, composed with image and annotations */
        ARRAY = 22,                 /**< Multi-dimensional array */
        DATA_DICT = 23,             /**< Python-based IO where data are stored as dict */
        OBJECT_DETECTION = 24,      /**< I/O for object detection management */
        INSTANCE_SEGMENTATION = 25, /**< I/O for instance segmentation management */
        SEMANTIC_SEGMENTATION = 26, /**< I/O for semantic segmentation management */
        KEYPOINTS = 27,             /**< I/O for keypoints management */
        TEXT = 28,                  /**< I/O for text fields management */
        TEXT_STREAM = 33,           /**< I/O for text stream management */
        POSITION = 29,              /**< Position image sequence */
        JSON = 30,                  /**< JSON data */
        SCENE_3D = 31,              /**< I/O for 3d scenes (made up of images, shapes, vector fields, plots, text...) */
        NONE = 32                   /**< Unknown data type */
        // NEXT VALUE = 34
    };

    // Enum class mandatory to avoir name conflict on Windows...
    enum class GraphicsItem : int
    {
        LAYER,
        POINT,
        ELLIPSE,
        RECTANGLE,
        POLYGON,
        COMPLEX_POLYGON,
        POLYLINE,
        TEXT
    };

    /**
     * @enum GraphicsShape
     * @brief Graphics item shape values
     */
    enum class GraphicsShape : int
    {
        SELECTION,          /**< Rectangle selection area */
        POINT,              /**< Point item */
        ELLIPSE,            /**< Ellipse or circle item */
        RECTANGLE,          /**< Rectangle or square item */
        POLYGON,            /**< Polygon item */
        FREEHAND_POLYGON,   /**< Polygon item (free hand) */
        LINE,               /**< Line item (2 points) */
        POLYLINE,           /**< Polyline item */
        FREEHAND_POLYLINE,  /**< Polyline item (free hand) */
        TEXT                /**< Text item */
    };

    enum class DataFileFormat : int
    {
        NONE, TXT, JSON, XML, YAML, CSV,
        BMP, JPG, JP2, PNG, TIF, WEBP,
        AVI, MPEG, MKV, WEBM
    };

    enum class AlgoType : int
    {
        INFER,
        TRAIN,
        DATASET,
        OTHER
    };

    /**
     * @brief
     *
     */
    using Dimension = std::pair<DataDimension,size_t>;
    /**
     * @brief
     *
     */
    using Dimensions = std::vector<Dimension>;
    /**
     * @brief
     *
     */
    using Bounds = std::pair<size_t,size_t>;
    /**
     * @brief
     *
     */
    using DimensionBounds = std::pair<DataDimension, Bounds>;
    /**
     * @brief
     *
     */
    using SubsetBounds = std::vector<DimensionBounds>;
    /**
     * @brief
     *
     */
    using DimensionIndices = Dimensions;

    constexpr double _pi = 3.14159265358979323846;

    using MapIntStr = std::map<int, std::string>;
    using MapString = std::map<std::string, std::string>;
    using UMapString = std::unordered_map<std::string, std::string>;
    using PairString = std::pair<std::string, std::string>;
    using VectorString = std::vector<std::string>;
    using VectorPairString = std::vector<std::pair<std::string, std::string>>;
    using VectorPairQString = std::vector<std::pair<QString, QString>>;
    using CColor = std::vector<uchar>;
    using Keypoint = std::pair<int, CPointF>;
}

using namespace Ikomia;

#endif // COREDEFINE_HPP
