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

#ifndef DATAUTILS_HPP
#define DATAUTILS_HPP

#include <QObject>
#include <QDir>
#include <QSqlQuery>
#include <QPainter>
#include <QRandomGenerator>
#include "UtilsTools.hpp"
#include "Main/CoreDefine.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "Data/CMat.hpp"
#include "Data/CMeasure.h"
#include "Graphics/CGraphicsLayer.h"
#include "Graphics/CGraphicsLayer.h"
#include "Graphics/CGraphicsPoint.h"
#include "Graphics/CGraphicsEllipse.h"
#include "Graphics/CGraphicsPolygon.h"
#include "Graphics/CGraphicsPolyline.h"
#include "Graphics/CGraphicsComplexPolygon.h"
#include "Graphics/CGraphicsRectangle.h"
#include "Graphics/CGraphicsText.h"

namespace Ikomia
{
    namespace Utils
    {
        inline std::string to_string(const cv::KeyPoint& kpt)
        {
            return "angle:" + std::to_string(kpt.angle) +
                    "; class:" + std::to_string(kpt.class_id) +
                    "; x:" + std::to_string(kpt.pt.x) +
                    "; y:" + std::to_string(kpt.pt.y) +
                    "; size:" + std::to_string(kpt.size);
        }

        template<typename T>
        inline std::string to_string(const cv::Point_<T>& pt)
        {
            return "x:" + std::to_string(pt.x) +
                    "; y:" + std::to_string(pt.y);
        }

        namespace Data
        {
            inline std::string          getDimensionName(DataDimension dim)
            {
                switch(dim)
                {
                    case DataDimension::IMAGE:
                        return "Image";
                    case DataDimension::MODALITY:
                        return QObject::tr("Modality").toStdString();
                    case DataDimension::POSITION:
                        return QObject::tr("Position").toStdString();
                    case DataDimension::TIME:
                        return QObject::tr("Time").toStdString();
                    case DataDimension::VOLUME:
                        return QObject::tr("Volume").toStdString();
                    case DataDimension::STUDY:
                        return QObject::tr("Study").toStdString();
                    case DataDimension::SERIE:
                        return QObject::tr("Serie").toStdString();
                    default:
                        return QObject::tr("Dimension").toStdString();;
                }
            }

            inline size_t               getDimensionSize(Dimensions dims, DataDimension dim)
            {
                for(size_t i=0; i<dims.size(); ++i)
                {
                    if(dims[i].first == dim)
                        return dims[i].second;
                }
                return 0;
            }

            inline Bounds               getDimensionBounds(const SubsetBounds& bounds, DataDimension dim)
            {
                for(size_t i=0; i<bounds.size(); ++i)
                {
                    if(bounds[i].first == dim)
                        return bounds[i].second;
                }
                return Bounds(-1, -1);
            }

            inline size_t               getBoundsSize(const Bounds& bounds)
            {
                return bounds.second - bounds.first + 1;
            }

            inline size_t               getSubsetBoundsSize(const SubsetBounds& bounds)
            {
                if(bounds.size() == 0)
                    return 0;

                size_t nb = 1;
                for(size_t i=0; i<bounds.size(); ++i)
                    nb *= getBoundsSize(bounds[i].second);

                return nb;
            }

            inline DimensionIndices     subsetBoundsToIndices(const SubsetBounds& bounds)
            {
                DimensionIndices indices;
                for(size_t i=0; i<bounds.size(); ++i)
                    indices.push_back(std::make_pair(bounds[i].first, bounds[i].second.first));
                return indices;
            }

            inline DimensionIndices     allocateDimensionIndices(Dimensions dims)
            {
                DimensionIndices indices;
                for(size_t i=0; i<dims.size(); ++i)
                    indices.push_back(std::make_pair(dims[i].first, 0));
                return indices;
            }

            inline void                 setDimensionIndex(DimensionIndices& indices, size_t dimIndex, size_t index)
            {
                if(dimIndex < indices.size())
                    indices[dimIndex].second = index;
            }

            inline void                 setSubsetBounds(SubsetBounds& bounds, DataDimension dim, size_t dimIndexFirst, size_t dimIndexLast)
            {
                for(size_t i=0; i<bounds.size(); ++i)
                {
                    if(bounds[i].first == dim)
                    {
                        bounds[i].second = {dimIndexFirst, dimIndexLast};
                    }
                }
            }

            inline std::string          getFileFormatExtension(DataFileFormat format)
            {
                switch(format)
                {
                    case DataFileFormat::NONE: return "";
                    case DataFileFormat::TXT: return ".txt";
                    case DataFileFormat::JSON: return ".json";
                    case DataFileFormat::XML: return ".xml";
                    case DataFileFormat::YAML: return ".yml";
                    case DataFileFormat::CSV: return ".csv";
                    case DataFileFormat::BMP: return ".bmp";
                    case DataFileFormat::JPG: return ".jpg";
                    case DataFileFormat::JP2: return ".jp2";
                    case DataFileFormat::PNG: return ".png";
                    case DataFileFormat::TIF: return ".tif";
                    case DataFileFormat::WEBP: return ".webp";
                    case DataFileFormat::AVI: return ".avi";
                    case DataFileFormat::MPEG: return ".mp4";
                    case DataFileFormat::MKV: return ".mkv";
                    case DataFileFormat::WEBM: return ".webm";
                }
                return "";
            }

            inline DataFileFormat       getFileFormatFromExtension(const std::string& extension)
            {
                if (extension == ".txt")
                    return DataFileFormat::TXT;
                else if (extension == ".json")
                    return DataFileFormat::JSON;
                else if (extension == ".xml")
                    return DataFileFormat::XML;
                else if (extension == ".yml")
                    return DataFileFormat::YAML;
                else if (extension == ".csv")
                    return DataFileFormat::CSV;
                else if (extension == ".bmp")
                    return DataFileFormat::BMP;
                else if (extension == ".jpg")
                    return DataFileFormat::JPG;
                else if (extension == ".jp2")
                    return DataFileFormat::JP2;
                else if (extension == ".png")
                    return DataFileFormat::PNG;
                else if (extension == ".tif")
                    return DataFileFormat::TIF;
                else if (extension == ".webp")
                    return DataFileFormat::WEBP;
                else if (extension == ".avi")
                    return DataFileFormat::AVI;
                else if (extension == ".mp4")
                    return DataFileFormat::MPEG;
                else
                    return DataFileFormat::NONE;
            }
        }

        namespace Database
        {
            inline QString  getMainPath()
            {
                QDir dir;
                QString path = IkomiaApp::getQIkomiaFolder();
                dir.mkpath(path);
                path += "/Ikomia.pcl";
                return path;
            }

            inline QString  getMainConnectionName()
            {
                return "MainConnection";
            }
        }

        namespace Graphics
        {
            inline void             paintSelectedVertex(QPainter *painter, const QPointF &vertex, const qreal lineSize)
            {
                assert(painter);
                const qreal halfSize = 2.0 * lineSize;
                const qreal size = 2.0 * halfSize;

                QPen pen(QColor(0, 0, 0, 255));
                pen.setCosmetic(true);
                painter->setPen(pen);
                painter->setBrush(QColor(255, 255, 255, 255));
                painter->drawRect(QRectF(vertex.x() - halfSize, vertex.y() - halfSize, size, size));
            }
            inline qreal            getSelectionMargin(const qreal lineSize)
            {
                return 2.0 * lineSize;
            }
            inline QColor           toQColor(const CColor& color)
            {
                if(color.size() == 3)
                    return QColor(color[0], color[1], color[2]);
                else if(color.size() == 4)
                    return QColor(color[0], color[1], color[2], color[3]);
                else
                    return QColor();
            }
            inline CColor           toCColor(const QColor& color)
            {
                CColor ccolor = {(uchar)(color.red()), (uchar)(color.green()), (uchar)(color.blue()), (uchar)(color.alpha())};
                return ccolor;
            }
            inline QColor           getRandomQColor()
            {
                return QColor::fromRgb(QRandomGenerator::global()->generate());
            }
            inline CColor           getRandomCColor()
            {
                return toCColor(QColor::fromRgb(QRandomGenerator::global()->generate()));
            }
            inline CGraphicsLayer*  createLayer(const QString& name, const std::vector<ProxyGraphicsItemPtr>& items, const GraphicsContextPtr &globalContext)
            {
                //Ugly switch case but we can't do polymorphism to allow Python bindings
                //We don't want to manage Qt-based pointer into CProxyGraphicsItem based object
                auto pLayer = new CGraphicsLayer(name);
                for(size_t i=0; i<items.size(); ++i)
                {
                    GraphicsItem type = static_cast<GraphicsItem>(items[i]->getType());
                    switch(type)
                    {
                        case GraphicsItem::LAYER: break;
                        case GraphicsItem::RECTANGLE:
                        {
                            auto pItem = new CGraphicsRectangle(globalContext, std::static_pointer_cast<CProxyGraphicsRect>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::ELLIPSE:
                        {
                            auto pItem = new CGraphicsEllipse(globalContext, std::static_pointer_cast<CProxyGraphicsEllipse>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::POINT:
                        {
                            auto pItem = new CGraphicsPoint(globalContext, std::static_pointer_cast<CProxyGraphicsPoint>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::POLYGON:
                        {
                            auto pItem = new CGraphicsPolygon(globalContext, std::static_pointer_cast<CProxyGraphicsPolygon>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::POLYLINE:
                        {
                            auto pItem = new CGraphicsPolyline(globalContext, std::static_pointer_cast<CProxyGraphicsPolyline>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::COMPLEX_POLYGON:
                        {
                            auto pItem = new CGraphicsComplexPolygon(globalContext, std::static_pointer_cast<CProxyGraphicsComplexPoly>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                        case GraphicsItem::TEXT:
                        {
                            auto pItem = new CGraphicsText(globalContext, std::static_pointer_cast<CProxyGraphicsText>(items[i]), pLayer);
                            Q_UNUSED(pItem);
                            break;
                        }
                    }
                }
                return pLayer;
            }
            inline CColor           colorFromJson(const QJsonObject &obj)
            {
                CColor color;
                color.push_back(obj["r"].toInt());
                color.push_back(obj["g"].toInt());
                color.push_back(obj["b"].toInt());
                color.push_back(obj["a"].toInt());
                return color;
            }
        }

        namespace Font
        {
            inline double   getQtToOcvFontScaleFactor(int size)
            {
                //OpenCV scale factor of 1.0 corresponds approximatively to Qt font size of 30
                return (double)size / 30.0;
            }
        }

        namespace Workflow
        {
            // TODO: not elegant and could lead to maintenance issue...
            inline QString  getIODataName(IODataType dataType)
            {
                switch(dataType)
                {
                    case IODataType::NONE:
                        return QObject::tr("None");
                    case IODataType::IMAGE:
                        return QObject::tr("Image");
                    case IODataType::IMAGE_BINARY:
                        return QObject::tr("Binary image");
                    case IODataType::IMAGE_LABEL:
                        return QObject::tr("Label image");
                    case IODataType::VOLUME:
                        return QObject::tr("Volume");
                    case IODataType::VOLUME_BINARY:
                        return QObject::tr("Binary volume");
                    case IODataType::VOLUME_LABEL:
                        return QObject::tr("Label volume");
                    case IODataType::POSITION:
                        return QObject::tr("Position image sequence");
                    case IODataType::INPUT_GRAPHICS:
                    case IODataType::OUTPUT_GRAPHICS:
                        return QObject::tr("Graphics");
                    case IODataType::BLOB_VALUES:
                        return QObject::tr("Blob values");
                    case IODataType::NUMERIC_VALUES:
                        return QObject::tr("Numeric values");
                    case IODataType::VIDEO:
                        return QObject::tr("Video");
                    case IODataType::VIDEO_BINARY:
                        return QObject::tr("Binary video");
                    case IODataType::VIDEO_LABEL:
                        return QObject::tr("Label Video");
                    case IODataType::LIVE_STREAM:
                        return QObject::tr("Live stream");
                    case IODataType::LIVE_STREAM_BINARY:
                        return QObject::tr("Binary live stream");
                    case IODataType::LIVE_STREAM_LABEL:
                        return QObject::tr("Label live stream");
                    case IODataType::WIDGET:
                        return QObject::tr("Custom result view");
                    case IODataType::DESCRIPTORS:
                        return QObject::tr("Descriptors");
                    case IODataType::PROJECT_FOLDER:
                        return QObject::tr("Folder");
                    case IODataType::FOLDER_PATH:
                        return QObject::tr("Folder path");
                    case IODataType::FILE_PATH:
                        return QObject::tr("File path");
                    case IODataType::DNN_DATASET:
                        return QObject::tr("Deep learning dataset");
                    case IODataType::ARRAY:
                        return QObject::tr("Multi-dimensional array");
                    case IODataType::DATA_DICT:
                        return QObject::tr("Generic Python dict");
                    case IODataType::OBJECT_DETECTION:
                        return QObject::tr("Object detection");
                    case IODataType::INSTANCE_SEGMENTATION:
                        return QObject::tr("Instance segmentation");
                    case IODataType::SEMANTIC_SEGMENTATION:
                        return QObject::tr("Semantic segmentation");
                    case IODataType::KEYPOINTS:
                        return QObject::tr("Keypoints detection");
                    case IODataType::TEXT:
                        return QObject::tr("Text detection");
                    case IODataType::JSON:
                        return QObject::tr("JSON data");
                    case IODataType::SCENE_3D:
                        return QObject::tr("3D scene representation");
                }
                return QString();
            }

            // TODO: not elegant and could lead to maintenance issue...
            inline std::string  getIODataEnumName(IODataType dataType)
            {
                switch(dataType)
                {
                    case IODataType::NONE:
                        return "IODataType.NONE";
                    case IODataType::IMAGE:
                        return "IODataType.IMAGE";
                    case IODataType::IMAGE_BINARY:
                        return "IODataType.IMAGE_BINARY";
                    case IODataType::IMAGE_LABEL:
                        return "IODataType.IMAGE_LABEL";
                    case IODataType::VOLUME:
                        return "IODataType.VOLUME";
                    case IODataType::VOLUME_BINARY:
                        return "IODataType.VOLUME_BINARY";
                    case IODataType::VOLUME_LABEL:
                        return "IODataType.VOLUME_LABEL";
                    case IODataType::POSITION:
                        return "IODataType.POSITION";
                    case IODataType::INPUT_GRAPHICS:
                        return "IODataType.INPUT_GRAPHICS";
                    case IODataType::OUTPUT_GRAPHICS:
                        return "IODataType.OUTPUT_GRAPHICS";
                    case IODataType::BLOB_VALUES:
                        return "IODataType.BLOB_VALUES";
                    case IODataType::NUMERIC_VALUES:
                        return "IODataType.NUMERIC_VALUES";
                    case IODataType::VIDEO:
                        return "IODataType.VIDEO";
                    case IODataType::VIDEO_BINARY:
                        return "IODataType.VIDEO_BINARY";
                    case IODataType::VIDEO_LABEL:
                        return "IODataType.VIDEO_LABEL";
                    case IODataType::LIVE_STREAM:
                        return "IODataType.LIVE_STREAM";
                    case IODataType::LIVE_STREAM_BINARY:
                        return "IODataType.LIVE_STREAM_BINARY";
                    case IODataType::LIVE_STREAM_LABEL:
                        return "IODataType.LIVE_STREAM_LABEL";
                    case IODataType::WIDGET:
                        return "IODataType.WIDGET";
                    case IODataType::DESCRIPTORS:
                        return "IODataType.DESCRIPTORS";
                    case IODataType::PROJECT_FOLDER:
                        return "IODataType.PROJECT_FOLDER";
                    case IODataType::FOLDER_PATH:
                        return "IODataType.FOLDER_PATH";
                    case IODataType::FILE_PATH:
                        return "IODataType.FILE_PATH";
                    case IODataType::DNN_DATASET:
                        return "IODataType.DNN_DATASET";
                    case IODataType::ARRAY:
                        return "IODataType.ARRAY";
                    case IODataType::DATA_DICT:
                        return "IODataType.DATA_DICT";
                    case IODataType::OBJECT_DETECTION:
                        return "IODataType.OBJECT_DETECTION";
                    case IODataType::INSTANCE_SEGMENTATION:
                        return "IODataType.INSTANCE_SEGMENTATION";
                    case IODataType::SEMANTIC_SEGMENTATION:
                        return "IODataType.SEMANTIC_SEGMENTATION";
                    case IODataType::KEYPOINTS:
                        return "IODataType.KEYPOINTS";
                    case IODataType::TEXT:
                        return "IODataType.TEXT";
                    case IODataType::JSON:
                        return "IODataType.JSON";
                    case IODataType::SCENE_3D:
                        return "IODataType.SCENE_3D";
                }
                return "";
            }
        }

        namespace Image
        {
            /**
            * @brief Gets the maximum theoretical pixel value deduced from bits per pixel.
            * @param img: image.
            * @return maximum pixel value.
            */
            inline double   getMaxValue(const CMat& img)
            {
                double value = 0;
                int depth = img.depth();

                switch(depth)
                {
                    case CV_8S: value = 127; break;
                    case CV_8U: value = 255; break;
                    case CV_16S: value = 32767; break;
                    case CV_16U: value = 65535; break;
                    case CV_32S: value = 2147483647; break;
                    case CV_32F: value = 1.0; break;
                    case CV_64F: value = 1.0; break;
                }
                return value;
            }
        }

        namespace Plugin
        {
            inline std::string  getAlgoTypeString(const AlgoType& type)
            {
                switch(type)
                {
                    case AlgoType::INFER:
                        return "INFER";
                    case AlgoType::TRAIN:
                        return "TRAIN";
                    case AlgoType::DATASET:
                        return "DATASET";
                    case AlgoType::OTHER:
                    default:
                        return "OTHER";
                }
            }
            inline AlgoType     getAlgoTypeFromString(const std::string& name)
            {
                if (name == "INFER" || name == "INFERENCE")
                    return AlgoType::INFER;
                else if (name == "TRAIN")
                    return AlgoType::TRAIN;
                else if (name == "DATASET")
                    return AlgoType::DATASET;
                else
                    return AlgoType::OTHER;
            }
        }
    }
}

using namespace Ikomia;

#endif // DATAUTILS_HPP
