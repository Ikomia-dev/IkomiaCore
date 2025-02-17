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

#include "CGraphicsOutput.h"
#include <QJsonDocument>
#include <QJsonArray>
#include "DataProcessTools.hpp"
#include "Main/CoreTools.hpp"
#include "Graphics/CGraphicsPoint.h"
#include "Graphics/CGraphicsEllipse.h"
#include "Graphics/CGraphicsPolygon.h"
#include "Graphics/CGraphicsPolyline.h"
#include "Graphics/CGraphicsComplexPolygon.h"
#include "Graphics/CGraphicsRectangle.h"
#include "Graphics/CGraphicsText.h"
#include "Graphics/CGraphicsRegistration.h"
#include "CGraphicsInput.h"

CGraphicsOutput::CGraphicsOutput() : CWorkflowTaskIO(IODataType::OUTPUT_GRAPHICS, "GraphicsOutput")
{
    m_description = QObject::tr("Graphics items organized in layer.\n"
                                "Represent shapes and types of objects extracted from image.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

CGraphicsOutput::CGraphicsOutput(const std::string &name) : CWorkflowTaskIO(IODataType::OUTPUT_GRAPHICS, name)
{
    m_description = QObject::tr("Graphics items organized in layer.\n"
                                "Represent shapes and types of objects extracted from image.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

CGraphicsOutput::CGraphicsOutput(const CGraphicsOutput& out) : CWorkflowTaskIO(out)
{
    m_items = out.m_items;
    m_imageIndex = out.m_imageIndex;
}

CGraphicsOutput::CGraphicsOutput(CGraphicsOutput&& out) : CWorkflowTaskIO(out)
{
    m_items = std::move(out.m_items);
    m_imageIndex = std::move(out.m_imageIndex);
}

CGraphicsOutput::CGraphicsOutput(const CGraphicsInput &in) : CWorkflowTaskIO(in)
{
    m_dataType = IODataType::OUTPUT_GRAPHICS;
    m_items = in.getItems();
}

CGraphicsOutput::~CGraphicsOutput()
{
}

CGraphicsOutput &CGraphicsOutput::operator=(const CGraphicsOutput &out)
{
    CWorkflowTaskIO::operator=(out);
    m_items = out.m_items;
    m_imageIndex = out.m_imageIndex;
    return *this;
}

std::string CGraphicsOutput::repr() const
{
    std::stringstream s;
    s << "CGraphicsOutput(" << m_name << ")";
    return s.str();
}

CGraphicsOutput &CGraphicsOutput::operator=(CGraphicsOutput&& out)
{
    CWorkflowTaskIO::operator=(out);
    m_items = std::move(out.m_items);
    m_imageIndex = std::move(out.m_imageIndex);
    return *this;
}

CGraphicsOutput &CGraphicsOutput::operator=(const CGraphicsInput &in)
{
    CWorkflowTaskIO::operator=(in);
    m_dataType = IODataType::OUTPUT_GRAPHICS;
    m_items = in.getItems();
    return *this;
}

void CGraphicsOutput::setNewLayer(const std::string name)
{
    m_layerName = name;
    clearData();
}

void CGraphicsOutput::setImageIndex(int index)
{
    m_imageIndex = index;
}

void CGraphicsOutput::setItems(const std::vector<ProxyGraphicsItemPtr> &items)
{
    m_items = items;
}

int CGraphicsOutput::getImageIndex() const
{
    return m_imageIndex;
}

std::vector<ProxyGraphicsItemPtr> CGraphicsOutput::getItems() const
{
    return m_items;
}

CMat CGraphicsOutput::getImageWithGraphics(const CMat &image) const
{
    if(image.empty())
        return CMat();

    CMat img =image.clone();
    Utils::Image::burnGraphics(img, getItems());
    return img;
}

CMat CGraphicsOutput::getImageWithMaskAndGraphics(const CMat &image) const
{
    return getImageWithGraphics(image);
}

bool CGraphicsOutput::isDataAvailable() const
{
    return m_items.size() > 0;
}

CGraphicsLayer *CGraphicsOutput::createLayer(const GraphicsContextPtr &globalContext)
{
    return Utils::Graphics::createLayer(QString::fromStdString(m_name), m_items, globalContext);
}

void CGraphicsOutput::addItem(const ProxyGraphicsItemPtr &itemPtr)
{
    m_items.push_back(itemPtr);
}

ProxyGraphicsItemPtr CGraphicsOutput::addPoint(const CPointF &pt)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPoint>(pt));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addPoint(const CPointF &pt, const CGraphicsPointProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPoint>(pt, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addRectangle(float x, float y, float width, float height)
{
    m_items.push_back(std::make_shared<CProxyGraphicsRect>(x, y, width, height));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addRectangle(float x, float y, float width, float height, const CGraphicsRectProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsRect>(x, y, width, height, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addEllipse(float x, float y, float width, float height)
{
    m_items.push_back(std::make_shared<CProxyGraphicsEllipse>(x, y, width, height));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addEllipse(float x, float y, float width, float height, const CGraphicsEllipseProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsEllipse>(x, y, width, height, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addPolygon(const std::vector<CPointF> &points)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPolygon>(points));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addPolygon(const std::vector<CPointF> &points, const CGraphicsPolygonProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPolygon>(points, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addPolyline(const std::vector<CPointF> &points)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPolyline>(points));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addPolyline(const std::vector<CPointF> &points, const CGraphicsPolylineProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsPolyline>(points, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addComplexPolygon(const PolygonF &outer, const std::vector<PolygonF> &inners)
{
    m_items.push_back(std::make_shared<CProxyGraphicsComplexPoly>(outer, inners));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addComplexPolygon(const PolygonF &outer, const std::vector<PolygonF> &inners, const CGraphicsPolygonProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsComplexPoly>(outer, inners, prop));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addText(const std::string &text, float x, float y)
{
    m_items.push_back(std::make_shared<CProxyGraphicsText>(text, x, y));
    return m_items.back();
}

ProxyGraphicsItemPtr CGraphicsOutput::addText(const std::string &text, float x, float y, const CGraphicsTextProperty &prop)
{
    m_items.push_back(std::make_shared<CProxyGraphicsText>(text, x, y, prop));
    return m_items.back();
}

void CGraphicsOutput::clearData()
{
    m_items.clear();
}

void CGraphicsOutput::copy(const WorkflowTaskIOPtr &ioPtr)
{
    auto type = ioPtr->getDataType();
    if (type == IODataType::OUTPUT_GRAPHICS)
    {
        auto pGraphicsOutput = dynamic_cast<const CGraphicsOutput*>(ioPtr.get());
        if(pGraphicsOutput)
            *this = *pGraphicsOutput;
    }
    else if (type == IODataType::INPUT_GRAPHICS)
    {
        auto pGraphicsInput = dynamic_cast<const CGraphicsInput*>(ioPtr.get());
        if(pGraphicsInput)
            *this = *pGraphicsInput;
    }
}

CGraphicsOutput::GraphicsOutputPtr CGraphicsOutput::clone() const
{
    return std::static_pointer_cast<CGraphicsOutput>(cloneImp());
}

WorkflowTaskIOPtr CGraphicsOutput::cloneImp() const
{
    return std::shared_ptr<CGraphicsOutput>(new CGraphicsOutput(*this));
}

QJsonObject CGraphicsOutput::toJsonInternal() const
{
    QJsonObject root;
    root["layer"] = QString::fromStdString(m_layerName);

    QJsonArray itemArray;
    for(size_t i=0; i<m_items.size(); ++i)
    {
        QJsonObject itemObj;
        m_items[i]->toJson(itemObj);
        itemArray.append(itemObj);
    }
    root["items"] = itemArray;
    root["imageIndex"] = m_imageIndex;
    return root;
}

void CGraphicsOutput::fromJsonInternal(const QJsonDocument &jsonDoc)
{
    QJsonObject root = jsonDoc.object();
    if(root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading graphics: empty JSON structure", __func__, __FILE__, __LINE__);

    m_layerName = root["layer"].toString().toStdString();
    m_imageIndex = root["imageIndex"].toInt();

    // Load items
    m_items.clear();
    CGraphicsRegistration registry;
    auto factory = registry.getProxyFactory();
    QJsonArray items = root["items"].toArray();

    for(int i=0; i<items.size(); ++i)
    {
        QJsonObject item = items[i].toObject();
        auto itemPtr = factory.createObject(static_cast<size_t>(item["type"].toInt()));
        itemPtr->fromJson(item);
        m_items.push_back(itemPtr);
    }
}

void CGraphicsOutput::save(const std::string &path)
{
    CWorkflowTaskIO::save(path);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson(QJsonDocument::Compact));
}

void CGraphicsOutput::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading graphics: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

std::string CGraphicsOutput::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CGraphicsOutput::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

void CGraphicsOutput::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading graphics output: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}


