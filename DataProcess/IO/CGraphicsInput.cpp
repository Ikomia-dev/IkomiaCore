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

#include "CGraphicsInput.h"
#include "Graphics/CGraphicsLayer.h"
#include "Graphics/CGraphicsRegistration.h"
#include "CGraphicsOutput.h"
#include "CObjectDetectionIO.h"
#include "CInstanceSegIO.h"
#include "CKeypointsIO.h"
#include "CTextIO.h"
#include "UtilsTools.hpp"
#include <QJsonDocument>
#include <QJsonArray>

CGraphicsInput::CGraphicsInput() : CWorkflowTaskIO(IODataType::INPUT_GRAPHICS, "CGraphicsInput")
{
    m_description = QObject::tr("Graphics items organized in layer.\n"
                                "Represent shapes and types of objects in image.\n"
                                "Graphics can be created interactively by user.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

CGraphicsInput::CGraphicsInput(const std::string &name) : CWorkflowTaskIO(IODataType::INPUT_GRAPHICS, name)
{
    m_description = QObject::tr("Graphics items organized in layer.\n"
                                "Represent shapes and types of objects in image.\n"
                                "Graphics can be created interactively by user.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

CGraphicsInput::CGraphicsInput(CGraphicsLayer *pLayer, const std::string &name) : CWorkflowTaskIO(IODataType::INPUT_GRAPHICS, name)
{
    m_description = QObject::tr("Graphics items organized in layer.\n"
                                "Represent shapes and types of objects in image.\n"
                                "Graphics can be created interactively by user.").toStdString();
    m_saveFormat = DataFileFormat::JSON;
    m_pLayer = pLayer;
    m_source = GraphicsSource::GRAPHICS_LAYER;
}

CGraphicsInput::CGraphicsInput(const CGraphicsInput &in) : CWorkflowTaskIO(in)
{
    m_source = in.m_source;
    m_pLayer = in.m_pLayer;
    m_items = in.m_items;
}

CGraphicsInput::CGraphicsInput(CGraphicsInput &&in) : CWorkflowTaskIO(in)
{
    m_source = std::move(in.m_source);
    m_pLayer = std::move(in.m_pLayer);
    m_items = std::move(in.m_items);
}

CGraphicsInput::CGraphicsInput(const CGraphicsOutput &out) : CWorkflowTaskIO(out)
{
    m_dataType = IODataType::INPUT_GRAPHICS;
    m_pLayer = nullptr;
    m_items = out.getItems();
    m_source = GraphicsSource::EXTERNAL_DATA;
}

CGraphicsInput &CGraphicsInput::operator=(const CGraphicsInput &in)
{
    CWorkflowTaskIO::operator=(in);
    m_source = in.m_source;
    m_pLayer = in.m_pLayer;
    m_items = in.m_items;
    return *this;
}

CGraphicsInput &CGraphicsInput::operator=(CGraphicsInput &&in)
{
    CWorkflowTaskIO::operator=(in);
    m_source = std::move(in.m_source);
    m_pLayer = std::move(in.m_pLayer);
    m_items = std::move(in.m_items);
    return *this;
}

CGraphicsInput &CGraphicsInput::operator=(const CGraphicsOutput &out)
{
    CWorkflowTaskIO::operator=(out);
    m_dataType = IODataType::INPUT_GRAPHICS;
    m_pLayer = nullptr;
    m_items = out.getItems();
    m_source = GraphicsSource::EXTERNAL_DATA;
    return *this;
}

void CGraphicsInput::setLayer(CGraphicsLayer *pLayer)
{
    m_pLayer = pLayer;
    m_items.clear();
    m_source = GraphicsSource::GRAPHICS_LAYER;
}

void CGraphicsInput::setItems(const std::vector<ProxyGraphicsItemPtr> &items)
{
    m_pLayer = nullptr;
    m_items = items;
    m_source = GraphicsSource::EXTERNAL_DATA;
}

const CGraphicsLayer *CGraphicsInput::getLayer() const
{
    return m_pLayer;
}

std::vector<ProxyGraphicsItemPtr> CGraphicsInput::getItems() const
{
    if(m_source == GraphicsSource::GRAPHICS_LAYER && m_pLayer)
    {
        //Graphics from layer can be modified by user
        //so we retrieve a new list for each call
        std::vector<ProxyGraphicsItemPtr> items;
        auto graphicsItems = m_pLayer->getChildItems();

        for(int i=0; i<graphicsItems.size(); ++i)
        {
            auto pItem = dynamic_cast<CGraphicsItem*>(graphicsItems[i]);
            items.push_back(pItem->createProxyGraphicsItem());
        }
        return items;
    }
    else
        return m_items;
}

QRectF CGraphicsInput::getBoundingRect() const
{
    QRectF rect;

    for(size_t i=0; i<m_items.size(); ++i)
        rect = rect.united(m_items[i]->getBoundingQRect());

    return rect;
}

bool CGraphicsInput::isDataAvailable() const
{
    if(m_source == GraphicsSource::GRAPHICS_LAYER)
        return m_pLayer != nullptr;
    else
        return m_items.size() > 0;
}

void CGraphicsInput::clearData()
{
    m_pLayer = nullptr;
    m_items.clear();
}

void CGraphicsInput::copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr)
{
    auto type = ioPtr->getDataType();
    if (type == IODataType::INPUT_GRAPHICS)
    {
        auto pGraphicsInput = dynamic_cast<const CGraphicsInput*>(ioPtr.get());
        if(pGraphicsInput)
            *this = *pGraphicsInput;
    }
    else if (type == IODataType::OUTPUT_GRAPHICS)
    {
        auto pGraphicsOutput = dynamic_cast<const CGraphicsOutput*>(ioPtr.get());
        if(pGraphicsOutput)
            *this = *pGraphicsOutput;
    }
    else if (type == IODataType::OBJECT_DETECTION)
    {
        auto pObjectDetectionIO = std::dynamic_pointer_cast<CObjectDetectionIO>(ioPtr);
        if (pObjectDetectionIO)
        {
            auto graphicsOutPtr = pObjectDetectionIO->getGraphicsIO();
            if (graphicsOutPtr)
            {
                auto pGraphicsOut = dynamic_cast<const CGraphicsOutput*>(graphicsOutPtr.get());
                if (pGraphicsOut)
                    *this = *pGraphicsOut;
            }
        }
    }
    else if (type == IODataType::INSTANCE_SEGMENTATION)
    {
        auto pInstanceSegIO = std::dynamic_pointer_cast<CInstanceSegIO>(ioPtr);
        if (pInstanceSegIO)
        {
            auto graphicsOutPtr = pInstanceSegIO->getGraphicsIO();
            if (graphicsOutPtr)
            {
                auto pGraphicsOut = dynamic_cast<const CGraphicsOutput*>(graphicsOutPtr.get());
                if (pGraphicsOut)
                    *this = *pGraphicsOut;
            }
        }
    }
    else if (type == IODataType::KEYPOINTS)
    {
        auto keyptsIOPtr = std::dynamic_pointer_cast<CKeypointsIO>(ioPtr);
        if (keyptsIOPtr)
        {
            auto graphicsOutPtr = keyptsIOPtr->getGraphicsIO();
            if (graphicsOutPtr)
            {
                auto pGraphicsOut = dynamic_cast<const CGraphicsOutput*>(graphicsOutPtr.get());
                if (pGraphicsOut)
                    *this = *pGraphicsOut;
            }
        }
    }
    else if (type == IODataType::TEXT)
    {
        auto textIOPtr = std::dynamic_pointer_cast<CTextIO>(ioPtr);
        if (textIOPtr)
        {
            auto graphicsOutPtr = textIOPtr->getGraphicsIO();
            if (graphicsOutPtr)
            {
                auto pGraphicsOut = dynamic_cast<const CGraphicsOutput*>(graphicsOutPtr.get());
                if (pGraphicsOut)
                    *this = *pGraphicsOut;
            }
        }
    }
}

CGraphicsInput::GraphicsInputPtr CGraphicsInput::clone() const
{
    return std::static_pointer_cast<CGraphicsInput>(cloneImp());
}

std::shared_ptr<CWorkflowTaskIO> CGraphicsInput::cloneImp() const
{
    return std::shared_ptr<CGraphicsInput>(new CGraphicsInput(*this));
}

QJsonObject CGraphicsInput::toJsonInternal() const
{
    QJsonObject root;
    if (m_pLayer)
        root["layer"] = m_pLayer->getName();
    else
        root["layer"] = "InputLayer";

    QJsonArray itemArray;
    for(size_t i=0; i<m_items.size(); ++i)
    {
        QJsonObject itemObj;
        m_items[i]->toJson(itemObj);
        itemArray.append(itemObj);
    }
    root["items"] = itemArray;
    return root;
}

void CGraphicsInput::fromJsonInternal(const QJsonDocument &jsonDoc)
{
    QJsonObject root = jsonDoc.object();
    if(root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading graphics: empty JSON structure", __func__, __FILE__, __LINE__);

    // Load items
    m_items.clear();
    m_source = GraphicsSource::EXTERNAL_DATA;
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

void CGraphicsInput::load(const std::string &path)
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

void CGraphicsInput::save(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson());
}

std::string CGraphicsInput::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CGraphicsInput::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

void CGraphicsInput::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading graphics input: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}
