/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the Ikomia API libraries.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef CGRAPHICSINPUT_H
#define CGRAPHICSINPUT_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "Graphics/CGraphicsItem.hpp"

/** @cond INTERNAL */

class CGraphicsLayer;
class QGraphicsItem;
class CGraphicsOutput;
class CObjectDetectionIO;

class DATAPROCESSSHARED_EXPORT CGraphicsInput : public CWorkflowTaskIO
{
    public:

        enum GraphicsSource { GRAPHICS_LAYER, EXTERNAL_DATA };

        using GraphicsInputPtr = std::shared_ptr<CGraphicsInput>;

        CGraphicsInput();
        CGraphicsInput(const std::string& name);
        CGraphicsInput(CGraphicsLayer* pLayer, const std::string& name="CGraphicsInput");
        CGraphicsInput(const CGraphicsInput& in);
        CGraphicsInput(CGraphicsInput&& in);
        CGraphicsInput(const CGraphicsOutput &out);

        virtual ~CGraphicsInput() = default;

        CGraphicsInput&     operator=(const CGraphicsInput& in);
        CGraphicsInput&     operator=(CGraphicsInput&& in);
        CGraphicsInput&     operator=(const CGraphicsOutput& out);

        std::string         repr() const override;

        void                setLayer(CGraphicsLayer* pLayer);
        void                setItems(const std::vector<ProxyGraphicsItemPtr>& items);

        const CGraphicsLayer*               getLayer() const;
        std::vector<ProxyGraphicsItemPtr>   getItems() const;
        QRectF                              getBoundingRect() const;
        CMat                                getImageWithGraphics(const CMat &image) const override;
        CMat                                getImageWithMaskAndGraphics(const CMat &image) const override;

        bool                isDataAvailable() const override;

        void                clearData() override;

        void                copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr) override;

        GraphicsInputPtr    clone() const;

        void                load(const std::string &path) override;

        void                save(const std::string &path) override;

        std::string         toJson() const override;
        std::string         toJson(const std::vector<std::string>& options) const override;
        void                fromJson(const std::string &jsonStr) override;

    private:

        virtual WorkflowTaskIOPtr cloneImp() const override;

        QJsonObject         toJsonInternal() const;
        void                fromJsonInternal(const QJsonDocument& jsonDoc);

    private:

        //Must be seen as weak pointer
        const CGraphicsLayer*               m_pLayer = nullptr;
        std::vector<ProxyGraphicsItemPtr>   m_items;
        GraphicsSource                      m_source = GraphicsSource::GRAPHICS_LAYER;
};

using GraphicsInputPtr = std::shared_ptr<CGraphicsInput>;

class DATAPROCESSSHARED_EXPORT CGraphicsInputFactory: public CWorkflowTaskIOFactory
{
    public:

        CGraphicsInputFactory()
        {
            m_name = "CGraphicsInput";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CGraphicsInput>();
        }
};

/** @endcond */

#endif // CGRAPHICSINPUT_H
