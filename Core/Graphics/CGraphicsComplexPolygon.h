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

#ifndef CGRAPHICSCOMPLEXPOLYGON_H
#define CGRAPHICSCOMPLEXPOLYGON_H

#include <QGraphicsPathItem>
#include "CGraphicsPolygon.h"

/** @cond INTERNAL */

//-----------------------------------
//- Class CProxyGraphicsComplexPoly
//- Proxy class without Qt dependency
//-----------------------------------
class CORESHARED_EXPORT CProxyGraphicsComplexPoly: public CProxyGraphicsItem
{
    public:

        CProxyGraphicsComplexPoly();
        CProxyGraphicsComplexPoly(const PolygonF& outer, const std::vector<PolygonF>& inners);
        CProxyGraphicsComplexPoly(const PolygonF& outer, const std::vector<PolygonF>& inners, const CGraphicsPolygonProperty& property);

        void                        setOuter(const PolygonF& poly);
        void                        setInners(const std::vector<PolygonF>& polygons);
        void                        setProperty(const CGraphicsPolygonProperty& prop);
        void                        setCategory(const std::string& categ) override;

        PolygonF                    getOuter() const;
        std::vector<PolygonF>       getInners() const;
        QRectF                      getBoundingQRect() const override;
        std::vector<float>          getBoundingRect() const override;
        CGraphicsPolygonProperty    getProperty() const;
        std::string                 getCategory() const override;

        void                        translate(float dx, float dy) override;

        void                        insertToImage(CMat& image, CGraphicsConversion& filler, bool bForceFill, bool bBinary, bool bgr=false) const override;

        std::shared_ptr<CProxyGraphicsItem> clone() const override;

        void                        toJson(QJsonObject &obj) const override;

        void                        fromJson(const QJsonObject &obj) override;

    public:

        PolygonF                    m_outer;
        std::vector<PolygonF>       m_inners;
        CGraphicsPolygonProperty    m_property;
};

//-----------------------------------------------------------------------
//- Class CProxyGraphicsComplexPolyFactory
//- Factory for dynamic CProxyGraphicsComplexPoly instanciation (from type)
//-----------------------------------------------------------------------
class CProxyGraphicsComplexPolyFactory: public CProxyGraphicsFactory
{
    public:

        CProxyGraphicsComplexPolyFactory()
        {
            m_type = static_cast<size_t>(GraphicsItem::COMPLEX_POLYGON);
        }

        ~CProxyGraphicsComplexPolyFactory(){}

        ProxyGraphicsItemPtr  create() override
        {
            return std::make_shared<CProxyGraphicsComplexPoly>();
        }
};

//-------------------------------
//- Class CGraphicsComplexPolygon
//-------------------------------
class CORESHARED_EXPORT CGraphicsComplexPolygon : public QGraphicsPathItem, public CGraphicsItem
{
    public:

        enum { Type = UserType + static_cast<size_t>(GraphicsItem::COMPLEX_POLYGON) };

        CGraphicsComplexPolygon(QGraphicsItem* pParent=nullptr);
        CGraphicsComplexPolygon(const CGraphicsPolygonProperty& property, QGraphicsItem* pParent=nullptr);
        CGraphicsComplexPolygon(size_t id, const CGraphicsPolygonProperty& property, QGraphicsItem* pParent=nullptr);
        CGraphicsComplexPolygon(const GraphicsContextPtr &globalContext, const std::shared_ptr<CProxyGraphicsComplexPoly> &proxyItem, QGraphicsItem *pParent=nullptr);
        CGraphicsComplexPolygon(const CGraphicsComplexPolygon& polygon);

        CGraphicsComplexPolygon&    operator=(const CGraphicsComplexPolygon& polygon);

        QGraphicsItem*              clone() const override;

        void                        setPenColor(QColor color) override;
        void                        setBrushColor(QColor color) override;

        int                         type() const override;

        QRectF                      boundingRect() const override;

        QPolygonF                   getOuterPolygon() const;
        QList<QPolygonF>            getInnerPolygons() const;
        QByteArray                  getJsonData(CGraphicsJSON &jsonMgr) const override;

        void                        setOuterPolygon(QPolygonF polygon, bool bRedraw);
        void                        setInnerPolygons(const QList<QPolygonF>& innerList, bool bRedraw);

        void                        addInnerPolygon(QPolygonF polygon, bool bRedraw);

        void                        redraw();

        ProxyGraphicsItemPtr        createProxyGraphicsItem() const override;

        void                        buildPainterPath();
        void                        buildFromJsonData(CGraphicsJSON& jsonMgr, QByteArray data) override;

        void                        insertToImage(CMat& image, CGraphicsConversion& filler, bool bForceFill, bool bBinary) const override;

        void                        clear();

    private:

        void                        copy(const CGraphicsComplexPolygon& polygon);

    private:

        QPolygonF           m_outer;
        QList<QPolygonF>    m_innerList;
};

//-----------------------------------------------------------------------
//- Class CGraphicsComplexPolygonFactory
//- Factory for dynamic CGraphicsComplexPolygon instanciation (from type)
//-----------------------------------------------------------------------
class CGraphicsComplexPolygonFactory: public CGraphicsFactory
{
    public:

        CGraphicsComplexPolygonFactory()
        {
            m_type = CGraphicsComplexPolygon::UserType + static_cast<size_t>(GraphicsItem::COMPLEX_POLYGON);
        }

        ~CGraphicsComplexPolygonFactory(){}

        QGraphicsItem*  create(QGraphicsItem* pParent) override
        {
            return new CGraphicsComplexPolygon(pParent);
        }
};

/** @endcond */

#endif // CGRAPHICSCOMPLEXPOLYGON_H
