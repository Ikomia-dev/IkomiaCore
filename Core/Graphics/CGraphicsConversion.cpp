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

#include <QFont>
#include "UtilsTools.hpp"
#include "Main/CoreTools.hpp"
#include "CGraphicsConversion.h"
#include "Graphics/CGraphicsPoint.h"
#include "Graphics/CGraphicsRectangle.h"
#include "Graphics/CGraphicsEllipse.h"
#include "Graphics/CGraphicsPolygon.h"
#include "Graphics/CGraphicsPolyline.h"
#include "Graphics/CGraphicsComplexPolygon.h"
#include "Graphics/CGraphicsText.h"

CGraphicsConversion::CGraphicsConversion()
{
}

CGraphicsConversion::CGraphicsConversion(int width, int height)
{
    m_width = width;
    m_height = height;
}

QList<QGraphicsItem*> CGraphicsConversion::binaryMaskToGraphics(const CMat &mask, QGraphicsItem *pParent,
                                                                const CColor &penColor, const CColor &brushColor, int lineSize)
{
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> polygonArray;

    if(mask.data == nullptr || mask.type() != CV_8UC1)
        return QList<QGraphicsItem*>();

    cv::findContours(mask, polygonArray, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_L1);
    return blobsToGraphics(polygonArray, hierarchy, pParent, penColor, brushColor, lineSize);
}

std::vector<ProxyGraphicsItemPtr> CGraphicsConversion::binaryMaskToProxyGraphics(const CMat &mask, const CColor &penColor, const CColor &brushColor, int lineSize)
{
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> polygonArray;

    if(mask.data == nullptr || mask.type() != CV_8UC1)
        return std::vector<ProxyGraphicsItemPtr>();

    cv::findContours(mask, polygonArray, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_L1);
    return blobsToProxyGraphics(polygonArray, hierarchy, penColor, brushColor, lineSize);
}

QList<QGraphicsItem*> CGraphicsConversion::blobsToGraphics(const std::vector<std::vector<cv::Point>>& polygonArray,
                                                           const std::vector<cv::Vec4i>& hierarchy, QGraphicsItem *pParent,
                                                           const CColor &penColor, const CColor &brushColor, int lineSize)
{
    int holeId;
    QList<QGraphicsItem*> graphicsList;

    if(polygonArray.size() > 0)
    {
        CGraphicsPolygonProperty property;
        property.m_penColor = penColor;
        property.m_brushColor = brushColor;
        property.m_lineSize = lineSize;

        for(int id=0; id >=0; id=hierarchy[id][0])
        {
            if((holeId = hierarchy[id][2]) == -1)
            {
                auto poly = convertToQPolygon(polygonArray[id]);
                if(poly.size() > 0)
                {
                    CGraphicsPolygon* pGraphPolygon = new CGraphicsPolygon(property, pParent);
                    pGraphPolygon->setPolygon(poly);
                    graphicsList.push_back(pGraphPolygon);
                }
            }
            else
            {
                auto outer = convertToQPolygon(polygonArray[id]);
                if(outer.size() > 0)
                {
                    CGraphicsComplexPolygon* pGraphPolygon = new CGraphicsComplexPolygon(property, pParent);
                    pGraphPolygon->setOuterPolygon(outer, false);

                    for(; holeId >=0; holeId=hierarchy[holeId][0])
                    {
                        auto inner = convertToQPolygon(polygonArray[holeId]);
                        if(inner.size() > 0)
                            pGraphPolygon->addInnerPolygon(inner, false);
                    }
                    pGraphPolygon->redraw();
                    graphicsList.push_back(pGraphPolygon);
                }
            }
        }
    }
    return graphicsList;
}

std::vector<ProxyGraphicsItemPtr> CGraphicsConversion::blobsToProxyGraphics(const std::vector<std::vector<cv::Point> > &polygonArray,
                                                                            const std::vector<cv::Vec4i> &hierarchy,
                                                                            const CColor &penColor, const CColor &brushColor, int lineSize)
{
    int holeId;
    std::vector<ProxyGraphicsItemPtr> graphicsList;

    if(polygonArray.size() > 0)
    {
        CGraphicsPolygonProperty property;
        property.m_penColor = penColor;
        property.m_brushColor = brushColor;
        property.m_lineSize = lineSize;

        for(int id=0; id >=0; id=hierarchy[id][0])
        {
            if((holeId = hierarchy[id][2]) == -1)
            {
                auto poly = convertToPolygon(polygonArray[id]);
                if(poly.size() > 0)
                {
                    auto graphPolygonPtr = std::make_shared<CProxyGraphicsPolygon>(poly, property);
                    graphicsList.push_back(graphPolygonPtr);
                }
            }
            else
            {
                auto outer = convertToPolygon(polygonArray[id]);
                if(outer.size() > 0)
                {
                    std::vector<PolygonF> inners;
                    for(; holeId >=0; holeId=hierarchy[holeId][0])
                    {
                        auto inner = convertToPolygon(polygonArray[holeId]);
                        if(inner.size() > 0)
                            inners.push_back(inner);
                    }
                    auto graphPolygonPtr = std::make_shared<CProxyGraphicsComplexPoly>(outer, inners, property);
                    graphicsList.push_back(graphPolygonPtr);
                }
            }
        }
    }
    return graphicsList;
}

CMat CGraphicsConversion::graphicsToBinaryMask(const QList<QGraphicsItem*>& graphics)
{
    assert(m_width > 0 && m_height > 0);

    CMat mask(m_height, m_width, CV_8UC1, cv::Scalar(0));

    //Double dispatch design pattern
    for(auto it : graphics)
    {
        auto pItem = dynamic_cast<CGraphicsItem*>(it);
        if (pItem && pItem->getItemType() != GraphicsItem::TEXT)
            pItem->insertToImage(mask, *this, true, true);
    }
    return mask;
}

CMat CGraphicsConversion::graphicsToBinaryMask(const std::vector<std::shared_ptr<CProxyGraphicsItem> > &graphics)
{
    assert(m_width > 0 && m_height > 0);

    CMat mask(m_height, m_width, CV_8UC1, cv::Scalar(0));

    //Double dispatch design pattern
    for(auto it : graphics)
    {
        if (!it->isTextItem())
            std::dynamic_pointer_cast<CProxyGraphicsItem>(it)->insertToImage(mask, *this, true, true);
    }
    return mask;
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsPoint *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);
    Q_UNUSED(bForceFill);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Ellipse insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->rect());
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    cv::Scalar color = {255, 255, 255, 255};
    if(!bBinary)
    {
        QColor penColor = pItem->brush().color();
        color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
    }
    cv::RotatedRect rcObj(cv::Point2f(rc.topLeft().x(), rc.topLeft().y()),
                          cv::Point2f(rc.topRight().x(), rc.topRight().y()),
                          cv::Point2f(rc.bottomRight().x(), rc.bottomRight().y()));
    cv::ellipse(image, rcObj, color, cv::FILLED);
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsRectangle *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Rectangle insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->rect());
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    int thickness;
    cv::Scalar color = {255, 255, 255, 255};
    QColor brushColor = pItem->brush().color();

    if(brushColor.alpha() == 0 && bForceFill == false)
    {
        thickness = pItem->pen().width();
        if(!bBinary)
        {
            QColor penColor = pItem->pen().color();
            color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
        }
    }
    else
    {
        thickness = cv::FILLED;
        if(!bBinary)
            color = {(double)brushColor.red(), (double)brushColor.green(), (double)brushColor.blue(), (double)brushColor.alpha()};
    }
    cv::Rect rcObj(rc.x(), rc.y(), rc.width(), rc.height());
    cv::rectangle(image, rcObj, color, thickness, cv::LINE_8);
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsEllipse *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Ellipse insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->rect());
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    int thickness;
    cv::Scalar color = {255, 255, 255, 255};
    QColor brushColor = pItem->brush().color();

    if(brushColor.alpha() == 0 && bForceFill == false)
    {
        thickness = pItem->pen().width();
        if(!bBinary)
        {
            QColor penColor = pItem->pen().color();
            color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
        }
    }
    else
    {
        thickness = cv::FILLED;
        if(!bBinary)
            color = {(double)brushColor.red(), (double)brushColor.green(), (double)brushColor.blue(), (double)brushColor.alpha()};
    }
    cv::RotatedRect rcObj(cv::Point2f(rc.topLeft().x(), rc.topLeft().y()),
                          cv::Point2f(rc.topRight().x(), rc.topRight().y()),
                          cv::Point2f(rc.bottomRight().x(), rc.bottomRight().y()));
    cv::ellipse(image, rcObj, color, thickness, cv::LINE_8);
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsPolygon *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Polygon insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->boundingRect());
    QRectF rcImg(0, 0, m_width, m_height);
    auto poly = pItem->polygon();

    if(!rcImg.contains(rc))
        poly = poly.intersected(QPolygonF(rcImg));

    std::vector<std::vector<cv::Point>> polygonArray;
    polygonArray.push_back(convertToCvPolygon(pItem, poly));

    cv::Scalar color = {255, 255, 255, 255};
    QColor brushColor = pItem->brush().color();

    if(brushColor.alpha() == 0 && bForceFill == false)
    {
        if(!bBinary)
        {
            QColor penColor = pItem->pen().color();
            color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
        }
        cv::polylines(image, polygonArray, true, color, pItem->pen().width(), cv::LINE_8);
    }
    else
    {
        if(!bBinary)
            color = {(double)brushColor.red(), (double)brushColor.green(), (double)brushColor.blue(), (double)brushColor.alpha()};

        cv::fillPoly(image, polygonArray, color, cv::LINE_8);
    }
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsPolyline *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);
    Q_UNUSED(bForceFill);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Polyline insertion failed : invalid image", __func__, __FILE__, __LINE__);

    cv::Scalar color = {255, 255, 255, 255};
    QPen pen = pItem->pen();

    if(!bBinary)
    {
        QColor penColor = pen.color();
        color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
    }
    std::vector<std::vector<cv::Point>> polylineArray;
    polylineArray.push_back(convertToCvPolygon(pItem, pItem->getPolygon()));
    cv::polylines(image, polylineArray, false, color, pen.width());
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsText *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);
    Q_UNUSED(bForceFill);
    Q_UNUSED(bBinary);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Text insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->boundingRect());
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        throw CException(CoreExCode::INVALID_DIMENSION, "Text does not fit to image", __func__, __FILE__, __LINE__);

    size_t startPos=0, endPos;
    std::string text = pItem->toPlainText().toStdString();
    bool bMultiline = (endPos = text.find('\n')) != std::string::npos;
    //auto pt = pItem->mapToScene(pItem->pos());
    cv::Point pos = cv::Point(rc.x(), rc.y());
    cv::Scalar color(pItem->defaultTextColor().red(), pItem->defaultTextColor().green(), pItem->defaultTextColor().blue());
    double fontScale = Utils::Font::getQtToOcvFontScaleFactor(pItem->font().pointSize());

    int thickness = 1;
    if (pItem->font().pointSize() > 16 && pItem->font().bold())
        thickness = 2;

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    if(pItem->font().italic())
        fontFace |= cv::FONT_ITALIC;

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    if(bMultiline)
    {
        while(endPos != std::string::npos)
        {
            std::string lineText = text.substr(startPos, endPos - startPos);
            cv::putText(image, lineText, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
            pos.y += textSize.height + 1;
            startPos = endPos + 1;
            endPos = text.find('\n', startPos);
        }
        std::string lineText = text.substr(startPos, endPos - startPos);
        cv::putText(image, lineText, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
    }
    else
    {
        int margin = 0.2 * textSize.height;
        // Light gray background
        cv::Scalar bckColor(200, 200, 200);
        cv::Rect rcBck(pos.x - margin, pos.y - textSize.height - margin, textSize.width + 2*margin, textSize.height + baseline + 2*margin);
        cv::rectangle(image, rcBck, bckColor, cv::FILLED);
        cv::putText(image, text, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
    }
}

void CGraphicsConversion::insertToImage(CMat &image, const CGraphicsComplexPolygon *pItem, bool bForceFill, bool bBinary)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Complex polygon insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->mapRectToScene(pItem->boundingRect());
    QRectF rcImg(0, 0, m_width, m_height);

    auto outerPoly = pItem->getOuterPolygon();
    if(!rcImg.contains(rc))
        outerPoly = outerPoly.intersected(QPolygonF(rcImg));

    std::vector<std::vector<cv::Point>> polygonArray;
    polygonArray.push_back(convertToCvPolygon(pItem, outerPoly));

    QList<QPolygonF> inners = pItem->getInnerPolygons();
    for(int i=0; i<inners.size(); ++i)
        polygonArray.push_back(convertToCvPolygon(pItem, inners[i]));

    cv::Scalar color = {255, 255, 255, 255};
    QColor brushColor = pItem->brush().color();

    if(brushColor.alpha() == 0 && bForceFill == false)
    {
        if(!bBinary)
        {
            QColor penColor = pItem->pen().color();
            color = {(double)penColor.red(), (double)penColor.green(), (double)penColor.blue(), (double)penColor.alpha()};
        }
        cv::polylines(image, polygonArray, true, color, pItem->pen().width(), cv::LINE_8);
    }
    else
    {
        if(!bBinary)
            color = {(double)brushColor.red(), (double)brushColor.green(), (double)brushColor.blue(), (double)brushColor.alpha()};

        cv::fillPoly(image, polygonArray, color, cv::LINE_8);
    }
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsPoint *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);
    Q_UNUSED(bForceFill);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Ellipse insertion failed : invalid image", __func__, __FILE__, __LINE__);

    float radius = (float)pItem->m_property.m_size / 2.0f;
    QRectF rc(pItem->m_point.m_x - radius, pItem->m_point.m_y - radius, pItem->m_property.m_size, pItem->m_property.m_size);
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    cv::Scalar color = {255, 255, 255, 255};
    if(!bBinary)
    {
        auto penColor = pItem->m_property.m_brushColor;
        if(bgr)
            color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
        else
            color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
    }
    cv::RotatedRect rcObj(cv::Point2f(rc.topLeft().x(), rc.topLeft().y()),
                          cv::Point2f(rc.topRight().x(), rc.topRight().y()),
                          cv::Point2f(rc.bottomRight().x(), rc.bottomRight().y()));
    cv::ellipse(image, rcObj, color, cv::FILLED);
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsRect *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Rectangle insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc(pItem->m_x, pItem->m_y, pItem->m_width, pItem->m_height);
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    int thickness;
    cv::Scalar color = {255, 255, 255, 255};
    auto brushColor = pItem->m_property.m_brushColor;

    if(brushColor[3] == 0 && bForceFill == false)
    {
        thickness = pItem->m_property.m_lineSize;
        if(!bBinary)
        {
            auto penColor = pItem->m_property.m_penColor;
            if(bgr)
                color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
            else
                color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
        }
    }
    else
    {
        thickness = cv::FILLED;
        if(!bBinary)
        {
            if(bgr)
                color = {(double)brushColor[2], (double)brushColor[1], (double)brushColor[0], (double)brushColor[3]};
            else
                color = {(double)brushColor[0], (double)brushColor[1], (double)brushColor[2], (double)brushColor[3]};
        }
    }
    cv::Rect rcObj(rc.x(), rc.y(), rc.width(), rc.height());
    cv::rectangle(image, rcObj, color, thickness, cv::LINE_8);
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsEllipse *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Ellipse insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc(pItem->m_x, pItem->m_y, pItem->m_width, pItem->m_height);
    QRectF rcImg(0, 0, m_width, m_height);

    if(!rcImg.contains(rc))
        rc = rc.intersected(rcImg);

    int thickness;
    cv::Scalar color = {255, 255, 255, 255};
    CColor brushColor = pItem->m_property.m_brushColor;

    if(brushColor[3] == 0 && bForceFill == false)
    {
        thickness = pItem->m_property.m_lineSize;
        if(!bBinary)
        {
            CColor penColor = pItem->m_property.m_penColor;
            if(bgr)
                color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
            else
                color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
        }
    }
    else
    {
        thickness = cv::FILLED;
        if(!bBinary)
        {
            if(bgr)
                color = {(double)brushColor[2], (double)brushColor[1], (double)brushColor[0], (double)brushColor[3]};
            else
                color = {(double)brushColor[0], (double)brushColor[1], (double)brushColor[2], (double)brushColor[3]};
        }
    }
    cv::RotatedRect rcObj(cv::Point2f(rc.topLeft().x(), rc.topLeft().y()),
                          cv::Point2f(rc.topRight().x(), rc.topRight().y()),
                          cv::Point2f(rc.bottomRight().x(), rc.bottomRight().y()));
    cv::ellipse(image, rcObj, color, thickness, cv::LINE_8);
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsPolygon *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Polygon insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->getBoundingQRect();
    QRectF rcImg(0, 0, m_width, m_height);
    std::vector<std::vector<cv::Point>> polygonArray;

    if(!rcImg.contains(rc))
        polygonArray.push_back(convertToCvPolygon(clipPolygon(pItem->m_points)));
    else
        polygonArray.push_back(convertToCvPolygon(pItem->m_points));

    cv::Scalar color = {255, 255, 255, 255};
    CColor brushColor = pItem->m_property.m_brushColor;

    if(brushColor[3] == 0 && bForceFill == false)
    {
        if(!bBinary)
        {
            CColor penColor = pItem->m_property.m_penColor;
            if(bgr)
                color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
            else
                color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
        }
        cv::polylines(image, polygonArray, true, color, pItem->m_property.m_lineSize, cv::LINE_8);
    }
    else
    {
        if(!bBinary)
        {
            if(bgr)
                color = {(double)brushColor[2], (double)brushColor[1], (double)brushColor[0], (double)brushColor[3]};
            else
                color = {(double)brushColor[0], (double)brushColor[1], (double)brushColor[2], (double)brushColor[3]};
        }
        cv::fillPoly(image, polygonArray, color, cv::LINE_8);
    }
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsPolyline *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);
    Q_UNUSED(bForceFill);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Polyline insertion failed : invalid image", __func__, __FILE__, __LINE__);

    cv::Scalar color = {255, 255, 255, 255};
    if(!bBinary)
    {
        CColor penColor = pItem->m_property.m_penColor;
        if(bgr)
            color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
        else
            color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
    }
    std::vector<std::vector<cv::Point>> polylineArray;
    polylineArray.push_back(convertToCvPolygon(pItem->m_points));
    cv::polylines(image, polylineArray, false, color, pItem->m_property.m_lineSize);
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsText *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);
    Q_UNUSED(bForceFill);
    Q_UNUSED(bBinary)

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Text insertion failed : invalid image", __func__, __FILE__, __LINE__);

    size_t startPos=0, endPos;
    bool bMultiline = (endPos = pItem->m_text.find('\n')) != std::string::npos;
    cv::Point pos = cv::Point(pItem->m_x, pItem->m_y);
    double fontScale = Utils::Font::getQtToOcvFontScaleFactor(pItem->m_property.m_fontSize);

    int thickness = 1;
    if (pItem->m_property.m_fontSize > 16 && pItem->m_property.m_bBold)
        thickness = 2;

    cv::Scalar color;
    if(bgr)
        color = cv::Scalar(pItem->m_property.m_color[2], pItem->m_property.m_color[1], pItem->m_property.m_color[0]);
    else
        color = cv::Scalar(pItem->m_property.m_color[0], pItem->m_property.m_color[1], pItem->m_property.m_color[2]);

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    if(pItem->m_property.m_bItalic)
        fontFace |= cv::FONT_ITALIC;

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(pItem->m_text, fontFace, fontScale, thickness, &baseline);

    if(bMultiline)
    {
        while(endPos != std::string::npos)
        {
            std::string lineText = pItem->m_text.substr(startPos, endPos - startPos);
            cv::putText(image, lineText, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
            pos.y += textSize.height + 1;
            startPos = endPos + 1;
            endPos = pItem->m_text.find('\n', startPos);
        }
        std::string lineText = pItem->m_text.substr(startPos, endPos - startPos);
        cv::putText(image, lineText, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
    }
    else
    {
        int margin = 0.2 * textSize.height;
        // Light gray background
        cv::Scalar bckColor(200, 200, 200);
        cv::Rect rcBck(pos.x - margin, pos.y - textSize.height - margin, textSize.width + 2*margin, textSize.height + baseline + 2*margin);
        cv::rectangle(image, rcBck, bckColor, cv::FILLED);
        cv::putText(image, pItem->m_text, pos, fontFace, fontScale, color, thickness, cv::LINE_AA);
    }
}

void CGraphicsConversion::insertToImage(CMat &image, const CProxyGraphicsComplexPoly *pItem, bool bForceFill, bool bBinary, bool bgr)
{
    assert(pItem);

    if(image.data == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Complex polygon insertion failed : invalid image", __func__, __FILE__, __LINE__);

    QRectF rc = pItem->getBoundingQRect();
    QRectF rcImg(0, 0, m_width, m_height);
    std::vector<std::vector<cv::Point>> polygonArray;

    if(!rcImg.contains(rc))
    {
        polygonArray.push_back(convertToCvPolygon(clipPolygon(pItem->m_outer)));
        for(size_t i=0; i<pItem->m_inners.size(); ++i)
            polygonArray.push_back(convertToCvPolygon(clipPolygon(pItem->m_inners[i])));
    }
    else
    {
        polygonArray.push_back(convertToCvPolygon(pItem->m_outer));
        for(size_t i=0; i<pItem->m_inners.size(); ++i)
            polygonArray.push_back(convertToCvPolygon(pItem->m_inners[i]));
    }

    cv::Scalar color = {255, 255, 255, 255};
    CColor brushColor = pItem->m_property.m_brushColor;

    if(brushColor[3] == 0 && bForceFill == false)
    {
        if(!bBinary)
        {
            CColor penColor = pItem->m_property.m_penColor;
            if(bgr)
                color = {(double)penColor[2], (double)penColor[1], (double)penColor[0], (double)penColor[3]};
            else
                color = {(double)penColor[0], (double)penColor[1], (double)penColor[2], (double)penColor[3]};
        }
        cv::polylines(image, polygonArray, true, color, pItem->m_property.m_lineSize, cv::LINE_8);
    }
    else
    {
        if(!bBinary)
        {
            if(bgr)
                color = {(double)brushColor[2], (double)brushColor[1], (double)brushColor[0], (double)brushColor[3]};
            else
                color = {(double)brushColor[0], (double)brushColor[1], (double)brushColor[2], (double)brushColor[3]};
        }
        cv::fillPoly(image, polygonArray, color, cv::LINE_8);
    }
}

std::vector<CPointF> CGraphicsConversion::convertToPolygon(const std::vector<cv::Point> &cvPoly)
{
    std::vector<CPointF> poly;

    if(cvPoly.size() > 1)
    {
        for(size_t j=0; j<cvPoly.size(); ++j)
            poly.push_back(CPointF(cvPoly[j].x, cvPoly[j].y));

        //Close polygon
        poly.push_back(CPointF(cvPoly[0].x, cvPoly[0].y));
    }
    return poly;
}

QPolygonF CGraphicsConversion::convertToQPolygon(const std::vector<cv::Point> &cvPoly)
{
    QPolygonF qtPoly;

    if(cvPoly.size() > 1)
    {
        for(size_t j=0; j<cvPoly.size(); ++j)
            qtPoly.push_back(QPointF(cvPoly[j].x, cvPoly[j].y));

        //Close polygon
        qtPoly.push_back(QPointF(cvPoly[0].x, cvPoly[0].y));
    }
    return qtPoly;
}

QPolygonF CGraphicsConversion::convertToQPolygon(const std::vector<CPointF> &poly)
{
    QPolygonF qtPoly;

    if(poly.size() > 1)
    {
        for(size_t j=0; j<poly.size(); ++j)
            qtPoly.push_back(QPointF(poly[j].m_x, poly[j].m_y));

        //Close polygon
        qtPoly.push_back(QPointF(poly[0].m_x, poly[0].m_y));
    }
    return qtPoly;
}

std::vector<cv::Point> CGraphicsConversion::convertToCvPolygon(const QGraphicsItem* pItem, const QPolygonF &qtPoly)
{
    std::vector<cv::Point> cvPoly;

    for(int i=0; i<qtPoly.size(); ++i)
    {
        auto pt = pItem->mapToScene(qtPoly[i]);
        cv::Point cvPt(pt.x(), pt.y());
        clipPoint(cvPt);
        cvPoly.push_back(cvPt);
    }
    return cvPoly;
}

std::vector<cv::Point> CGraphicsConversion::convertToCvPolygon(const std::vector<CPointF> &poly)
{
    std::vector<cv::Point> cvPoly;

    for(size_t i=0; i<poly.size(); ++i)
        cvPoly.push_back(cv::Point(poly[i].m_x, poly[i].m_y));

    return cvPoly;
}

void CGraphicsConversion::clipPoint(cv::Point &pt)
{
    if(pt.x < 0)
        pt.x = 0;
    else if(pt.x >= m_width)
        pt.x = m_width - 1;

    if(pt.y < 0)
        pt.y = 0;
    else if(pt.y >= m_height)
        pt.y = m_height - 1;
}

std::vector<CPointF> CGraphicsConversion::clipPolygon(const std::vector<CPointF> &pts)
{
    std::vector<CPointF> clipPts = pts;
    for(size_t i=0; i<clipPts.size(); ++i)
    {
        if(clipPts[i].m_x < 0)
            clipPts[i].m_x = 0;
        else if(clipPts[i].m_x >= (float)m_width)
            clipPts[i].m_x = (float)(m_width - 1);

        if(clipPts[i].m_y < 0)
            clipPts[i].m_y = 0;
        else if(clipPts[i].m_y >= (float)m_height)
            clipPts[i].m_y = (float)(m_height - 1);
    }
    return clipPts;
}
