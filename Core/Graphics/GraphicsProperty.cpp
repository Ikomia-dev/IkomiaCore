#include "GraphicsProperty.h"
#include "Main/CoreTools.hpp"

//----------------------------//
//- CGraphicsEllipseProperty -//
//----------------------------//
Ikomia::CColor CGraphicsEllipseProperty::getPenColor() const
{
    return m_penColor;
}

CColor CGraphicsEllipseProperty::getBrushColor() const
{
    return m_brushColor;
}

void CGraphicsEllipseProperty::setPenColor(const CColor &color)
{
    m_penColor = color;
}

void CGraphicsEllipseProperty::setBrushColor(const CColor &color)
{
    m_brushColor = color;
}

void CGraphicsEllipseProperty::fromJson(const QJsonObject &obj)
{
    m_penColor = Utils::Graphics::colorFromJson(obj["pen"].toObject());
    m_brushColor = Utils::Graphics::colorFromJson(obj["brush"].toObject());
    m_lineSize = obj["line"].toInt();
    m_category = obj["category"].toString().toStdString();
}

//--------------------------//
//- CGraphicsPointProperty -//
//--------------------------//
CColor CGraphicsPointProperty::getPenColor() const
{
    return m_penColor;
}

CColor CGraphicsPointProperty::getBrushColor() const
{
    return m_brushColor;
}

void CGraphicsPointProperty::setPenColor(const CColor &color)
{
    m_penColor = color;
}

void CGraphicsPointProperty::setBrushColor(const CColor &color)
{
    m_brushColor = color;
}

void CGraphicsPointProperty::fromJson(const QJsonObject &obj)
{
    m_penColor = Utils::Graphics::colorFromJson(obj["pen"].toObject());
    m_brushColor = Utils::Graphics::colorFromJson(obj["brush"].toObject());
    m_size = obj["size"].toInt();
    m_category = obj["category"].toString().toStdString();
}

//----------------------------//
//- CGraphicsPolygonProperty -//
//----------------------------//
CColor CGraphicsPolygonProperty::getPenColor() const
{
    return m_penColor;
}

CColor CGraphicsPolygonProperty::getBrushColor() const
{
    return m_brushColor;
}

void CGraphicsPolygonProperty::setPenColor(const CColor &color)
{
    m_penColor = color;
}

void CGraphicsPolygonProperty::setBrushColor(const CColor &color)
{
    m_brushColor = color;
}

void CGraphicsPolygonProperty::fromJson(const QJsonObject &obj)
{
    m_penColor = Utils::Graphics::colorFromJson(obj["pen"].toObject());
    m_brushColor = Utils::Graphics::colorFromJson(obj["brush"].toObject());
    m_lineSize = obj["line"].toInt();
    m_category = obj["category"].toString().toStdString();
}

//-----------------------------//
//- CGraphicsPolylineProperty -//
//-----------------------------//
CColor CGraphicsPolylineProperty::getColor() const
{
    return m_penColor;
}

void CGraphicsPolylineProperty::setColor(const CColor &color)
{
    m_penColor = color;
}

void CGraphicsPolylineProperty::fromJson(const QJsonObject &obj)
{
    m_penColor = Utils::Graphics::colorFromJson(obj["pen"].toObject());
    m_lineSize = obj["line"].toInt();
    m_category = obj["category"].toString().toStdString();
}

//-------------------------//
//- CGraphicsRectProperty -//
//-------------------------//
CColor CGraphicsRectProperty::getPenColor() const
{
    return m_penColor;
}

CColor CGraphicsRectProperty::getBrushColor() const
{
    return m_brushColor;
}

void CGraphicsRectProperty::setPenColor(const CColor &color)
{
    m_penColor = color;
}

void CGraphicsRectProperty::setBrushColor(const CColor &color)
{
    m_brushColor = color;
}

void CGraphicsRectProperty::fromJson(const QJsonObject &obj)
{
    m_penColor = Utils::Graphics::colorFromJson(obj["pen"].toObject());
    m_brushColor = Utils::Graphics::colorFromJson(obj["brush"].toObject());
    m_lineSize = obj["line"].toInt();
    m_category = obj["category"].toString().toStdString();
}

//-------------------------//
//- CGraphicsTextProperty -//
//-------------------------//
CColor CGraphicsTextProperty::getColor() const
{
    return m_color;
}

void CGraphicsTextProperty::setColor(const CColor &color)
{
    m_color = color;
}

void CGraphicsTextProperty::fromJson(const QJsonObject &obj)
{
    m_color = Utils::Graphics::colorFromJson(obj["color"].toObject());
    m_fontName = obj["font"].toString().toStdString();
    m_fontSize = obj["size"].toInt();
    m_bBold = obj["bold"].toBool();
    m_bItalic = obj["italic"].toBool();
    m_bUnderline = obj["underline"].toBool();
    m_bStrikeOut = obj["strikeout"].toBool();
    m_category = obj["category"].toString().toStdString();
}
