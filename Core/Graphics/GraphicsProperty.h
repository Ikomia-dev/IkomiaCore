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

#ifndef GRAPHICSPROPERTY_H
#define GRAPHICSPROPERTY_H

#include "Main/CoreDefine.hpp"
#include <QJsonObject>

/** @file GraphicsProperty.h */

//----------------------------//
//- CGraphicsEllipseProperty -//
//----------------------------//
class CGraphicsEllipseProperty
{
    public:

        CColor      getPenColor() const;
        CColor      getBrushColor() const;

        void        setPenColor(const CColor& color);
        void        setBrushColor(const CColor& color);

        void        fromJson(const QJsonObject& obj);

    public:

        CColor      m_penColor = {255,0,0,255};
        CColor      m_brushColor =  {255,0,0,0};
        int         m_lineSize = 2;
        std::string m_category = "Default";
};

//--------------------------//
//- CGraphicsPointProperty -//
//--------------------------//
class CGraphicsPointProperty
{
    public:

        CColor      getPenColor() const;
        CColor      getBrushColor() const;

        void        setPenColor(const CColor& color);
        void        setBrushColor(const CColor& color);

        void        fromJson(const QJsonObject& obj);

    public:

        CColor      m_penColor = {255,0,0,255};
        CColor      m_brushColor = {255,0,0,255};
        int         m_size = 3;
        std::string m_category = "Default";
};

//----------------------------//
//- CGraphicsPolygonProperty -//
//----------------------------//
class CGraphicsPolygonProperty
{
    public:

        CColor      getPenColor() const;
        CColor      getBrushColor() const;

        void        setPenColor(const CColor& color);
        void        setBrushColor(const CColor& color);

        void        fromJson(const QJsonObject &obj);

    public:

        CColor      m_penColor = {255,0,0,255};
        CColor      m_brushColor = {255,0,0,0};
        int         m_lineSize = 2;
        std::string m_category = "Default";
};

//-----------------------------//
//- CGraphicsPolylineProperty -//
//-----------------------------//
class CGraphicsPolylineProperty
{
    public:

        CColor      getColor() const;
        void        setColor(const CColor& color);
        void        fromJson(const QJsonObject &obj);

    public:

        CColor      m_penColor = {255,0,0,255};
        int         m_lineSize = 2;
        std::string m_category = "Default";
};

//-------------------------//
//- CGraphicsRectProperty -//
//-------------------------//
class CGraphicsRectProperty
{
    public:

        CColor      getPenColor() const;
        CColor      getBrushColor() const;

        void        setPenColor(const CColor& color);
        void        setBrushColor(const CColor& color);

        void        fromJson(const QJsonObject& obj);

    public:

        CColor      m_penColor = {255,0,0,255};
        CColor      m_brushColor = {255,0,0,0};
        int         m_lineSize = 2;
        std::string m_category = "Default";
};

//-------------------------//
//- CGraphicsTextProperty -//
//-------------------------//
class CGraphicsTextProperty
{
    public:

        CColor      getColor() const;
        void        setColor(const CColor& color);
        void        fromJson(const QJsonObject& obj);

    public:

        CColor      m_color = {255,0,0,255};
        std::string m_fontName = "Helvetica";
        int         m_fontSize = 12;
        bool        m_bBold = false;
        bool        m_bItalic = false;
        bool        m_bUnderline = false;
        bool        m_bStrikeOut = false;
        std::string m_category = "Default";
};

#endif // GRAPHICSPROPERTY_H
