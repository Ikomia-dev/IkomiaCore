/*
 * Copyright (C) 2023 Ikomia SAS
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

#include "CScene3dVectorField.h"
#include "DataProcessTools.hpp"

#include <QJsonArray>
#include <QJsonObject>


CScene3dVectorField::CScene3dVectorField() :
    CScene3dObject(),
    m_data(CMat()),
    m_scaleFactor(1.0)
{ }

CScene3dVectorField::CScene3dVectorField(const CMat &data, double scaleFactor, bool isVisible) :
    CScene3dObject(isVisible),
    m_data(data),
    m_scaleFactor(scaleFactor)
{ }

CScene3dVectorField::CScene3dVectorField(const CScene3dVectorField &vf) :
    CScene3dObject(vf),
    m_data(vf.getData()),
    m_scaleFactor(vf.getScaleFactor())
{ }

CScene3dVectorField& CScene3dVectorField::operator = (const CScene3dVectorField& vf)
{
    CScene3dObject::operator = (vf);

    m_data = vf.getData();
    m_scaleFactor = vf.getScaleFactor();

    return *this;
}

CMat CScene3dVectorField::getData() const
{
    return m_data;
}

void CScene3dVectorField::setData(const CMat &data)
{
    m_data = data;
}

double CScene3dVectorField::getScaleFactor() const
{
    return m_scaleFactor;
}

void CScene3dVectorField::setScaleFactor(double scaleFactor)
{
    m_scaleFactor = scaleFactor;
}

QJsonObject CScene3dVectorField::toJson() const
{
    QJsonObject obj = CScene3dObject::toJson();

    // The CMat is converted into a QJsonArray
    QJsonArray dataArray;
    for(std::size_t i = 0; i < m_data.getNbRows(); ++i)
    {
        for(std::size_t j = 0; j < m_data.getNbCols(); ++j)
        {
            dataArray.push_back(m_data.at<double>(i, j));
        }
    }

    // JSON data
    obj["kind"] = "VECTOR_FIELD";
    obj["nbRows"] = static_cast<int>(m_data.getNbRows());
    obj["nbCols"] = static_cast<int>(m_data.getNbCols());
    obj["cvDataType"] = static_cast<int>(m_data.type());
    obj["dataArray"] = dataArray;
    obj["scaleFactor"] = m_scaleFactor;

    return obj;
}

CScene3dVectorFieldPtr CScene3dVectorField::fromJson(const QJsonObject& obj)
{
    if(obj["kind"] != "VECTOR_FIELD")
    {
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Invalid object type: 'VECTOR_FIELD' expected", __func__, __FILE__, __LINE__);
    }

    // JSON data
    int nbRows = obj["nbRows"].toInt();
    int nbCols = obj["nbCols"].toInt();
    int dataType = obj["cvDataType"].toInt();
    double scaleFactor = obj["scaleFactor"].toDouble();
    bool isVisible = obj["isVisible"].toBool();

    // The original CMat is reconstructed from data stored into the "dataArray" array
    QJsonArray dataArray = obj["dataArray"].toArray();
    CMat data(nbRows, nbCols, dataType);

    std::size_t index = 0;
    for(std::size_t i = 0; i < nbRows; ++i)
    {
        for(std::size_t j = 0; j < nbCols; ++j)
        {
            data.at<double>(i, j) = dataArray[index].toDouble();
            ++index;
        }
    }


    return CScene3dVectorField::create(
        data,
        scaleFactor,
        isVisible
    );
}

CScene3dVectorFieldPtr CScene3dVectorField::create(const CMat &data, double scaleFactor, bool isVisible)
{
    return std::make_shared<CScene3dVectorField>(data, scaleFactor, isVisible);
}
