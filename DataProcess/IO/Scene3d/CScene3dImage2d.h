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

#ifndef CSCENE3DIMAGE2D_H
#define CSCENE3DIMAGE2D_H

#include <memory>

#include "Data/CMat.hpp"
#include "CScene3dObject.h"


class CScene3dImage2d;

/**
 * Alias onto the CScene3dImage2d's shared pointer type.
 */
using CScene3dImage2dPtr = std::shared_ptr<CScene3dImage2d>;


/**
 * @brief The CScene3dImage2d class represents a 2D image inside the 3D scene.
 * This image is always oriented along the (x,y) axis.
 */
class CScene3dImage2d : public CScene3dObject
{

public:
    /**
     * @brief Default constructor.
     */
    CScene3dImage2d();

    /**
     * @brief Custom constructor.
     * @param data: a CMat class containing the value of each pixel.
     * @param isVisible: true if the image should be displayed, false otherwise.
     */
    CScene3dImage2d(const CMat &data, bool isVisible);

    /**
     * @brief Copy constructor.
     */
    CScene3dImage2d(const CScene3dImage2d &img);

    /**
     * @brief Assignment operator.
     */
    CScene3dImage2d& operator = (const CScene3dImage2d& img);

    /**
     * @brief Return the image's data, a CMat class containing the value of each pixel.
     */
    CMat getData() const;

    /**
     * @brief Set the image's data.
     * @param data: a CMat class containing the value of each pixel.
     */
    void setData(const CMat &data);

    /**
     * @brief Return the image's width.
     */
    std::size_t getWidth() const;

    /**
     * @brief Return the image's height.
     */
    std::size_t getHeight() const;

    /**
     * @brief Static method used to create an new 'CScene3dImage2d' instance.
     * @param data: a CMat class containing the value of each pixel.
     * @param isVisible: true if the image should be displayed, false otherwise.
     * @return Return a shared_ptr of the created instance.
     */
    static CScene3dImage2dPtr create(
        const CMat &data,
        bool isVisible
    );

protected:
    /**
     * A CMat class containing the value of each pixel.
     */
    CMat m_data;
};

#endif // CSCENE3DIMAGE2D_H
