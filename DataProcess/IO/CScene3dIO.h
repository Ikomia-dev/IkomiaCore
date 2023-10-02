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

#ifndef CSCENE3DIO_H
#define CSCENE3DIO_H

#include "DataProcessGlobal.hpp"
#include "IO/Scene3d/CScene3d.h"
#include "Workflow/CWorkflowTaskIO.h"


/**
 * @ingroup groupDataProcess
 * @brief The CScene3dIO class defines an input or output for a workflow task dedicated to 3D scene management.
 * @details This class is designed to handle 3d scene as input or output of a workflow task.
 * A 3D scene can contain images, shapes like points, circles or polygons...
 * No visual representation is done with this class.
 */
class DATAPROCESSSHARED_EXPORT CScene3dIO : public CWorkflowTaskIO
{
    public:
        /**
         * @brief Default constructor.
         */
        CScene3dIO();

        /**
         * @brief Custom constructor.
         * @param name: input or output name.
         */
        CScene3dIO(const std::string& name);

        /**
         * @brief Copy constructor.
         */
        CScene3dIO(const CScene3dIO& io);

        /**
         * @brief Universal reference copy constructor.
         */
        CScene3dIO(const CScene3dIO&& io);

        /**
         * @brief Destructor.
         */
        virtual ~CScene3dIO() = default;

        /**
         * @brief Assignment operator.
         */
        CScene3dIO& operator=(const CScene3dIO& io);

        /**
         * @brief Universal reference assignment operator.
         */
        CScene3dIO& operator=(const CScene3dIO&& io);

        /**
         * @brief Compute a string used to represent the class.
         */
        std::string repr() const override;

        /**
         * @brief Checks whether the input/output have valid data or not.
         * @return True if data is not empty, False otherwise.
         */
        bool isDataAvailable() const override;

        /**
         * @brief Accessor associated to the 'm_scene3d' attribute.
         * @return Reference onto the 3D scene.
         */
        const CScene3d& getScene3d() const;

        /**
         * @brief Mutator associated to the 'm_scene3d' attribute.
         * @param The new 3d scene.
         */
        void setScene3d(const CScene3d& scene3d);

        /**
         * @brief Performs a deep copy the this instance
         * @return CScene3dIO smart pointer (std::shared_ptr).
         */
        std::shared_ptr<CScene3dIO> clone() const;

    private:
        virtual std::shared_ptr<CWorkflowTaskIO> cloneImp() const override;

    protected:
        CScene3d m_scene3d;
};

using CScene3dIOPtr = std::shared_ptr<CScene3dIO>;

class DATAPROCESSSHARED_EXPORT CScene3dIOFactory: public CWorkflowTaskIOFactory
{
    public:
        CScene3dIOFactory()
        {
            m_name = "CScene3dIO";
        }

        virtual WorkflowTaskIOPtr create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CScene3dIO>();
        }
};

#endif // CSCENE3DIO_H
