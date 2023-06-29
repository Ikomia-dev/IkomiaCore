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

#ifndef CJSONIO_H
#define CJSONIO_H

#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"



/**
 * @ingroup groupDataProcess
 * @brief The CJsonIO class defines an input or output for a workflow task that can be used to store data using the JSON format.
 * @details This class is designed to handle data as input or output of a workflow task.
 * A CJsonIO instance can be added as input or output to a CWorkflowTask or derived object.
 */
class DATAPROCESSSHARED_EXPORT CJsonIO: public CWorkflowTaskIO
{
    public:
        /**
         * @brief Default constructor
         */
        CJsonIO();

        /**
         * @brief Construct a CJsonIO instance with the given name.
         * @param name: std::string.
         */
        CJsonIO(const std::string& name);

        /**
         * @brief Construct a CJsonIO instance with the given data. Data must be a Qt's JSON document.
         * @param array: QJsonDocument object for C++ and ??? for Python.
         */
        CJsonIO(const QJsonDocument& rootJSON, const std::string& name="CJsonIO");

        /**
         * @brief Copy constructor.
         */
        CJsonIO(const CJsonIO& io);

        /**
         * @brief Universal reference copy constructor.
         */
        CJsonIO(const CJsonIO&& io);

        /**
         * @brief Destructor.
         */
        virtual ~CJsonIO();

        /**
         * @brief Assignment operator.
         */
        CJsonIO& operator=(const CJsonIO& io);

        /**
         * @brief Universal reference assignment operator.
         */
        CJsonIO& operator=(const CJsonIO&& io);

        /**
         * @brief
         */
        std::string repr() const override;

        /**
         * @brief Checks whether the input/output have valid data or not.
         * @return True if data is not empty, False otherwise.
         */
        bool isDataAvailable() const override;

        /**
         * @brief Clears data: the root data member stays valid but without data (isDataAvailable() == false).
         */
        void clearData() override;

        /**
         * @brief Return the whole JSON document.
         */
        QJsonDocument getData() const;

        /**
         * @brief Replaces existing data by a new JSON document.
         */
        void setData(const QJsonDocument& doc);

        /**
         * @brief Loads data from a file. An exception is thrown if data are not valid.
         */
        void load(const std::string &path) override;

        /**
         * @brief Writes data to a file.
         */
        void save(const std::string &path) override;

        /**
         * @brief Converts the JSON tree to a string.
         * @details Indentation of the result string can be controlled by the 'format' parameter.
         * @param format: QJsonDocument::Compact = no indent / QJsonDocument::Indent = default indentation
         */
        std::string toString(QJsonDocument::JsonFormat format=QJsonDocument::Compact) const;

        /**
         * @brief Parses a std::string to create a JSON tree.
         */
        void fromString(const std::string &str);

        /**
         * @brief Performs a deep copy of this instance.
         * @return CJsonIO smart pointer (std::shared_ptr).
         */
        std::shared_ptr<CJsonIO> clone() const;


    private:
        WorkflowTaskIOPtr cloneImp() const override;


    private:
        QJsonDocument m_rootJSON;
};

using JsonIOPtr = std::shared_ptr<CJsonIO>;

class DATAPROCESSSHARED_EXPORT CJsonIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CJsonIOFactory()
        {
            m_name = "CJsonIO";
        }

        virtual WorkflowTaskIOPtr create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CJsonIO>();
        }
};

#endif // CJSONIO_H
