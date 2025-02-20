﻿/*
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

#ifndef CWORKFLOWTASKIO_H
#define CWORKFLOWTASKIO_H

#include "Main/CoreDefine.hpp"
#include "DesignPattern/CAbstractFactory.hpp"
#include "DesignPattern/CObjectLocker.hpp"
#include "CWorkflowTaskParam.h"
#include "Data/CDataInfo.h"
#include "Data/CMat.hpp"
#include <QJsonDocument>

/**
 * @ingroup groupCore
 * @brief
 * This class is the base class for the management of input and output of a process.
 * Each process can have as many input and output, and each must be derived from this class.
 */
class CORESHARED_EXPORT CWorkflowTaskIO
{
    public:

        /**
         * @brief Default constructor.
         */
        CWorkflowTaskIO();
        /**
         * @brief Constructor with the given data type (::IODataType).
         */
        CWorkflowTaskIO(IODataType dataType);
        /**
         * @brief Constructor with the given data type (::IODataType) and name.
         */
        CWorkflowTaskIO(IODataType dataType, const std::string& name);
        /**
         * @brief Copy constructor.
         */
        CWorkflowTaskIO(const CWorkflowTaskIO& io);
        /**
         * @brief Universal reference copy constructor.
         */
        CWorkflowTaskIO(const CWorkflowTaskIO&& io);

        /**
         * @brief Assignement operator.
         */
        CWorkflowTaskIO&    operator=(const CWorkflowTaskIO& io);
        /**
         * @brief Universal reference assignement operator.
         */
        CWorkflowTaskIO&    operator=(const CWorkflowTaskIO&& io);

        friend CORESHARED_EXPORT std::ostream& operator<<(std::ostream& os, const CWorkflowTaskIO& io);

        virtual ~CWorkflowTaskIO();

        virtual std::string repr() const;

        //Getters
        /**
         * @brief Gets input or output name.
         * The name is used in the workflow view.
         * @return Name as std::string.
         */
        std::string         getName() const;
        /**
         * @brief Gets input or output description.
         * @return Description as std::string.
         */
        std::string         getDescription() const;
        /**
         * @brief Gets input or output data type.
         * @return Data type from enum values (::Data).
         */
        IODataType          getDataType() const;
        /**
         * @brief Get default export format.
         * @return File format (::DataExportFormat).
         */
        DataFileFormat      getSaveFormat() const;
        /**
         * @brief Get possible export formats.
         * @return Vector of file formats (::DataExportFormat).
         */
        virtual std::vector<DataFileFormat> getPossibleSaveFormats() const;
        /**
         * @brief Get full save path.
         * @return Save path as std::string.
         */
        std::string         getSavePath() const;
        /**
         * @brief Gets the number of dimensions for input or output. For example, IMAGE = 2 and VOLUME = 3.
         * @return Dimension count as size_t.
         */
        size_t              getDimensionCount() const;
        /**
         * @brief Gets the number of unit element in terms of processing scheme.
         * This value is used to define the number of progress steps for progress bar component.
         * For an image, the count is 1. For Z-stack volume, the count is the number of Z levels.
         * Should be overriden for custom input or output.
         * @return Unit element count as size_t.
         */
        virtual size_t      getUnitElementCount() const;
        /**
         * @brief Gets the information associated with the input/output.
         * @return Object of CDataInfoPtr class or derived.
         */
        virtual CDataInfoPtr    getDataInfo();
        /**
         * @brief Gets the source file path from global workflow input
         * @return std::string
         */
        std::string         getSourceFilePath();

        /**
         * @brief Gets sub IO objects of given types in case of composite IO
         */
        virtual std::vector<std::shared_ptr<CWorkflowTaskIO>>   getSubIOList(const std::set<IODataType> &dataTypes) const;

        /**
         * @brief Generate visual image with graphics
         */
        virtual CMat        getImageWithGraphics(const CMat& image) const;
        /**
         * @brief Generate visual image with mask as overlay
         */
        virtual CMat        getImageWithMask(const CMat& image) const;
        /**
         * @brief Generate visual image with graphics and mask as overlay
         */
        virtual CMat        getImageWithMaskAndGraphics(const CMat& image) const;

        /**
         * @brief Checks if input or output object contains valid data.
         * In input, it is the case when the previous process contains valid output data.
         * In output, It is the case when the associated process had been computed at least once.
         * Should be overriden for custom input or output.
         * @return True if there is valid data stored, False otherwise.
         */
        virtual bool        isDataAvailable() const;
        /**
          * @brief Check whether input/output is a combination of atomic I/O
          * @return True or False.
          */
        virtual bool        isComposite() const;
        /**
          * @brief Check whether data has to be saved automatically.
          * @return True or False.
          */
        bool                isAutoSave() const;
        /**
          * @brief Check whether output has to be displayed.
          * @return True or False.
          */
        bool                isDisplayable() const;

        //Setters
        /**
         * @brief Sets the data type for input or output.
         * @param type can be one of those ::Data.
         */
        void                setDataType(IODataType type);
        /**
         * @brief Sets the name for input or output.
         * @param name as std::string.
         */
        void                setName(const std::string& name);
        /**
         * @brief Set the number of dimension.
         * @param nb : size_t
         */
        void                setDimensionCount(size_t nb);
        /**
         * @brief Set information used by save() method to build save path automatically.
         * Used for auto-save outputs when batch mode is enabled.
         * @param folder : output folder
         * @param baseName : base file name which will be completed with custom data (extension, numbering...)
         */
        void                setSaveInfo(const std::string& folder, const std::string& baseName);

        void                setSavePath(const std::string& path);
        /**
         * @brief Sets the input/output description.
         * @param description as std::string.
         */
        void                setDescription(const std::string& description);
        /**
         * @brief Enable auto export.
         * @param bool.
         */
        void                setAutoSave(bool bEnable);
        /**
         * @brief Set default save format.
         * @param format (DataFileFormat).
         */
        void                setSaveFormat(DataFileFormat format);
        /**
         * @brief Make output displayable or not
         * @param bool.
         */
        void                setDisplayable(bool bEnable);

        /**
         * @brief Clears the data stored in the object.
         * Should be overriden for custom input or output.
         */
        virtual void        clearData();

        /**
         * @brief Makes a deep copy and returns it.
         * @return Shared pointer.
         */
        std::shared_ptr<CWorkflowTaskIO>    clone() const;

        /**
         * @brief Copies the given input or output.
         * @param Input or ouput as a constant shared pointer.
         */
        virtual void        copy(const std::shared_ptr<CWorkflowTaskIO>& ioPtr);
        /**
         * @brief Copies the static data from the given input or ouput.
         * Static data are those which are not generated at runtime.
         * Should be overriden for custom input or output.
         * @param Input or ouput as a constant shared pointer.
         */
        virtual void        copyStaticData(const std::shared_ptr<CWorkflowTaskIO>& ioPtr);

        /**
         * @brief Appends file to the temporary files list.
         * @brief Temporary files are removed when object is destroyed.
         * @param path (std::string): full path of temporary file.
         */
        void                addTemporaryFile(const std::string& path);

        virtual void        load(const std::string& path);

        /**
         * @brief Save the input or output data to disk using information set with setSaveInfo().
         * Should be overriden for custom input or output.
         */
        virtual void        save();

        /*
         * @brief Save the input or output data to disk.
         * Should be overriden for custom input or output.
         * @param path: full path.
         */
        virtual void        save(const std::string& path);

        /**
         * @brief Static function to get the class name of input or output according to the data type.
         * @param ioDataType : data type see ::Data
         * @param bInput: indicate if it is input or output
         * @return name given by the system for the given data type.
         */
        static std::string  getClassName(IODataType ioDataType);

        /**
        * @brief Return input/output data in JSON formatted string with default options
        * @return JSON formatted string
        */
        virtual std::string toJson() const;
        /**
        * @brief Return input/output data in JSON formatted string
        * @param options: format-specific options encoded as pairs (option_name, option_value)
        * @return JSON formatted string
        */
        virtual std::string toJson(const std::vector<std::string>& options) const;

        /**
         * @brief Set input/output data from JSON formatted string
         * @param jsonStr: JSON formatted string
         */
        virtual void        fromJson(const std::string& jsonStr);

    protected:

        std::string         toFormattedJson(const QJsonDocument& doc, const std::vector<std::string>& options) const;
        virtual void        to_ostream(std::ostream& os) const;

    private:

        virtual std::shared_ptr<CWorkflowTaskIO> cloneImp() const;

        void                deleteTemporaryFiles();

    protected:

        std::string             m_name = "";
        std::string             m_savePath;
        std::string             m_description;
        IODataType              m_dataType = IODataType::NONE;
        DataFileFormat          m_saveFormat = DataFileFormat::NONE;
        size_t                  m_dimCount = 0;
        CDataInfoPtr            m_infoPtr = nullptr;
        bool                    m_bNewDataInfo = false;
        bool                    m_bAutoSave = false;
        bool                    m_bDisplayable = true;
        mutable std::mutex      m_mutex;
        std::set<std::string>   m_tempFiles;
        friend class CObjectLocker<CWorkflowTaskIO>;
};

using WorkflowTaskParamPtr = std::shared_ptr<CWorkflowTaskParam>;
using WorkflowTaskIOPtr = std::shared_ptr<CWorkflowTaskIO>;
using InputOutputVect = std::vector<std::shared_ptr<CWorkflowTaskIO>>;
using InputOutputSet = std::set<std::shared_ptr<CWorkflowTaskIO>>;
using TaskIOLockerUPtr = std::unique_ptr<CObjectLocker<CWorkflowTaskIO>>;

class CWorkflowTaskIOFactory
{
    public:

        virtual ~CWorkflowTaskIOFactory(){}

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            return std::make_shared<CWorkflowTaskIO>(dataType);
        }

        std::string                 getName() const
        {
            return m_name;
        }

        void                        setName(const std::string& name)
        {
            m_name = name;
        }

    protected:

        std::string m_name = "default";
};

using TaskIOFactoryPtr = std::shared_ptr<CWorkflowTaskIOFactory>;
using WorkflowTaskIOFactories = std::vector<TaskIOFactoryPtr>;

//----- Process abstract factory -----//
class CWorkflowTaskIOAbstractFactory: public CAbstractFactory<std::string, WorkflowTaskIOPtr, IODataType>
{
    public:

        WorkflowTaskIOFactories& getList()
        {
            return m_factories;
        }

        void                    clear() override
        {
            CAbstractFactory<std::string, WorkflowTaskIOPtr, IODataType>::clear();
            m_factories.clear();
        }

    private:

        WorkflowTaskIOFactories m_factories;
};

#endif // CWORKFLOWTASKIO_H
