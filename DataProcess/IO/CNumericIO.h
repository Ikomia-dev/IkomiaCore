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

#ifndef CNUMERICIO_H
#define CNUMERICIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "Main/CoreTools.hpp"
#include <QJsonDocument>
#include <QJsonArray>

/** @file CNumericIO.hpp */

/**
 * @enum  NumericOutputType
 * @brief Output display type for numeric values
 */
enum class NumericOutputType
{
    NONE = 0,           /**< No display */
    TABLE = 2,          /**< Table display */
    PLOT = 4,           /**< Plot display */
    NUMERIC_FILE = 8    /**< Save to file */
};

/**
 * @enum  PlotType
 * @brief Plot type to display numeric values
 */
enum class PlotType
{
    CURVE,      /**< Curve */
    BAR,        /**< Bar diagram */
    MULTIBAR,   /**< Multi-bar diagram */
    HISTOGRAM,  /**< Histogram */
    PIE         /**< Pie chart */
};

class CNumericIOBase: public CWorkflowTaskIO
{
    public:

        using StringVector = std::vector<std::string>;
        using VectorOfStringVector = std::vector<std::vector<std::string>>;

        CNumericIOBase(): CWorkflowTaskIO(IODataType::NUMERIC_VALUES, "NumericIO")
        {
        }
        CNumericIOBase(const std::string& name): CWorkflowTaskIO(IODataType::NUMERIC_VALUES, name)
        {
        }
        CNumericIOBase(const CNumericIOBase& io) : CWorkflowTaskIO(io)
        {
            m_outputType = io.m_outputType;
            m_plotType = io.m_plotType;
        }
        CNumericIOBase(const CNumericIOBase&& io) : CWorkflowTaskIO(io)
        {
            m_outputType = std::move(io.m_outputType);
            m_plotType = std::move(io.m_plotType);
        }

        virtual ~CNumericIOBase(){}

        virtual CNumericIOBase&  operator=(const CNumericIOBase& io)
        {
            CWorkflowTaskIO::operator=(io);
            m_outputType = io.m_outputType;
            m_plotType = io.m_plotType;
            return *this;
        }
        virtual CNumericIOBase&  operator=(const CNumericIOBase&& io)
        {
            CWorkflowTaskIO::operator=(io);
            m_outputType = std::move(io.m_outputType);
            m_plotType = std::move(io.m_plotType);
            return *this;
        }

        /**
         * @brief Set the output display type used to visualize numeric values.
         * @param type : see ::NumericOutputType for details.
         */
        inline void                 setOutputType(NumericOutputType type)
        {
            m_outputType = type;
        }
        /**
         * @brief Set the plot type used to visualize numeric values.
         * @details This property is used only if the output display type is set to ::NumericOutputType::PLOT. Use setOutputType method to set the output display type.
         * @param type : see ::PlotType for details.
         */
        inline void                 setPlotType(PlotType type)
        {
            m_plotType = type;
        }

        /**
         * @brief Gets current output display type.
         * @return Output display type. See ::NumericOutputType for details.
         */
        inline NumericOutputType    getOutputType() const
        {
            return m_outputType;
        }
        /**
         * @brief Gets current plot type.
         * @return Plot type. See ::PlotType for details.
         */
        inline PlotType             getPlotType() const
        {
            return m_plotType;
        }
        /**
         * @brief Gets associated label list at index i.
         * @param i: index of the desired label list.
         * @return Vector of labels.
         */
        inline StringVector         getValueLabelList(size_t index) const
        {
            if(index < m_valueLabels.size())
                return m_valueLabels[index];
            else
                return std::vector<std::string>();
        }
        /**
         * @brief Gets associated header label for list at index i.
         * @param i: index of the desired list.
         * @return Header label.
         */
        inline std::string          getHeaderLabel(size_t index) const
        {
            if(index < m_headerLabels.size())
                return m_headerLabels[index];
            else
                return std::string();
        }
        /**
         * @brief Get all associated label lists.
         * @return Vector of vector of labels.
         */
        inline VectorOfStringVector getAllValueLabels() const
        {
            return m_valueLabels;
        }
        /**
         * @brief Get all associated header labels.
         * @return Vector of labels.
         */
        inline StringVector         getAllHeaderLabels() const
        {
            return m_headerLabels;
        }

        virtual VectorOfStringVector getAllValuesAsString() const = 0;

    protected:

        NumericOutputType                       m_outputType = NumericOutputType::TABLE;
        PlotType                                m_plotType = PlotType::CURVE;
        std::vector<std::string>                m_headerLabels;
        std::vector<std::vector<std::string>>   m_valueLabels;
};

/**
 * @ingroup groupCore
 * @brief The CNumericIO class defines an input or output for a workflow task dedicated to numeric values management.
 * @details This class is designed to handle numeric values as input or output of a workflow task. A CNumericIO<T> instance can be added as input or output to a CWorkflowTask or derived object.
 * It consists on a list of values (generic type), a list of associated labels and a display type (::NumericOutputType).
 * For the specific case of plot display, a plot type property is available (::PlotType).
 */
template<class Type>
class CNumericIO : public CNumericIOBase
{
    public:

        using NumericValues = std::vector<Type>;
        using VectorOfNumericValues = std::vector<std::vector<Type>>;
        using NumericIOPtr = std::shared_ptr<CNumericIO>;

        /**
         * @brief Default constructor
         */
        CNumericIO() : CNumericIOBase()
        {
            m_description = QObject::tr("Numerical values structured as table data (headers, labels and values).\n"
                                        "Can be displayed as table or plot.").toStdString();
            m_saveFormat = DataFileFormat::CSV;
        }

        CNumericIO(const std::string& name) : CNumericIOBase(name)
        {
            m_description = QObject::tr("Numerical values structured as table data (headers, labels and values).\n"
                                        "Can be displayed as table or plot.").toStdString();
            m_saveFormat = DataFileFormat::CSV;
        }
        /**
         * @brief Copy constructor
         */
        CNumericIO(const CNumericIO& io) : CNumericIOBase(io)
        {
            m_values = io.m_values;
            m_valueLabels = io.m_valueLabels;
            m_headerLabels = io.m_headerLabels;
        }
        /**
         * @brief Universal reference copy constructor
         */
        CNumericIO(const CNumericIO&& io) : CNumericIOBase(io)
        {
            m_values = std::move(io.m_values);
            m_valueLabels = std::move(io.m_valueLabels);
            m_headerLabels = std::move(io.m_headerLabels);
        }

        virtual ~CNumericIO(){}

        /**
         * @brief Assignement operator
         */
        virtual CNumericIO&  operator=(const CNumericIO& io)
        {
            CNumericIOBase::operator=(io);
            m_values = io.m_values;
            m_valueLabels = io.m_valueLabels;
            m_headerLabels = io.m_headerLabels;
            return *this;
        }
        /**
         * @brief Universal reference assignement operator
         */
        virtual CNumericIO&  operator=(const CNumericIO&& io)
        {
            CNumericIOBase::operator=(io);
            m_values = std::move(io.m_values);
            m_valueLabels = std::move(io.m_valueLabels);
            m_headerLabels = std::move(io.m_headerLabels);
            return *this;
        }

        std::string                 repr() const override
        {
            std::stringstream s;
            s << "CNumericIO(" << m_name << ")";
            return s.str();
        }

        /**
         * @brief Checks if some numeric values or labels are available.
         * @return True if numeric value list or label list are not empty, False otherwise.
         */
        bool                        isDataAvailable() const override
        {
            return (m_values.size() > 0 || m_valueLabels.size() > 0);
        }

        /**
         * @brief Clears numeric value list and label list.
         */
        virtual void                clearData() override
        {
            m_values.clear();
            m_valueLabels.clear();
            m_headerLabels.clear();
        }

        /**
         * @brief Appends value list of type Type.
         * @param values: generic values vector.
         */
        inline void                 addValueList(const std::vector<Type>& values)
        {
            m_values.push_back(values);
        }
        /**
         * @brief Appends value list of type Type.
         * @param values: generic values vector.
         * @param header: associated header label.
         */
        inline void                 addValueList(const std::vector<Type>& values, const std::string& headerLabel)
        {
            m_values.push_back(values);
            m_headerLabels.push_back(headerLabel);
        }
        /**
         * @brief Appends value list of type Type.
         * @param values: generic values vector.
         * @param labels: associated label for each value.
         */
        inline void                 addValueList(const std::vector<Type>& values, const std::vector<std::string>& labels)
        {
            if(values.size() != labels.size())
                throw CException(CoreExCode::INVALID_SIZE, "Value and label list must have the same size", __func__, __FILE__, __LINE__);

            m_values.push_back(values);
            m_valueLabels.push_back(labels);
        }
        /**
         * @brief Appends value list of type Type.
         * @param values: generic values vector.
         * @param header: associated header label.
         * @param labels: associated label for each value.
         */
        inline void                 addValueList(const std::vector<Type>& values, const std::string& headerLabel, const std::vector<std::string>& labels)
        {
            if(values.size() != labels.size())
                throw CException(CoreExCode::INVALID_SIZE, "Value and label list must have the same size", __func__, __FILE__, __LINE__);

            m_values.push_back(values);
            m_headerLabels.push_back(headerLabel);
            m_valueLabels.push_back(labels);
        }

        /**
         * @brief Gets value list at index i.
         * @param i: index of the desired value list.
         * @return Vector of values (generic type).
         */
        inline NumericValues        getValueList(size_t index) const
        {
            if(index < m_values.size())
                return m_values[index];
            else
                return std::vector<Type>();
        }
        /**
         * @brief Get all value lists.
         * @return Vector of vector of values (generic type).
         */
        inline VectorOfNumericValues getAllValues() const
        {
            return m_values;
        }
        inline VectorOfStringVector getAllValuesAsString() const override
        {
            VectorOfStringVector strValues;
            for(size_t i=0; i<m_values.size(); ++i)
            {
                StringVector valueList;
                for(size_t j=0; j<m_values[i].size(); ++j)
                    valueList.push_back(Utils::to_string(m_values[i][j]));

                strValues.push_back(valueList);
            }
            return strValues;
        }

        /**
         * @brief Performs a deep copy of the object.
         * @return CNumericIO smart pointer (std::shared_ptr).
         */
        NumericIOPtr                clone()
        {
            return std::static_pointer_cast<CNumericIO>(cloneImp());
        }

        void                        save(const std::string& path) override
        {
            CWorkflowTaskIO::save(path);

            auto extension = Utils::File::extension(path);
            if (extension == ".csv")
                saveCSV(path);
            else if (extension == ".json")
                saveJSON(path);
            else
                throw CException(CoreExCode::NOT_IMPLEMENTED, "Export format not available yet", __func__, __FILE__, __LINE__);
        }

        void                        load(const std::string& path) override
        {
            auto extension = Utils::File::extension(path);
            if(extension != ".csv")
                throw CException(CoreExCode::NOT_IMPLEMENTED, "Load format not available yet", __func__, __FILE__, __LINE__);

            loadCSV(path);
        }

        std::string                 toJson() const override
        {
            std::vector<std::string> options = {"json_format", "compact"};
            return toJson(options);
        }
        std::string                 toJson(const std::vector<std::string> &options) const override
        {
            QJsonObject root = toJsonInternal();
            QJsonDocument doc(root);
            return toFormattedJson(doc, options);
        }

        void                        fromJson(const std::string &jsonStr) override
        {
            Q_UNUSED(jsonStr);
        }

    private:

        virtual WorkflowTaskIOPtr   cloneImp() const override
        {
            return std::shared_ptr<CNumericIO>(new CNumericIO(*this));
        }

        void                        toJsonCommon(QJsonObject& root) const
        {
            root["outputType"] = static_cast<int>(m_outputType);
            root["plotType"] = static_cast<int>(m_plotType);

            QJsonArray colLabels;
            for (size_t i=0; i<m_headerLabels.size(); ++i)
                colLabels.append(QString::fromStdString(m_headerLabels[i]));

            root["headers"] = colLabels;

            QJsonArray valueLabels;
            for (size_t i=0; i<m_valueLabels.size(); ++i)
            {
                QJsonArray colValueLabels;
                for (size_t j=0; j<m_valueLabels[i].size(); ++j)
                    colValueLabels.append(QString::fromStdString(m_valueLabels[i][j]));

                valueLabels.append(colValueLabels);
            }
            root["valueLabels"] = valueLabels;
        }
        QJsonObject                 toJsonInternal() const
        {
            QJsonObject root;
            toJsonCommon(root);

            QJsonArray values;
            for (size_t i=0; i<m_values.size(); ++i)
            {
                QJsonArray colValues;
                for (size_t j=0; j<m_values[i].size(); ++j)
                    colValues.append(QString::fromStdString(Utils::to_string(m_values[i][j])));

                values.append(colValues);
            }
            root["values"] = values;
            return root;
        }

        void                        fromJsonCommon(const QJsonObject& root)
        {
            m_outputType = static_cast<NumericOutputType>(root["outputType"].toInt());
            m_plotType = static_cast<PlotType>(root["plotType"].toInt());

            m_headerLabels.clear();
            QJsonArray colLabels = root["headers"].toArray();

            for (int i=0; i<colLabels.size(); ++i)
                m_headerLabels.push_back(colLabels[i].toString().toStdString());

            m_valueLabels.clear();
            QJsonArray valueLabels = root["valueLabels"].toArray();

            for (int i=0; i<valueLabels.size(); ++i)
            {
                std::vector<std::string> labels;
                QJsonArray colValueLabels = valueLabels[i].toArray();

                for (int j=0; j<colValueLabels.size(); ++j)
                    labels.push_back(colValueLabels[i].toString().toStdString());

                m_valueLabels.push_back(labels);
            }
        }

        void                        saveCSV(const std::string &path) const
        {
            std::ofstream file;
            file.open(path, std::ios::out | std::ios::trunc);

            //Output type
            file << "Output type" << ";" << std::to_string(static_cast<int>(m_outputType)) << std::endl;

            //Plot type
            file << "Plot type" << ";" << std::to_string(static_cast<int>(m_plotType)) << std::endl;

            //Write header labels
            if(m_headerLabels.size() > 0)
            {
                if(m_valueLabels.size() > 0)
                    file << ";";

                for(size_t i=0; i<m_headerLabels.size(); ++i)
                    file << m_headerLabels[i] + ";";
                file << "\n";
            }
            else
                file << "No headers" << std::endl;

            //Count rows
            size_t nbRow = 0;
            for(size_t i=0; i<m_values.size(); ++i)
                nbRow = std::max(nbRow, m_values[i].size());

            //Write values and associated labels (if there are some)
            if(m_values.size() == m_valueLabels.size())
            {
                for(size_t i=0; i<nbRow; ++i)
                {
                    for(size_t j=0; j<m_values.size(); ++j)
                    {
                        file << m_valueLabels[j][i] + ";";
                        file << Utils::to_string(m_values[j][i]) + ";";
                    }
                    file << "\n";
                }
            }
            else if(m_valueLabels.size() == 1)
            {
                for(size_t i=0; i<nbRow; ++i)
                {
                    file << m_valueLabels[0][i] + ";";
                    for(size_t j=0; j<m_values.size(); ++j)
                        file << Utils::to_string(m_values[j][i]) + ";";

                    file << "\n";
                }
            }
            else
            {
                for(size_t i=0; i<nbRow; ++i)
                {
                    for(size_t j=0; j<m_values.size(); ++j)
                        file << Utils::to_string(m_values[j][i]) + ";";

                    file << "\n";
                }
            }
            file.close();
        }
        void                        saveJSON(const std::string& path) const
        {
            QFile jsonFile(QString::fromStdString(path));
            if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
                throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

            QJsonDocument doc(toJsonInternal());
            jsonFile.write(doc.toJson(QJsonDocument::Compact));
        }

        void                        loadCSV(const std::string& path)
        {
            Q_UNUSED(path);
        }

    private:

        VectorOfNumericValues       m_values;
};

// Partial specializations
template <>
DATAPROCESSSHARED_EXPORT std::string CNumericIO<std::string>::repr() const;

template <>
DATAPROCESSSHARED_EXPORT CNumericIOBase::VectorOfStringVector CNumericIO<std::string>::getAllValuesAsString() const;

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<std::string>::saveCSV(const std::string &path) const;

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<std::string>::loadCSV(const std::string &path);

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<double>::loadCSV(const std::string &path);

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<int>::loadCSV(const std::string &path);

template <>
DATAPROCESSSHARED_EXPORT QJsonObject CNumericIO<std::string>::toJsonInternal() const;

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<std::string>::fromJson(const std::string &jsonStr);

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<double>::fromJson(const std::string &jsonStr);

template <>
DATAPROCESSSHARED_EXPORT void CNumericIO<int>::fromJson(const std::string &jsonStr);


class DATAPROCESSSHARED_EXPORT CNumericIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CNumericIOFactory()
        {
            m_name = "CNumericIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CNumericIO<double>>();
        }
};

#endif // CNUMERICIO_H
