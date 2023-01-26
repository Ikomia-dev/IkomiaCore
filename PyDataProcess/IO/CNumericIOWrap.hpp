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

#ifndef CNUMERICIOWRAP_H
#define CNUMERICIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CNumericIO.h"

template<class Type>
class CNumericIOWrap : public CNumericIO<Type>, public wrapper<CNumericIO<Type>>
{
    public:

        CNumericIOWrap() : CNumericIO<Type>()
        {
        }
        CNumericIOWrap(const CNumericIO<Type>& io) : CNumericIO<Type>(io)
        {
        }

        virtual size_t  getUnitElementCount() const override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override getOver = this->get_override("get_unit_element_count"))
                    return getOver();
                return CNumericIO<Type>::getUnitElementCount();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        size_t          default_getUnitElementCount() const
        {
            CPyEnsureGIL gil;
            try
            {
                return this->CNumericIO<Type>::getUnitElementCount();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        virtual bool    isDataAvailable() const override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override isDataOver = this->get_override("is_data_available"))
                    return isDataOver();
                return CNumericIO<Type>::isDataAvailable();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        bool            default_isDataAvailable() const
        {
            CPyEnsureGIL gil;
            try
            {
                return this->CNumericIO<Type>::isDataAvailable();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        virtual void    clearData() override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override clearDataOver = this->get_override("clear_data"))
                    clearDataOver();
                else
                    CNumericIO<Type>::clearData();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        void            default_clearData()
        {
            CPyEnsureGIL gil;
            try
            {
                this->CNumericIO<Type>::clearData();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        virtual void    copyStaticData(const std::shared_ptr<CWorkflowTaskIO>& ioPtr) override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override copyOver = this->get_override("copy_static_data"))
                    copyOver(ioPtr);
                else
                    CNumericIO<Type>::copyStaticData(ioPtr);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        void            default_copyStaticData(const std::shared_ptr<CWorkflowTaskIO>& ioPtr)
        {
            CPyEnsureGIL gil;
            try
            {
                this->CNumericIO<Type>::copyStaticData(ioPtr);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        void            load(const std::string &path) override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override loadOver = this->get_override("load"))
                    loadOver(path);
                else
                    CNumericIO<Type>::load(path);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        void            default_load(const std::string &path)
        {
            CPyEnsureGIL gil;
            try
            {
                this->CNumericIO<Type>::load(path);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        void            save(const std::string &path) override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override saveOver = this->get_override("save"))
                    saveOver(path);
                else
                    CNumericIO<Type>::save(path);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        void            default_save(const std::string &path)
        {
            CPyEnsureGIL gil;
            try
            {
                this->CNumericIO<Type>::save(path);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        std::string     toJson() const override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override toJsonOver = this->get_override("to_json"))
                    return toJsonOver();
                else
                    return CNumericIO<Type>::toJson();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        std::string     default_toJsonNoOpt() const
        {
            CPyEnsureGIL gil;
            try
            {
                return this->CNumericIO<Type>::toJson();
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        std::string     toJson(const std::vector<std::string>& options) const override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override toJsonOver = this->get_override("to_json"))
                    return toJsonOver(options);
                else
                    return CNumericIO<Type>::toJson(options);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        std::string     default_toJson(const std::vector<std::string>& options) const
        {
            CPyEnsureGIL gil;
            try
            {
                return this->CNumericIO<Type>::toJson(options);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }

        void            fromJson(const std::string &jsonStr) override
        {
            CPyEnsureGIL gil;
            try
            {
                if(override fromJsonOver = this->get_override("from_json"))
                    fromJsonOver(jsonStr);
                else
                    CNumericIO<Type>::fromJson(jsonStr);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
        void            default_fromJson(const std::string &jsonStr)
        {
            CPyEnsureGIL gil;
            try
            {
                this->CNumericIO<Type>::fromJson(jsonStr);
            }
            catch(boost::python::error_already_set&)
            {
                throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
            }
        }
};

#endif // CNUMERICIOWRAP_H
