/*
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

#ifndef CWORKFLOWTASKPARAMWRAP_H
#define CWORKFLOWTASKPARAMWRAP_H

#include "PyCoreGlobal.h"
#include "Workflow/CWorkflowTaskParam.h"

class CWorkflowTaskParamWrap : public CWorkflowTaskParam, public wrapper<CWorkflowTaskParam>
{
    public:

        CWorkflowTaskParamWrap();
        CWorkflowTaskParamWrap(const CWorkflowTaskParam& param);

        void        setParamMap(const UMapString &paramMap) override;
        void        default_setParamMap(const UMapString& paramMap);

        UMapString  getParamMap() const override;
        UMapString  default_getParamMap() const;

        uint        getHashValue() const override;
        uint        default_getHashValue() const;
};


// Reference: https://github.com/boostorg/python/blob/4fc3afa3ac1a1edb61a92fccd31d305ba38213f8/test/pickle3.cpp#L4
// Doc: https://www.boost.org/doc/libs/1_87_0/libs/python/doc/html/reference/topics/pickle_support.html
struct TaskParamPickleSuite: pickle_suite
{
    static tuple getinitargs(const CWorkflowTaskParamWrap& param)
    {
        Q_UNUSED(param)
        return boost::python::make_tuple();
    }

    static tuple getstate(object paramObj)
    {
        CWorkflowTaskParamWrap& param = extract<CWorkflowTaskParamWrap&>(paramObj)();
        return boost::python::make_tuple(paramObj.attr("__dict__"), param.m_cfg);
    }

    static void setstate(object paramObj, tuple state)
    {
        if (len(state) != 2)
        {
            PyErr_SetObject(PyExc_ValueError, ("expected 2-item tuple in call to __setstate__; got %s"% state).ptr());
            throw_error_already_set();
        }

        // restore the object's __dict__
        dict d = extract<dict>(paramObj.attr("__dict__"))();
        d.update(state[0]);

        // restore the internal state of the C++ object
        CWorkflowTaskParamWrap& param = extract<CWorkflowTaskParamWrap&>(paramObj)();
        param.m_cfg = extract<UMapString>(state[1]);
    }

    static bool getstate_manages_dict()
    {
        return true;
    }
};

#endif // CWORKFLOWTASKPARAMWRAP_H
