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

#ifndef PYTHONTHREAD_HPP
#define PYTHONTHREAD_HPP

#include "UtilsGlobal.hpp"

//Avoid conflict with Qt slots keyword
#undef slots
#include <Python.h>
#define slots

///@cond INTERNAL

class CPyAllowThreads
{
    public:

        CPyAllowThreads() : m_state(PyEval_SaveThread()){}
        ~CPyAllowThreads(){ PyEval_RestoreThread(m_state); }

    private:

        PyThreadState* m_state;
};

class CPyEnsureGIL
{
    public:

        CPyEnsureGIL()
        {
            if (Py_IsInitialized())
            {
                m_state = PyGILState_Ensure();
                m_acquired = true;
            }
        }
        ~CPyEnsureGIL()
        {
            if (m_acquired)
                PyGILState_Release(m_state);
        }

    private:

        PyGILState_STATE    m_state;
        bool                m_acquired = false;
};

///@endcond

#endif // PYTHONTHREAD_HPP
