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

#ifndef PYUTILSDOCSTRING_HPP
#define PYUTILSDOCSTRING_HPP


constexpr auto _moduleDocString = "Module offering various helper tools. It consists of Python bindings from C++ Ikomia Core.\n";

constexpr auto _getCurrentVersionDocString =
        "Get the current API version.\n\n"
        "Returns:"
        "   str: API version\n";

constexpr auto _pythonStateDocString =
        "Get the plugin compatibility state, see :py:class:`~ikomia.utils.pyutils.PluginState` for possible values.\n\n"
        "Args:\n"
        "   version (str): base version of the plugin\n"
        "   language (:py:class:`utils.pyutils.ApiLanguage`)\n\n"
        "Returns:\n"
        "   :py:class:`utils.pyutils.PluginState`: state\n";

constexpr auto _checkArchiKeywordsDocString =
        "Check whether the current API compilation architecture matches the given plugin architecture keywords. "
        "This function is only usefull for C++ plugins.\n\n"
        "Args:\n\n"
        "   keywords (str): plugin keywords\n\n"
        "Returns:\n\n"
        "   boolean: True if architectures match, False otherwise\n\n";

constexpr auto _getModelHubUrlDocString =
        "Return root URL of Ikomia model HUB.\n\n"
        "Returns:\n\n"
        "   str: Ikomia model hub URL\n\n";

//----------------------
//----- CException -----
//-----------------------
constexpr auto _exceptionDocString =
        "Base class for all Ikomia exceptions.\n";

constexpr auto _ctorExcDocString =
        "Constructor with parameters\n\n"
        "Args:\n\n"
        "   code (int): exception code\n\n"
        "   error (str): exception message\n\n"
        "   func (str): function from where exception is thrown\n\n"
        "   file (str): file from where exception is thrown\n\n"
        "   line (int): Line in source file from where exception is thrown\n";

constexpr auto _messageDocString =
        "Get formatted exception message.\n\n"
        "Returns:\n"
        "   str: Exception message\n";

//-----------------------
//----- CMemoryInfo -----
//-----------------------
constexpr auto _memoryInfoDocString =
        "Cross-platform functions to retrieve memory information.";

constexpr auto _availableMemoryDocString =
        "Get current amount of available memory.\n\n"
        "Returns:\n"
        "   int: available amount of memory in MB\n";

constexpr auto _memoryLoadDocString =
        "Get the current memory load.\n\n"
        "Returns:\n"
        "   int: memory load in percentage\n";

constexpr auto _totalMemoryDocString =
        "Get total amount of memory installed on the machine.\n\n"
        "Returns:\n"
        "   int: total amount of memory in MB\n";

//------------------
//----- CTimer -----
//------------------
constexpr auto _timerDocString =
        "Cross-platform timer with nanoseconds precision\n";

constexpr auto _getElapsedMsDocString =
        "Get elapsed time in ms since the last call or since the start (first call).\n\n"
        "Returns:\n"
        "   float: elapsed time in ms\n";

constexpr auto _getElapsedNsDocString =
        "Get elapsed time in ns since the last call or since the start (first call).\n\n"
        "Returns:\n"
        "   float: elapsed time in ns\n";

constexpr auto _getElapsedUsDocString =
        "Get elapsed time in µs since the last call or since the start (first call).\n\n"
        "Returns:\n"
        "   float: elapsed time in µs\n";

constexpr auto _getTotalElapsedMsDocString =
        "Get elapsed time in ms since the start.\n\n"
        "Returns:\n"
        "   float: elapsed time in ms\n";

constexpr auto _getTotalElapsedNsDocString =
        "Get elapsed time in ns since the start.\n\n"
        "Returns:\n"
        "   float: elapsed time in ns\n";

constexpr auto _getTotalElapsedUsDocString =
        "Get elapsed time in µs since the start.\n\n"
        "Returns:\n"
        "   float: elapsed time in µs\n";

constexpr auto _printElapsedTimeDocString =
        "Print elapsed time in ms since the last call or since the start (first call).\n"
        "String printed to the standard output stream.\n\n"
        "Args:\n"
        "   name (str): label identifying timed process\n";

constexpr auto _printTotalElapsedTimeDocString =
        "Print elapsed time in ms since the start.\n"
        "String printed to the standard output stream.\n\n"
        "Args:\n"
        "   name (str): label identifying timed process\n";

constexpr auto _startDocString = "Launch timer.\n";


#endif // PYUTILSDOCSTRING_HPP
