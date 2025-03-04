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

#ifndef UTILSTOOLS_HPP
#define UTILSTOOLS_HPP

#include <string>
#include <future>
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/predef.h>
#include <QDebug>
#include <QSqlDatabase>
#include <QSqlError>
#include <QSqlQuery>
#include <QPoint>
#include <QVariantList>
#include <QTextLayout>
#include <QTextDocument>
#include <QFontMetrics>
#include <QDir>
#include <QCoreApplication>
#include <QGuiApplication>
#include <QDesktopServices>
#include <QProcess>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "CException.h"
#include "UtilsDefine.hpp"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <QRegularExpression>
#include "CSemanticVersion.h"

//Avoid conflict with Qt slots keyword
#undef slots
#include <Python.h>
#define slots

#include "boost/python.hpp"
#include "PythonThread.hpp"
#include <fstream>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#endif

///@cond INTERNAL

namespace Ikomia
{
    namespace Utils
    {
        template<typename T>
        QVariantList toVariantList( const QList<T> &list )
        {
            QVariantList newList;
            for(const T &item : list )
                newList << item;

            return newList;
        }

        template<class Function, class... Args>
        std::future<typename std::result_of<Function(Args...)>::type> async(
            Function&& f,
            Args&&... args)
        {
            // Typedef du type de retour
            using R = typename std::result_of<Function(Args...)>::type;
            // Binding de la fonction f(args)
            auto bound_task = std::bind(std::forward<Function>(f), std::forward<Args>(args)...);
            // Création d'une tâche qui retourne un objet de type R à partir de notre fonction binding
            auto task = std::packaged_task<R()>{std::move(bound_task)};
            // Get du résultat (non blocking à la destruction)
            auto ret = task.get_future();
            // Threading de notre tâche
            auto t = std::thread{std::move(task)};
            // On détache le thread de celui qui le call
            t.detach();
            return ret;
        }

        namespace adl_helper
        {
            using std::to_string;

            template<class T>
            std::string as_string(T&& t)
            {
                return to_string(std::forward<T>(t));
            }
        }

        template<class T>
        std::string to_string(T&& t)
        {
            return adl_helper::as_string(std::forward<T>(t));
        }

        namespace Python
        {
            using namespace boost::python;

            inline std::string  getVersion(const std::string& shape="major.minor.patch")
            {
                CPyEnsureGIL gil;
                std::string version = "";
                auto pyVersion = import("sys").attr("version_info");
                tuple versionTuple = extract<tuple>(pyVersion);
                int major = extract<int>(versionTuple[0]);

                if (shape == "major")
                    version = std::to_string(major);
                else
                {
                    int minor = extract<int>(versionTuple[1]);
                    if(shape == "major.minor")
                        version = std::to_string(major) + "." + std::to_string(minor);
                    else
                    {
                        int patch = extract<int>(versionTuple[2]);
                        version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
                    }
                }
                return version;
            }
            inline std::string  getMinSupportedVersion()
            {
                return "3.7.0";
            }
            inline std::string  getMaxSupportedVersion()
            {
                return "3.10.0";
            }
            inline std::string  getExceptionType(PyObject *pType)
            {
                std::string ret;
                if(pType)
                {
                    handle<> hType(pType);
                    str typeStr(hType);
                    // Extract the string from the boost::python object
                    extract<std::string> eTypeStr(typeStr);

                    // If a valid string extraction is available, use it
                    //  otherwise use fallback
                    if(eTypeStr.check())
                        ret = eTypeStr();
                    else
                        ret = "Unknown exception type";
                }
                return ret;
            }
            inline std::string  getExceptionValue(PyObject *pValue)
            {
                std::string ret;
                if(pValue)
                {
                    handle<> hValue(pValue);
                    str valueStr(hValue);
                    extract<std::string> returned(valueStr);

                    if(returned.check())
                        ret =  ": " + returned();
                    else
                        ret = std::string(": Unparseable Python error: ");
                }
                return ret;
            }
            inline std::string  getExceptionTraceback(PyObject *pTraceback)
            {
                std::string ret;
                if(pTraceback)
                {
                    try
                    {
                        handle<> hTb(pTraceback);
                        // Load the traceback module and the format_tb function
                        object tb(import("traceback"));
                        object formatTb(tb.attr("format_tb"));
                        // Call format_tb to get a list of traceback strings
                        object tbList(formatTb(hTb));
                        // Join the traceback strings into a single string
                        object tbStr(str("\n").join(tbList));
                        // Extract the string, check the extraction, and fallback in necessary
                        extract<std::string> returned(tbStr);

                        if(returned.check())
                            ret = ": " + returned();
                        else
                            ret = std::string(": Unparseable Python traceback");
                    }
                    catch(error_already_set&)
                    {
                        ret = std::string(": Unparseable Python traceback");
                    }
                }
                return ret;
            }
            inline std::set<std::string> getImportedModules()
            {
                CPyEnsureGIL gil;
                object main_module = import("__main__");
                object main_namespace = main_module.attr("__dict__");

                str code
                (
                    "import types\n\n"
                    "def imported_modules():\n"
                    "   modules = list()\n"
                    "   for name, val in globals().items():\n"
                    "       if isinstance(val, types.ModuleType):\n"
                    "           modules.append(val.__name__)\n"
                    "   return modules"
                );
                exec(code, main_namespace, main_namespace);
                object imported_modules = main_namespace["imported_modules"];
                object modules = imported_modules();

                std::set<std::string> importedModules;
                for(int i=0; i<len(modules); ++i)
                    importedModules.insert(extract<std::string>(modules[i]));

                return importedModules;
            }
            inline std::string  getIkomiaApiLibFolder()
            {
                CPyEnsureGIL gil;
                try
                {
                    auto pyIkPath = import("ikomia").attr("__file__");
                    extract<std::string> ikPath(pyIkPath);

                    if(ikPath.check())
                    {
                        QFileInfo info(QString::fromStdString(ikPath));
                        return info.absolutePath().toStdString() + "/lib";
                    }
                }
                catch (const boost::python::error_already_set &)
                {
                    return "";
                }
                return "";
            }
            inline std::string  getIkomiaApiFolder()
            {
                CPyEnsureGIL gil;
                try
                {
                    auto pyIkPath = import("ikomia.core").attr("get_ikomia_root_folder")();
                    extract<std::string> ikPath(pyIkPath);

                    if(ikPath.check())
                        return ikPath();
                }
                catch (const boost::python::error_already_set &)
                {
                    return "";
                }
                return "";
            }

            inline bool         isModuleImported(const std::string& name)
            {
                CPyEnsureGIL gil;
                object main_module = import("__main__");
                object main_namespace = main_module.attr("__dict__");

                str code
                (
                    "import types\n\n"
                    "def imported_modules():\n"
                    "   modules = list()\n"
                    "   for name, val in globals().items():\n"
                    "       if isinstance(val, types.ModuleType):\n"
                    "           modules.append(val.__name__)\n"
                    "   return modules"
                );
                exec(code, main_namespace, main_namespace);
                object imported_modules = main_namespace["imported_modules"];
                object modules = imported_modules();

                for(int i=0; i<len(modules); ++i)
                {
                    std::string moduleName = extract<std::string>(modules[i]);
                    if(moduleName == name)
                        return true;
                }
                return false;
            }
            inline bool         isFolderModule(const std::string& folder)
            {
                boost::filesystem::directory_iterator iter(folder), end;
                for(; iter != end; ++iter)
                {
                    if(iter->path().filename() == "__init__.py")
                        return true;
                }
                return false;
            }

            inline void         addToPythonPath(const std::string &path)
            {
                CPyEnsureGIL gil;
                boost::python::object main_module = boost::python::import("__main__");
                boost::python::object main_namespace = main_module.attr("__dict__");
                boost::python::str currentDir(path);
                boost::python::object sys = boost::python::import("sys");
                boost::python::object pathObj = sys.attr("path");

                bool bExist = false;
                for(int i=0; i<len(pathObj) && bExist == false; ++i)
                {
                    if(pathObj[i] == currentDir)
                        bExist = true;
                }

                if(bExist == false)
                {
                    sys.attr("path").attr("insert")(0, currentDir);
                    main_namespace["sys"] = sys;
                }
            }

            inline object       reloadModule(const std::string& name)
            {
                CPyEnsureGIL gil;
                object main_module = boost::python::import("__main__");
                object main_namespace = main_module.attr("__dict__");
                QString script = QString("import importlib\nimportlib.reload(%1)").arg(QString::fromStdString(name));
                str pyScript(script.toStdString());
                exec(pyScript, main_namespace, main_namespace);
                return main_namespace[name];
            }

            inline void         unloadModule(const std::string& name, bool bStrictCompare)
            {
                CPyEnsureGIL gil;
                object main_module = boost::python::import("__main__");
                object main_namespace = main_module.attr("__dict__");

                QString script;

                if(bStrictCompare)
                {
                    script = QString(
                            "import sys\n"
                            "modules_to_delete = [m for m in sys.modules.keys() if '%1' == m]\n"
                            "for m in modules_to_delete: del(sys.modules[m])\n"
                            ).arg(QString::fromStdString(name));
                }
                else
                {
                    script = QString(
                            "import sys\n"
                            "modules_to_delete = [m for m in sys.modules.keys() if '%1' in m]\n"
                            "for m in modules_to_delete: del(sys.modules[m])\n"
                            ).arg(QString::fromStdString(name));

                }
                str pyScript(script.toStdString());
                exec(pyScript, main_namespace, main_namespace);
            }

            inline std::string  handlePythonException()
            {
                CPyEnsureGIL gil;
                PyObject *pType = nullptr, *pValue = nullptr, *pTraceback = nullptr;
                // Fetch the exception info from the Python C API
                PyErr_Fetch(&pType, &pValue, &pTraceback);
                PyErr_NormalizeException(&pType, &pValue, &pTraceback);

                // Fallback error
                std::string ret("Unfetchable Python error");

                // If the fetch got a type pointer, parse the type into the exception string
                ret = getExceptionType(pType);

                // Do the same for the exception value (the stringification of the exception)
                ret += getExceptionValue(pValue);

                // Parse lines from the traceback using the Python traceback module
                ret += getExceptionTraceback(pTraceback);
                return ret;
            }

            inline void         print(const std::string& msg, const QtMsgType type=QtMsgType::QtInfoMsg)
            {
                CPyEnsureGIL gil;
                auto strMsg = str(msg);
                object file;

                switch(type)
                {
                    case QtDebugMsg:
                    {
                        // Log file only
                        std::ofstream logfile;
                        logfile.open(getIkomiaApiFolder() + "/log.txt", std::ios_base::app);
                        logfile << "DEBUG:" << msg;
                        break;
                    }
                    case QtInfoMsg:
                    default:
                        try
                        {
                            file = import("sys").attr("stdout");
                            auto write = file.attr("write");
                            write(strMsg);
                        }
                        catch (const error_already_set &)
                        {
                            /* If print() is called from code that is executed as
                             * part of garbage collection during interpreter shutdown,
                             * importing 'sys' can fail. Give up rather than crashing the
                             * interpreter in this case. */
                            return;
                        }
                        break;

                    case QtWarningMsg:
                    case QtCriticalMsg:
                    case QtFatalMsg:
                        try
                        {
                            file = import("sys").attr("stderr");
                            auto write = file.attr("write");
                            write(strMsg);
                        }
                        catch (const error_already_set &)
                        {
                            /* If print() is called from code that is executed as
                             * part of garbage collection during interpreter shutdown,
                             * importing 'sys' can fail. Give up rather than crashing the
                             * interpreter in this case. */
                            return;
                        }
                        break;
                }
            }

            inline void         runScript(const std::string& script)
            {
                CPyEnsureGIL gil;
                str pyScript(script);
                object main_module = import("__main__");
                object main_namespace = main_module.attr("__dict__");
                exec(pyScript, main_namespace, main_namespace);
            }
        }

        namespace IkomiaApp
        {
            inline bool         isAppStarted()
            {
                auto windows = QGuiApplication::allWindows();
                return windows.size() > 0;
            }
            inline std::string  getIkomiaFolder()
            {
                if(isAppStarted() || !Py_IsInitialized())
                    return QDir::homePath().toStdString() + "/Ikomia";
                else
                    return Utils::Python::getIkomiaApiFolder();
            }
            inline QString      getQIkomiaFolder()
            {
                if(isAppStarted() || !Py_IsInitialized())
                    return QDir::homePath() + "/Ikomia";
                else
                    return QString::fromStdString(Utils::Python::getIkomiaApiFolder());
            }
            inline QString      getGmicFolder()
            {
    #if defined(Q_OS_WIN64)
                return QDir::homePath() + "/AppData/Roaming/gmic";
    #elif defined(Q_OS_LINUX)
                return QDir::homePath() + ".config/gmic";
    #elif defined(Q_OS_MACOS)
                return QDir::homePath() + "Library/Preferences/gmic";
    #endif
            }
            inline QString      getTranslationsFolder()
            {
    #if defined(Q_OS_WIN64)
                return "translations/";
    #elif defined(Q_OS_LINUX)
                return QCoreApplication::applicationDirPath() + "/../translations/";
    #elif defined(Q_OS_MACOS)
                return ""; // TODO
    #endif
            }
            inline std::string  getCurrentVersionNumber()
            {
                return "0.13.0";
            }
            inline std::string  getCurrentVersionName()
            {
                return "0.14.0";
            }
            inline std::string  getIkomiaLibFolder()
            {
                if(isAppStarted())
                {
                    auto appDirPath = QCoreApplication::applicationDirPath();
                    QDir appDir(appDirPath);
                    appDir.cdUp();
                    return appDir.absolutePath().toStdString() + "/lib";
                }
                else
                    return Utils::Python::getIkomiaApiLibFolder();
            }
        }

        namespace OS
        {
            inline OSType       getCurrent()
            {
                #if defined(Q_OS_LINUX)
                    return OSType::LINUX;
                #elif defined(Q_OS_MACOS)
                    return OSType::OSX;
                #elif defined(Q_OS_WIN64)
                    return OSType::WIN;
                #else
                    return OSType::NONE;
                #endif
            }
            inline void         openUrl(const std::string& url)
            {
                bool bGuiStarted = Utils::IkomiaApp::isAppStarted();
                if(bGuiStarted)
                    QDesktopServices::openUrl(QUrl(QString::fromStdString(url)));
                else
                {
                    QProcess proc;
                    QStringList args;

                    #if defined(Q_OS_LINUX)
                        QString cmd = "/bin/sh";
                        args << "xdg-open";
                    #elif defined(Q_OS_WIN64)
                        QString cmd = "cmd.exe";
                        args << "/c" << "start";
                    #elif defined(Q_OS_MACOS)
                        QString cmd = "/bin/sh";
                        args << "open";
                    #endif

                    args << QString::fromStdString(url);
                    proc.start(cmd, args);
                    proc.waitForFinished();
                }
            }
            inline std::string  getName(OSType type)
            {
                std::string osName = "ALL";
                switch(type)
                {
                    case OSType::LINUX:
                        osName = "LINUX";
                        break;
                    case OSType::WIN:
                        osName = "WINDOWS";
                        break;
                    case OSType::OSX:
                        osName = "MACOS";
                        break;
                    case OSType::ALL:
                        osName = "ALL";
                        break;
                }
                return osName;
            }
            inline CpuArch      getCpuArch()
            {
                #if BOOST_ARCH_X86
                    #if BOOST_ARCH_X86_64
                        return CpuArch::X86_64;
                    #else
                        return CpuArch::NOT_SUPPORTED;
                    #endif
                #elif BOOST_ARCH_ARM
                    #if BOOST_ARCH_ARM > BOOST_VERSION_NUMBER(8, 0, 0)
                        #if BOOST_ARCH_WORD_BITS == 64
                            return CpuArch::ARM_64;
                        #elif BOOST_ARCH_WORD_BITS == 32
                            return CpuArch::ARM_32;
                        #else
                            return CpuArch::NOT_SUPPORTED;
                        #endif
                    #else
                        return CpuArch::NOT_SUPPORTED;
                    #endif
                #else
                    return CpuArch::NOT_SUPPORTED;
                #endif
            }
            inline std::string  getCpuArchName(CpuArch arch)
            {
                std::string archName;
                switch(arch)
                {
                    case CpuArch::X86_64:
                        archName = "X86_64";
                        break;
                    case CpuArch::ARM_64:
                        archName = "ARM_64";
                        break;
                    case CpuArch::ARM_32:
                        archName = "ARM_32";
                        break;
                    case CpuArch::NOT_SUPPORTED:
                        archName = "";
                }
                return archName;
            }
            inline std::string  getCudaVersionName()
            {
                #if defined(CUDA10)
                    return "CUDA10";
                #elif defined(CUDA11)
                    return "CUDA11";
                #elif defined(CUDA12)
                    return "CUDA12";
                #else
                    return "";
                #endif
            }
        }

        namespace String
        {
            inline void tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ")
            {
                tokens.clear();
                // Skip delimiters at beginning.
                std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
                // Find first "non-delimiter".
                std::string::size_type pos = str.find_first_of(delimiters, lastPos);

                while(std::string::npos != pos || std::string::npos != lastPos)
                {
                    // Found a token, add it to the vector.
                    tokens.push_back(str.substr(lastPos, pos - lastPos));
                    // Skip delimiters.  Note the "not_of"
                    lastPos = str.find_first_not_of(delimiters, pos);
                    // Find next "non-delimiter"
                    pos = str.find_first_of(delimiters, lastPos);
                }
            }

            inline std::string dbFormat(const std::string& str)
            {
                std::string newStr = str;
                boost::replace_all(newStr, "'", "''");
                return newStr;
            }

            inline std::string httpFormat(const std::string& str)
            {
                std::string newStr = str;
                boost::replace_all(newStr, " ", "");
                return newStr;
            }

            inline bool replace(std::string& str, const std::string& from, const std::string& to)
            {
                size_t start_pos = str.find(from);
                if(start_pos == std::string::npos)
                    return false;
                str.replace(start_pos, from.length(), to);
                return true;
            }

            inline void findSubstrings(const std::string& word, std::set<std::string>& substrings)
            {
                size_t l = word.length();
                for(size_t start = 0; start < l; start++)
                {
                    for(size_t length = 1; length < l - start + 1; length++)
                        substrings.insert(word.substr(start, length));
                }
            }

            inline std::string longestCommonSubstring(const std::string& first, const std::string& second)
            {
                std::set<std::string> firstSubstrings, secondSubstrings;
                findSubstrings(first, firstSubstrings);
                findSubstrings(second, secondSubstrings);

                std::set<std::string> common;
                std::set_intersection(  firstSubstrings.begin(), firstSubstrings.end(),
                                        secondSubstrings.begin(), secondSubstrings.end(),
                                        std::inserter(common, common.begin()));

                std::vector<std::string> commonSubs(common.begin(), common.end());
                std::sort(commonSubs.begin(), commonSubs.end(), [](const std::string &s1, const std::string &s2)
                {
                    return s1.length( ) > s2.length( ) ;
                });

                return *(commonSubs.begin());
            }

            inline QString getElidedString(const QString& str, const QFont& font, int width, int lineCount)
            {
                QTextDocument txtDoc;
                txtDoc.setHtml(str);
                txtDoc.setDefaultFont(font);
                txtDoc.setTextWidth(width);
                txtDoc.adjustSize();
                QString formattedStr = txtDoc.toPlainText();

                QTextLayout txtLayout(formattedStr);
                txtLayout.setFont(font);
                int line = 0, widthUsed = 0;
                txtLayout.beginLayout();

                while(++line < lineCount)
                {
                    QTextLine txtLine = txtLayout.createLine();
                    if(!txtLine.isValid())
                        break;

                    txtLine.setLineWidth(width);
                    widthUsed += txtLine.naturalTextWidth();
                }
                txtLayout.endLayout();
                widthUsed += width;
                QFontMetrics fmt(font);
                return fmt.elidedText(formattedStr, Qt::ElideRight, widthUsed);
            }

            // Erase all Occurrences of given substring from main string.
            inline void eraseAllSubStr(std::string& mainStr, const std::string& toErase)
            {
                size_t pos = std::string::npos;

                // Search for the substring in string in a loop untill nothing is found
                while ((pos  = mainStr.find(toErase) )!= std::string::npos)
                {
                    // If found then erase it from string
                    mainStr.erase(pos, toErase.length());
                }
            }

            inline QString toCamelCase(const QString& str)
            {
                QStringList parts = str.split('_', Qt::SkipEmptyParts);
                for(int i=0; i<parts.size(); ++i)
                    parts[i].replace(0, 1, parts[i][0].toUpper());

                return parts.join("");
            }
            inline std::string toCamelCase(const std::string& str)
            {
                return toCamelCase(QString::fromStdString(str)).toStdString();
            }
            inline std::string toLower(const std::string& str)
            {
                std::string lower(str);
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
                return lower;
            }
            inline std::string makeNumberString(int number, int nbLeadingZero)
            {
                std::ostringstream os;
                os << std::setfill('0') << std::setw(nbLeadingZero) << number;
                return os.str();
            }
        }

        namespace Database
        {
            inline QSqlDatabase connect(const QString& path, const QString& connectionName)
            {
                auto db = QSqlDatabase::database(connectionName);
                if(!db.isValid())
                {
                    db = QSqlDatabase::addDatabase("QSQLITE", connectionName);
                    db.setDatabaseName(path);

                    if(!db.isValid())
                        throw CException(DatabaseExCode::INVALID_QUERY, db.lastError().text().toStdString(), __func__, __FILE__, __LINE__);

                    bool bOpen = db.open();
                    if(!bOpen)
                        throw CException(DatabaseExCode::INVALID_QUERY, db.lastError().text().toStdString(), __func__, __FILE__, __LINE__);

                    QSqlQuery q(db);
                    if(!q.exec("PRAGMA foreign_keys = ON;"))
                        throw CException(DatabaseExCode::INVALID_QUERY, q.lastError().text().toStdString(), __func__, __FILE__, __LINE__);
                }
                return db;
            }

            inline int getQuerySize(QSqlQuery q)
            {
                int initialPos = q.at();
                int size = 0;

                if(q.last())
                    size = q.at() + 1;
                else
                    size = 0;

                // Important to restore initial pos
                q.seek(initialPos);
                return size;
            }

            inline QString getFTSKeywords(const QString& text)
            {
                QString txtKey = text;
                //Ensure text conformity
                txtKey.replace("-", " ");
                txtKey.replace("*", "");
                txtKey.replace("/", "");
                txtKey.replace("+", " ");
                txtKey.replace("?", "");
                txtKey.replace("!", "");

                QRegExp rx("[, ]"); // match a comma or a space
                QStringList list = txtKey.split(rx, Qt::SkipEmptyParts);

                if (list.size() == 0)
                    txtKey.clear();
                else if (list.size() == 1)
                    txtKey = list.at(0)+"*";
                else if (list.size()>1)
                {
                    for(int i=0; i<list.size()-1; ++i)
                        txtKey = txtKey + list.at(i) + "* AND ";

                    txtKey = txtKey + list.back()+"*";
                }
                return txtKey;
            }

            inline QString getSqliteVersion(const QSqlDatabase& db)
            {
                QSqlQuery q(db);
                if(!q.exec("SELECT sqlite_version();"))
                    throw CException(DatabaseExCode::INVALID_QUERY, q.lastError().text().toStdString(), __func__, __FILE__, __LINE__);

                if(q.first())
                    return q.value(0).toString();
                else
                    return QString("");
            }
        }

        namespace Geometry
        {
            inline float sqrDistance(const QPointF &pt1, const QPointF &pt2)
            {
                return ((pt2.x() - pt1.x()) * (pt2.x() - pt1.x())) +
                        ((pt2.y() - pt1.y()) * (pt2.y() - pt1.y()));
            }
        }

        namespace Color
        {
            inline std::vector<std::string> getColorNameCSS()
            {
                std::vector<std::string> colorNames = {"AliceBlue","AntiqueWhite","Aqua","Aquamarine","Azure","Beige","Bisque","Black","BlanchedAlmond","Blue","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","DarkGrey","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen"};

                return colorNames;
            }
        }

        namespace Gpu
        {
            inline bool hasOpenCL_GPU()
            {
                using namespace std;
                if (!cv::ocl::haveOpenCL())
                {
                    qDebug() << "OpenCL is not available...";
                    return false;
                }
                cv::ocl::Context context;
                if (!context.create(cv::ocl::Device::TYPE_GPU))
                {
                    qDebug() << "Failed creating the context for GPU...";
                    return false;
                }

                qDebug() << context.ndevices() << " GPU devices are detected.";

                return true;
            }
            inline void showOpenCV_OpenCL()
            {
                using namespace std;
                if (!cv::ocl::haveOpenCL())
                {
                    cout << "OpenCL is not available..." << endl;
                    //return;
                }

                cv::ocl::Context context;
                if (!context.create(cv::ocl::Device::TYPE_GPU))
                {
                    cout << "Failed creating the context..." << endl;
                    //return;
                }

                cout << context.ndevices() << " GPU devices are detected." << endl; //This bit provides an overview of the OpenCL devices you have in your computer
                for (size_t i = 0; i < context.ndevices(); i++)
                {
                    cv::ocl::Device device = context.device(i);
                    cout << "name:              " << device.name() << endl;
                    cout << "available:         " << device.available() << endl;
                    cout << "imageSupport:      " << device.imageSupport() << endl;
                    cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
                    cout << endl;
                }
            }
            inline int  getCudaDeviceCount()
            {
                int cudaDeviceCount = 0;
                try
                {
                    cudaDeviceCount = cv::cuda::getCudaEnabledDeviceCount() > 0;
                }
                catch(std::exception& e)
                {
                    Q_UNUSED(e);
                    qDebug() << QObject::tr("No CUDA device found");
                }
                return cudaDeviceCount;
            }
            inline bool isCudaAvailable()
            {
                return getCudaDeviceCount() > 0;
            }
        }

        namespace File
        {            
            inline std::string extension(const std::string &fileName)
            {
                boost::filesystem::path file(fileName);
                std::string ext = file.extension().string();
                ext = Utils::String::toLower(ext);
                return ext;
            }

            template<typename... Args>
            std::string makePath(const std::string& base, Args...args)
            {
                std::vector<std::string> parts = {{args...}};
                boost::filesystem::path path = base;

                for (size_t i=0; i<parts.size(); ++i)
                    path /= parts[i];

                return path.string();
            }

            inline std::string getAvailablePath(const std::string& originalPath)
            {
                std::string path = originalPath;
                boost::filesystem::path boostPath(originalPath);
                std::string ext = boostPath.extension().string();
                std::string folder = boostPath.parent_path().string();
                std::string fileName = boostPath.stem().string();

                int i = 1;
                while(boost::filesystem::exists(path) == true)
                {
                    path = makePath(folder, fileName + std::to_string(i) + ext);
                    ++i;
                }
                return path;
            }

            inline std::string getFileNameWithoutExtension(const std::string& filePath)
            {
                boost::filesystem::path path(filePath);
                while(!path.extension().empty())
                    path = path.stem();

                return path.stem().string();
            }

            inline std::string getFileName(const std::string& path)
            {
                boost::filesystem::path boostPath(path);
                return boostPath.filename().string();
            }

            inline std::string getParentPath(const std::string& filePath)
            {
                boost::filesystem::path path(filePath);
                return path.parent_path().string();
            }

            inline std::string getPathFromPattern(const std::string& pathPattern, int index)
            {
                // We can't use std::regex for now -> switch to QRegularExpression
                // since PyTorch don't fix C++ 11 ABI compatibility with std::regex
                // https://github.com/pytorch/pytorch/issues/50779

                std::string path = pathPattern;
                auto parent = getParentPath(pathPattern);
                auto filePattern = getFileName(pathPattern);
                QRegularExpression re("(.+)%[0-9]*([0-9]+)d(\\.[0-9a-z]+)");
                QRegularExpressionMatch match = re.match(QString::fromStdString(filePattern));

                if(match.hasMatch())
                {
                    auto name = match.captured(1).toStdString();
                    auto digits = match.captured(2).toInt();
                    auto extension = match.captured(3).toStdString();
                    boost::format fmt = boost::format("%1%%2%%3%") % name % boost::io::group(std::setw(digits), std::setfill('0'), index) % extension;
                    path = makePath(parent, fmt.str());
                }

                /*const std::regex regex(R"((.+)%[0-9]*([0-9]+)d(\.[0-9a-z]+))");
                std::smatch match;

                if(std::regex_search(filePattern, match, regex))
                {
                    size_t matchCount = match.size();
                    if(matchCount >= 4)
                    {
                        auto name = match.str(1);
                        auto digits = std::stoi(match.str(2));
                        auto extension = match.str(3);
                        boost::format fmt = boost::format("%1%%2%%3%") % name % boost::io::group(std::setw(digits), std::setfill('0'), index) % extension;
                        path = parent + "/" + fmt.str();
                    }
                }*/
                return path;
            }

            inline bool isFileExist(const std::string& path)
            {
                boost::filesystem::path file(path);
                return boost::filesystem::exists(file);
            }

            inline bool isFileSequenceExist(const std::string& pathPattern)
            {
                // We can't use std::regex for now -> switch to QRegularExpression
                // since PyTorch don't fix C++ 11 ABI compatibility with std::regex
                // https://github.com/pytorch/pytorch/issues/50779

                // Search for file with pattern like {prefix}%(0)nd.{ext}'
                boost::filesystem::path path(pathPattern);
                boost::filesystem::path directory(path.parent_path().string());
                auto filePattern = getFileName(pathPattern);

                // Check pattern validity
                QRegularExpression regexPattern("(.+)%[0-9]*([0-9]+)d(\\.[0-9a-z]+)");
                QRegularExpressionMatch match = regexPattern.match(QString::fromStdString(filePattern));

                if(!match.hasMatch())
                    return false;

                // Extract pattern information
                auto prefix = match.captured(1);
                auto digits = match.captured(2);
                auto extension = match.captured(3);

                // Regex to check file numbering validity
                const QString strRegexFile = prefix + ".*[0-9]{" + digits + "}" + extension;
                QRegularExpression regexFile(strRegexFile);

                for(const auto& entry : boost::filesystem::directory_iterator(directory))
                {
                    const auto filename = entry.path().filename().string();
                    QRegularExpressionMatch matchFile = regexFile.match(QString::fromStdString(filename));

                    if(matchFile.hasMatch())
                        return true;
                }

                // Check pattern validity
                /*std::smatch match;
                const std::regex regexPattern(R"((.+)%[0-9]*([0-9]+)d(\.[0-9a-z]+))");

                if(!std::regex_search(filePattern, match, regexPattern))
                    return false;

                size_t matchCount = match.size();
                if(matchCount < 4)
                    return false;

                // Extract pattern information
                auto prefix = match.str(1);
                auto digits = match.str(2);
                auto extension = match.str(3);

                // Regex to check file numbering validity
                const std::string strRegexFile = prefix + ".*[0-9]{" + digits + "}" + extension;
                const std::regex regexFile(strRegexFile);

                for(const auto& entry : boost::filesystem::directory_iterator(directory))
                {
                    const auto filename = entry.path().filename().string();
                    if(std::regex_search(filename, match, regexFile))
                        return true;
                }*/

                return false;
            }

            inline void createDirectory(const std::string path)
            {
                try
                {
                    boost::filesystem::path folderPath(path);
                    if (!path.empty() && !boost::filesystem::exists(folderPath))
                    {
                        if(!boost::filesystem::create_directories(folderPath))
                            throw CException(CoreExCode::INVALID_FILE, "Could not create directory: " + path);
                    }
                }
                catch (const boost::filesystem::filesystem_error& e)
                {
                    throw CException(CoreExCode::INVALID_FILE, e.code().message());
                }
            }

            inline bool copyDirectory(const QString& fromDirectory, const QString& toDirectory, bool bReplaceOnConflict)
            {
                QDir fromDir(fromDirectory);
                if(fromDir.exists() == false)
                    return false;

                QDir toDir(toDirectory);
                if(toDir.exists() == false)
                {
                    if(toDir.mkpath(toDirectory) == false)
                        return false;
                }

                foreach (QString copyFile, fromDir.entryList(QDir::Files))
                {
                    QCoreApplication::processEvents();
                    QString from = fromDirectory + "/" + copyFile;
                    QString to = toDirectory + "/" + copyFile;

                    if(QFile::exists(to))
                    {
                        if(bReplaceOnConflict)
                        {
                            if(QFile::remove(to) == false)
                                return false;
                        }
                        else
                            continue;
                    }

                    if (QFile::copy(from, to) == false)
                        return false;
                }

                foreach (QString copyDir, fromDir.entryList(QDir::Dirs | QDir::NoDotAndDotDot))
                {
                    QCoreApplication::processEvents();
                    QString from = fromDirectory + "/" + copyDir;
                    QString to = toDirectory + "/" + copyDir;

                    if(toDir.mkpath(to) == false)
                        return false;

                    if(copyDirectory(from, to, bReplaceOnConflict) == false)
                        return false;
                }

                return true;
            }

            inline QString conformName(const QString& fileName)
            {
                QString goodStr = fileName;
                //Remove spaces
                goodStr.replace(' ', "");
                //Remove ':' (forbidden on Windows)
                goodStr.replace(':', "-");
                return goodStr;
            }

            inline void showLocation(const QString& path)
            {
                QFileInfo info(path);

                #if defined(Q_OS_WIN)
                    QStringList args;
                    if (!info.isDir())
                        args << "/select,";
                    args << QDir::toNativeSeparators(path);
                    if (QProcess::startDetached("explorer", args))
                        return;
                #elif defined(Q_OS_MAC)
                    QStringList args;
                    args << "-e";
                    args << "tell application \"Finder\"";
                    args << "-e";
                    args << "activate";
                    args << "-e";
                    args << "select POSIX file \"" + path + "\"";
                    args << "-e";
                    args << "end tell";
                    args << "-e";
                    args << "return";
                    if (!QProcess::execute("/usr/bin/osascript", args))
                        return;
                #endif

                QDesktopServices::openUrl(QUrl::fromLocalFile(info.isDir() ? path : info.path()));
            }

            inline void moveFile(const std::string& pathFrom, const std::string& pathTo)
            {
                try
                {
                    boost::filesystem::rename(pathFrom, pathTo);
                }
                catch(const boost::filesystem::filesystem_error& e)
                {
                    Q_UNUSED(e)
                    try
                    {
                        boost::filesystem::copy_file(pathFrom, pathTo, boost::filesystem::copy_options::overwrite_existing);
                        boost::filesystem::remove(pathFrom);
                    }
                    catch(const boost::filesystem::filesystem_error& e)
                    {
                        throw CException(CoreExCode::INVALID_FILE, e.code().message(), __func__, __FILE__, __LINE__);
                    }
                }
            }

            inline bool isDirContainsFile(const std::string& directory, const std::string& filePath)
            {
                boost::filesystem::path dir(directory);
                boost::filesystem::path file(filePath);

                // If dir ends with "/" and isn't the root directory, then the final
                // component returned by iterators will include "." and will interfere
                // with the std::equal check below, so we strip it before proceeding.
                if (dir.filename() == ".")
                    dir.remove_filename();

                // We're also not interested in the file's name.
                assert(file.has_filename());
                file.remove_filename();

                // If dir has more components than file, then file can't possibly
                // reside in dir.
                auto dir_len = std::distance(dir.begin(), dir.end());
                auto file_len = std::distance(file.begin(), file.end());

                if (dir_len > file_len)
                    return false;

                // This stops checking when it reaches dir.end(), so it's OK if file
                // has more directory components afterward. They won't be checked.
                return std::equal(dir.begin(), dir.end(), file.begin());
            }
        }

        namespace Plugin
        {
            inline std::string  getPythonPath()
            {
                return IkomiaApp::getIkomiaFolder() + "/Plugins/Python";
            }
            inline std::string  getCppPath()
            {
                return IkomiaApp::getIkomiaFolder() + "/Plugins/C++";
            }
            inline PluginState  getCppApiState(const std::string& minVersion, const std::string& maxVersion)
            {
                const std::set<std::string> breakChanges = {"0.3.0", "0.4.0", "0.4.1", "0.5.0", "0.6.0", "0.6.1", "0.7.0", "0.8.0", "0.8.1", "0.9.0", "0.9.1", "0.10.0", "0.11.0", "0.11.1", "0.13.0", "0.14.0"};
                CSemanticVersion algoMinVersion(minVersion);

                for(auto it=breakChanges.begin(); it!=breakChanges.end(); ++it)
                {
                    CSemanticVersion breakChangesVersion((*it));
                    if(algoMinVersion < breakChangesVersion)
                        return PluginState::DEPRECATED;
                }

                CSemanticVersion currentVersion(Utils::IkomiaApp::getCurrentVersionNumber());
                if(algoMinVersion > currentVersion)
                    return PluginState::INVALID;

                if (maxVersion.empty() == false)
                {
                    CSemanticVersion algoMaxVersion(maxVersion);
                    if (algoMaxVersion < currentVersion)
                        return PluginState::INVALID;
                }
                return PluginState::VALID;
            }
            inline PluginState  getPythonApiState(const std::string& minVersion, const std::string& maxVersion)
            {
                const std::set<std::string> breakChanges = {"0.3.0", "0.6.0", "0.8.0", "0.9.0"};
                CSemanticVersion algoMinVersion(minVersion);

                for(auto it=breakChanges.begin(); it!=breakChanges.end(); ++it)
                {
                    CSemanticVersion breakChangesVersion((*it));
                    if(algoMinVersion < breakChangesVersion)
                        return PluginState::DEPRECATED;
                }

                CSemanticVersion currentVersion(Utils::IkomiaApp::getCurrentVersionNumber());
                if(algoMinVersion > currentVersion)
                    return PluginState::INVALID;

                if (maxVersion.empty() == false)
                {
                    CSemanticVersion algoMaxVersion(maxVersion);
                    if (algoMaxVersion < currentVersion)
                        return PluginState::INVALID;
                }
                return PluginState::VALID;
            }
            inline PluginState  getApiCompatibilityState(const std::string& minVersion, const std::string& maxVersion, ApiLanguage language)
            {
                if(language == ApiLanguage::CPP)
                    return getCppApiState(minVersion, maxVersion);
                else if(language == ApiLanguage::PYTHON)
                    return getPythonApiState(minVersion, maxVersion);
                else
                    return PluginState::INVALID;
            }
            inline std::string  getCurrentApiVersion()
            {
                return "0.14.0";
            }
            inline std::string  getModelHubUrl()
            {
                return "https://s3.eu-west-3.amazonaws.com/models.ikomia.com";
            }
            inline std::string  getLanguageString(ApiLanguage language)
            {
                std::string name;
                switch(language)
                {
                    case ApiLanguage::CPP: name = "CPP"; break;
                    case ApiLanguage::PYTHON: name = "PYTHON"; break;
                }
                return name;
            }
            inline std::string  getLicenseString(License license)
            {
                std::string name;
                switch(license)
                {
                    case CUSTOM: name = "CUSTOM"; break;
                    case AGPL_30: name = "AGPL_30"; break;
                    case APACHE_20: name = "APACHE_20"; break;
                    case BSD_2_CLAUSE: name = "BSD_2_CLAUSE"; break;
                    case BSD_3_CLAUSE: name = "BSD_3_CLAUSE"; break;
                    case CC0_10: name = "CC0_10"; break;
                    case CC_BY_NC_40: name = "CC_BY_NC_40"; break;
                    case GPL_30: name = "GPL_30"; break;
                    case LGPL_30: name = "LGPL_30"; break;
                    case MIT: name = "MIT"; break;
                }
                return name;
            }
            inline License      getLicenseFromName(const std::string& name)
            {
                std::string licenseStr = Utils::String::toLower(name);
                if (licenseStr.find("mit") != std::string::npos)
                    return License::MIT;
                else if (licenseStr.find("agpl") != std::string::npos || licenseStr.find("affero") != std::string::npos)
                    return License::AGPL_30;
                else if (licenseStr.find("lgpl") != std::string::npos || licenseStr.find("lesser") != std::string::npos)
                    return License::LGPL_30;
                else if (licenseStr.find("gpl") != std::string::npos || licenseStr.find("gnu") != std::string::npos)
                    return License::GPL_30;
                else if (licenseStr.find("apache") != std::string::npos)
                    return License::APACHE_20;
                else if (licenseStr.find("bsd") != std::string::npos)
                {
                    if (licenseStr.find("2-clause") != std::string::npos)
                        return License::BSD_2_CLAUSE;
                    else
                        return License::BSD_3_CLAUSE;
                }
                else if (licenseStr.find("creative") != std::string::npos || licenseStr.find("cc") != std::string::npos)
                {
                    if (licenseStr.find("nc") != std::string::npos || licenseStr.find("non commercial") != std::string::npos)
                        return License::CC_BY_NC_40;
                    else
                        return License::CC0_10;
                }
                else
                    return License::CUSTOM;
            }
        }

        namespace MLflow
        {
            inline std::string getTrackingURI()
            {
                return "http://localhost:5000";
            }
            inline std::string getBackendStoreURI()
            {
                return Utils::IkomiaApp::getIkomiaFolder() + "/MLflow";
            }
            inline std::string getArtifactURI()
            {
                return Utils::IkomiaApp::getIkomiaFolder() + "/MLflow";
            }
        }

        namespace Tensorboard
        {
            inline std::string getTrackingURI()
            {
                return "http://localhost:6006";
            }
            inline std::string getLogDirUri()
            {
                return Utils::IkomiaApp::getIkomiaFolder() + "/Tensorboard";
            }
        }

        namespace Jupyter
        {
            inline std::string getServerUri()
            {
                return "http://localhost:8888";
            }

            inline std::string getNotebookDir()
            {
                return Utils::Plugin::getPythonPath();
            }
        }

        inline void print(const QString& msg, const QtMsgType type=QtMsgType::QtInfoMsg)
        {
            if(IkomiaApp::isAppStarted())
            {
                switch(type)
                {
                    case QtDebugMsg:
                        qDebug().noquote() << msg;
                        break;

                    case QtInfoMsg:
                        qInfo().noquote() << msg;
                        break;

                    case QtWarningMsg:
                        qWarning().noquote() << msg;
                        break;

                    case QtCriticalMsg:
                        qCritical().noquote() << msg;
                        break;

                    case QtFatalMsg:
                        qFatal(msg.toLatin1().constData());
                        break;

                    default:
                        qInfo().noquote() << msg;
                        break;
                }
            }
            else
            {
                if (Py_IsInitialized())
                    Python::print(msg.toStdString() + "\n", type);
                else
                {
                    switch(type)
                    {
                        case QtDebugMsg:
                        case QtInfoMsg:
                            std::cout << msg.toStdString() << std::endl;
                            break;

                        case QtWarningMsg:
                        case QtCriticalMsg:
                        case QtFatalMsg:
                            std::cerr << msg.toStdString() << std::endl;
                            break;

                        default:
                            std::cout << msg.toStdString() << std::endl;
                            break;
                    }
                }
            }
        }

        inline void print(const std::string& msg, const QtMsgType type=QtMsgType::QtInfoMsg)
        {
            if(IkomiaApp::isAppStarted())
            {
                switch(type)
                {
                    case QtDebugMsg:
                        qDebug().noquote() << QString::fromStdString(msg);
                        break;

                    case QtInfoMsg:
                        qInfo().noquote() << QString::fromStdString(msg);
                        break;

                    case QtWarningMsg:
                        qWarning().noquote() << QString::fromStdString(msg);
                        break;

                    case QtCriticalMsg:
                        qCritical().noquote() << QString::fromStdString(msg);
                        break;

                    case QtFatalMsg:
                        qFatal(msg.c_str());
                        break;

                    default:
                        qInfo().noquote() << QString::fromStdString(msg);
                        break;
                }
            }
            else
            {
                if (Py_IsInitialized())
                    Python::print(msg + "\n", type);
                else
                {
                    switch(type)
                    {
                        case QtDebugMsg:
                        case QtInfoMsg:
                            std::cout << msg << std::endl;
                            break;

                        case QtWarningMsg:
                        case QtCriticalMsg:
                        case QtFatalMsg:
                            std::cerr << msg << std::endl;
                            break;

                        default:
                            std::cout << msg << std::endl;
                            break;
                    }
                }
            }
        }

        inline void print(const char* msg, const QtMsgType type=QtMsgType::QtInfoMsg)
        {
            print(std::string(msg), type);
        }

        inline void deprecationWarning(const std::string& msg, const std::string& maxVersion="", const QtMsgType type=QtMsgType::QtWarningMsg)
        {
            std::string fullMsg = msg;
            if (!maxVersion.empty())
                fullMsg += ". Will be removed from " + maxVersion + " version.";
            else
                fullMsg += ". Will be removed in future version.";

            print(fullMsg, type);
        }

    #ifdef __linux__
        namespace linuxHelp
        {
            inline void findCamera(const std::string& path)
            {
                int fd;
                struct v4l2_capability vid_cap;

                if((fd = open(path.c_str(), O_RDONLY)) == -1){
                    perror("cam_info: Can't open device");
                    return;
                }

                if(ioctl(fd, VIDIOC_QUERYCAP, &vid_cap) == -1)
                    perror("cam_info: Can't get capabilities");

                int type;
                if(ioctl(fd, VIDIOC_STREAMON, &type) < 0){
                    perror("Could not start streaming, VIDIOC_STREAMON");
                    return;
                }

                close(fd);
            }
        }
    #endif
    }
}

using namespace Ikomia;

///@endcond

#endif // UTILSTOOLS_HPP
