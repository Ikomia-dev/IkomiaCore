// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the IkomiaStudio software.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "CPluginTools.h"
#include "UtilsTools.hpp"
#include "PythonThread.hpp"

Utils::CPluginTools::CPluginTools()
{
}

std::string Utils::CPluginTools::getTransferPath()
{
    return IkomiaApp::getIkomiaFolder() + "/Plugins/Transfer";
}

std::string Utils::CPluginTools::getDirectory(const std::string& name, int language)
{
    std::string directory;
    if(language == ApiLanguage::CPP)
        directory = Utils::Plugin::getCppPath() + "/" + Utils::String::httpFormat(name);
    else
        directory = Utils::Plugin::getPythonPath() + "/" + Utils::String::httpFormat(name);

    return directory;
}

std::string Utils::CPluginTools::getDirectory(const std::string &name)
{
    std::string directory = Utils::Plugin::getPythonPath() + "/" + Utils::String::httpFormat(name);
    if (Utils::File::isFileExist(directory))
        return directory;

    directory = Utils::Plugin::getCppPath() + "/" + Utils::String::httpFormat(name);
    if (Utils::File::isFileExist(directory))
        return directory;

    return "";
}

std::string Utils::CPluginTools::getPythonPluginFolder(const std::string &name)
{
    auto pluginFolder = Utils::CPluginTools::getPythonValidPluginFolder(name);
    if (pluginFolder.empty())
    {
        pluginFolder = getDirectory(name, ApiLanguage::PYTHON);
        if(boost::filesystem::exists(pluginFolder) == false)
            return std::string();
    }
    return pluginFolder;
}

std::string Utils::CPluginTools::getReadmeDescription(const std::string &name)
{
    // List of file patterns used to search for plugin documentation file
    // readme.md is reserved for git-based repository information
    const QSet<QString> docFiles = {
        "doc.md", "doc.html", "doc.htm",
        "documentation.md", "documentation.html", "documentation.htm",
        "info.md", "info.html", "info.htm",
        "readme.md", "readme.html", "readme.htm",
    };

    std::string pluginDir = getDirectory(name);
    QString docFilePath;
    QDir qpluginDir(QString::fromStdString(pluginDir));

    // Check if local doc file exists
    foreach (QString fileName, qpluginDir.entryList(QDir::Files|QDir::NoSymLinks))
    {
        if(docFiles.contains(fileName.toLower()))
        {
            docFilePath = qpluginDir.absoluteFilePath(fileName);
            break;
        }
    }

    if(!docFilePath.isEmpty())
    {
        // Load doc file
        QFile file(docFilePath);
        if(file.open(QFile::ReadOnly | QFile::Text))
        {
            QString mdContent(file.readAll());
            return mdContent.toStdString();
        }
    }
    return std::string();
}

std::string Utils::CPluginTools::getPythonValidPluginFolder(const std::string &name)
{
    CPyEnsureGIL gil;
    QDir pluginsDir(QString::fromStdString(Utils::Plugin::getPythonPath()));

    foreach (QString directory, pluginsDir.entryList(QDir::Dirs|QDir::NoDotAndDotDot))
    {
        QString currentPluginDirPath = pluginsDir.absoluteFilePath(directory);
        QDir currentPluginDir(currentPluginDirPath);

        foreach (QString fileName, currentPluginDir.entryList(QDir::Files))
        {
            auto pluginDirName = currentPluginDir.dirName();
            if(fileName == pluginDirName + ".py")
            {
                try
                {
                    boost::python::object main_module = boost::python::import("__main__");
                    boost::python::object main_namespace = main_module.attr("__dict__");
                    //Module names
                    std::string pluginName = pluginDirName.toStdString();
                    std::string moduleName = pluginName + "." + pluginName;
                    //Load main modules of plugin
                    boost::python::object mainModule = main_namespace[moduleName];
                    //Instantiate plugin factory
                    boost::python::object pyFactory = mainModule.attr(boost::python::str("IkomiaPlugin"))();
                    boost::python::extract<CPluginProcessInterface*> exFactory(pyFactory);

                    if(exFactory.check())
                    {
                        auto plugin = exFactory();
                        auto taskFactoryPtr = plugin->getProcessFactory();

                        if(taskFactoryPtr->getInfo().m_name == name)
                            return currentPluginDirPath.toStdString();
                    }
                }
                catch(boost::python::error_already_set&)
                {
                    //Plugin not loaded and considered invalid
                    continue;
                }
                catch(std::exception& e)
                {
                    //Plugin not loaded and considered invalid
                    continue;
                }
            }
        }
    }
    return std::string();
}

std::string Utils::CPluginTools::getCppValidPluginFolder(const std::string &name)
{
    QDir pluginsDir(QString::fromStdString(Utils::Plugin::getCppPath()));

    foreach (QString directory, pluginsDir.entryList(QDir::Dirs|QDir::NoDotAndDotDot))
    {
        QString currentPluginDirPath = pluginsDir.absoluteFilePath(directory);
        QDir currentPluginDir(currentPluginDirPath);

        foreach (QString fileName, currentPluginDir.entryList(QDir::Files|QDir::NoSymLinks))
        {
            QString pluginFile = currentPluginDir.absoluteFilePath(fileName);
            if(QLibrary::isLibrary(pluginFile))
            {
                QPluginLoader pluginLoader(pluginFile);
                QObject* pObject = pluginLoader.instance();

                if(pObject)
                {
                    auto pPlugin = qobject_cast<CPluginProcessInterface*>(pObject);
                    if(pPlugin)
                    {
                        auto taskFactoryPtr = pPlugin->getProcessFactory();
                        if(taskFactoryPtr)
                        {
                            if(taskFactoryPtr->getInfo().m_name == name)
                                return currentPluginDirPath.toStdString();
                        }
                    }
                }
                pluginLoader.unload();
            }
        }
    }
    return std::string();
}

boost::python::object Utils::CPluginTools::loadPythonModule(const std::string &name, bool bReload)
{
    CPyEnsureGIL gil;
    boost::python::object main_module = boost::python::import("__main__");
    boost::python::object main_namespace = main_module.attr("__dict__");
    boost::python::str moduleName(name);

    if(Utils::Python::isModuleImported(name) && bReload)
        return Utils::Python::reloadModule(name);
    else
    {
        boost::python::object module = boost::python::import(moduleName);
        main_namespace[name] = module;
        return module;
    }
}
