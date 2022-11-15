#include "UtilsTools.hpp"
#ifdef Q_OS_WIN64
#include <windows.h>
#endif
#include "CIkomiaRegistry.h"
#include "CPluginProcessInterface.hpp"


CIkomiaRegistry::CIkomiaRegistry()
{
    m_pluginsDir = Utils::IkomiaApp::getIkomiaFolder() + "/Plugins";
    loadCppPlugins();
}

CIkomiaRegistry::~CIkomiaRegistry()
{
    for(auto&& it: m_loaders)
        delete it;
}

void CIkomiaRegistry::setPluginsDirectory(const std::string &dir)
{
    m_pluginsDir = dir;
}

std::vector<std::string> CIkomiaRegistry::getAlgorithms() const
{
    std::vector<std::string> names;
    auto factory = m_processRegistrator.getProcessFactory();
    auto taskFactories = factory.getList();

    for(auto&& factoryPtr : taskFactories)
        names.push_back(factoryPtr->getInfo().getName());

    return names;
}

std::string CIkomiaRegistry::getPluginsDirectory() const
{
    return m_pluginsDir;
}

std::string CIkomiaRegistry::getPluginDirectory(const std::string &name) const
{
    auto info = getAlgorithmInfo(name);
    if (info.m_language == ApiLanguage::PYTHON)
        return m_pluginsDir + "/Python/" + name;
    else
        return m_pluginsDir + "/C++/" + name;
}

CTaskInfo CIkomiaRegistry::getAlgorithmInfo(const std::string &name) const
{
    return m_processRegistrator.getProcessInfo(name);
}

CProcessRegistration *CIkomiaRegistry::getTaskRegistrator()
{
    return &m_processRegistrator;
}

CTaskIORegistration *CIkomiaRegistry::getIORegistrator()
{
    return &m_ioRegistrator;
}

WorkflowTaskPtr CIkomiaRegistry::createInstance(const std::string &processName)
{
    return m_processRegistrator.createProcessObject(processName, nullptr);
}

WorkflowTaskPtr CIkomiaRegistry::createInstance(const std::string &processName, const WorkflowTaskParamPtr &paramPtr)
{
    return m_processRegistrator.createProcessObject(processName, paramPtr);
}

WorkflowTaskWidgetPtr CIkomiaRegistry::createWidgetInstance(const std::string &processName, const WorkflowTaskParamPtr &paramPtr)
{
    return m_processRegistrator.createWidgetObject(processName, paramPtr);
}

void CIkomiaRegistry::registerTask(const TaskFactoryPtr &factoryPtr)
{
    m_processRegistrator.registerProcess(factoryPtr, nullptr);
}

void CIkomiaRegistry::registerTaskAndWidget(const TaskFactoryPtr &factoryPtr, WidgetFactoryPtr &widgetFactoryPtr)
{
    m_processRegistrator.registerProcess(factoryPtr, widgetFactoryPtr);
}

void CIkomiaRegistry::registerIO(const TaskIOFactoryPtr &factoryPtr)
{
    m_ioRegistrator.registerIO(factoryPtr);
}

void CIkomiaRegistry::loadCppPlugins()
{
    QString pluginRootPath = QString::fromStdString(m_pluginsDir + "/C++");
    QDir pluginsDir(pluginRootPath);

#ifdef Q_OS_WIN64
    //Add plugin root folder to the search path of the DLL loader
    SetDllDirectoryA(QDir::toNativeSeparators(pluginRootPath).toStdString().c_str());
#endif

    //Load plugins placed directly in the root folder
    foreach (QString fileName, pluginsDir.entryList(QDir::Files|QDir::NoSymLinks))
        _loadCppPlugin(pluginsDir.absoluteFilePath(fileName));

    //Scan sub-directories
    foreach (QString directory, pluginsDir.entryList(QDir::Dirs|QDir::NoDotAndDotDot))
    {
        auto dirPath = pluginsDir.absoluteFilePath(directory);
        QDir pluginDir(dirPath);

#ifdef Q_OS_WIN64
        //Add current plugin folder to the search path of the DLL loader
        SetDllDirectoryA(QDir::toNativeSeparators(dirPath).toStdString().c_str());
#endif

        foreach (QString fileName, pluginDir.entryList(QDir::Files|QDir::NoSymLinks))
            _loadCppPlugin(pluginDir.absoluteFilePath(fileName));
    }

#ifdef Q_OS_WIN64
    //Restore standard DLL search path
    SetDllDirectoryA(NULL);
#endif
}

void CIkomiaRegistry::loadCppPlugin(const std::string& directory)
{
    QDir pluginDir(QString::fromStdString(directory));
    foreach (QString fileName, pluginDir.entryList(QDir::Files|QDir::NoSymLinks))
        _loadCppPlugin(pluginDir.absoluteFilePath(fileName));
}

std::vector<std::string> CIkomiaRegistry::getBlackListedPackages()
{
    std::vector<std::string> packages;
    QString blackListPath = Utils::IkomiaApp::getQIkomiaFolder() + "/Python/packageBlacklist.txt";

    if (QFile::exists(blackListPath))
    {
        QFile blackListFile(blackListPath);
        if (blackListFile.open(QIODevice::ReadOnly))
        {
            QTextStream in(&blackListFile);
            while (!in.atEnd())
                packages.push_back(in.readLine().toStdString());
        }
    }
    return packages;
}

void CIkomiaRegistry::clear()
{
    m_processRegistrator.reset();
    m_ioRegistrator.reset();
}

void CIkomiaRegistry::_loadCppPlugin(const QString &fileName)
{
    try
    {
        if(QLibrary::isLibrary(fileName))
        {
            QPluginLoader* pLoader;
            if(m_loaders.contains(fileName))
                pLoader = m_loaders[fileName];
            else
                pLoader = m_loaders.insert(fileName, new QPluginLoader(fileName)).value();

            // Get root component object of the plugin
            QObject* pObject = pLoader->instance();

            // Check if plugin is loaded or root component instantiated
            if(pObject == nullptr)
            {
                Utils::print(QString("Plugin %1 could not be loaded: %2.").arg(fileName).arg(pLoader->errorString()).toStdString(), QtWarningMsg);
                return;
            }

            // Check if plugin is a CPluginProcessInterface
            CPluginProcessInterface* pPlugin = qobject_cast<CPluginProcessInterface*>(pObject);
            if(pPlugin == nullptr)
            {
                Utils::print(QString("Plugin %1 interface is not valid.").arg(fileName).toStdString(), QtWarningMsg);
                return;
            }

            auto taskFactoryPtr = pPlugin->getProcessFactory();
            if(taskFactoryPtr == nullptr)
            {
                Utils::print(QString("Plugin %1 has no process factory.").arg(fileName).toStdString(), QtWarningMsg);
                return;
            }
            taskFactoryPtr->getInfo().setInternal(false);
            taskFactoryPtr->getInfo().setLanguage(ApiLanguage::CPP);
            taskFactoryPtr->getInfo().setOS(Utils::OS::getCurrent());

            auto version = QString::fromStdString(taskFactoryPtr->getInfo().getIkomiaVersion());
            auto state = Utils::Plugin::getCppState(version);

            if(state == PluginState::DEPRECATED)
            {
                QString str = QString("Plugin %1 is deprecated: based on Ikomia %2 while the current version is %3.")
                        .arg(QString::fromStdString(taskFactoryPtr->getInfo().getName()))
                        .arg(version)
                        .arg(Utils::IkomiaApp::getCurrentVersionNumber());
                Utils::print(str.toStdString(), QtWarningMsg);
                return;
            }
            else if(state == PluginState::UPDATED)
            {
                QString str = QString("Plugin %1 is not compatible: you must update Ikomia Studio to version %2.")
                        .arg(QString::fromStdString(taskFactoryPtr->getInfo().getName()))
                        .arg(version);
                Utils::print(str.toStdString(), QtWarningMsg);
                return;
            }

            auto widgetFactoryPtr = pPlugin->getWidgetFactory();
            if(widgetFactoryPtr == nullptr)
            {
                QString str = QString("Plugin %1 has no widget factory.").arg(QString::fromStdString(taskFactoryPtr->getInfo().getName()));
                Utils::print(str.toStdString(), QtWarningMsg);
                return;
            }

            m_processRegistrator.registerProcess(taskFactoryPtr, widgetFactoryPtr);
            Utils::print(QString("Plugin %1 is loaded.").arg(fileName).toStdString(), QtDebugMsg);
        }
    }
    catch(std::exception& e)
    {
        auto msg = QString("Plugin %1 failed to load: ").arg(fileName).toStdString();
        Utils::print(msg + e.what(), QtWarningMsg);
    }
}
