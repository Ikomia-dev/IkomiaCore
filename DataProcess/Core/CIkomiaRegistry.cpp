#include "UtilsTools.hpp"
#include "CIkomiaRegistry.h"
#include "CPluginProcessInterface.hpp"
#include "Core/CPluginTools.h"

//-------------------------------//
//----- CDllSearchPathAdder -----//
//-------------------------------//
CDllSearchPathAdder::CDllSearchPathAdder(const std::string &directory)
{
#ifdef Q_OS_WIN64
    //Add directory to the search path of the DLL loader
    SetDllDirectoryA(directory.c_str());
#endif
}

CDllSearchPathAdder::~CDllSearchPathAdder()
{
#ifdef Q_OS_WIN64
    //Restore standard DLL search path
    SetDllDirectoryA(NULL);
#endif
}

void CDllSearchPathAdder::addDirectory(const std::string &directory)
{
#ifdef Q_OS_WIN64
    //Add directory to the search path of the DLL loader
    SetDllDirectoryA(directory.c_str());
#endif
}

//---------------------------//
//----- CIkomiaRegistry -----//
//---------------------------//
CIkomiaRegistry::CIkomiaRegistry()
{
    m_pluginsDir = Utils::IkomiaApp::getIkomiaFolder() + "/Plugins";

    // Python plugins directory is already added to Python path while Ikomia Studio is starting.
    // Moreover, Python is not initialized when single registry object is created which causes crash at startup.
    if (Utils::IkomiaApp::isAppStarted() == false)
        Utils::Python::addToPythonPath(m_pluginsDir + "/Python");
}

CIkomiaRegistry::~CIkomiaRegistry()
{
    for(auto&& it: m_loaders)
        delete it;
}

void CIkomiaRegistry::setPluginsDirectory(const std::string &dir)
{
    m_pluginsDir = dir;
    Utils::Python::addToPythonPath(m_pluginsDir + "/Python");
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
    try
    {
        auto info = getAlgorithmInfo(name);
        if (info.m_language == ApiLanguage::PYTHON)
            return m_pluginsDir + "/Python/" + name;
        else
            return m_pluginsDir + "/C++/" + name;
    }
    catch(CException)
    {
        // Algorithm not loaded
        // Check if algorithm directory exists
        std::string pluginDir = m_pluginsDir + "/Python/" + name;
        if (Utils::File::isFileExist(pluginDir))
            return pluginDir;
        else
        {
            pluginDir = m_pluginsDir + "/C++/" + name;
            if (Utils::File::isFileExist(pluginDir))
                return pluginDir;
        }
    }
    return "";
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

TaskFactoryPtr CIkomiaRegistry::getTaskFactory(const std::string& name) const
{
    return m_processRegistrator.getTaskFactory(name);
}

bool CIkomiaRegistry::isAllLoaded() const
{
    return m_bAllLoaded;
}

WorkflowTaskPtr CIkomiaRegistry::createInstance(const std::string &processName)
{
    WorkflowTaskPtr taskPtr = m_processRegistrator.createProcessObject(processName, nullptr);
    if (taskPtr == nullptr)
    {
        // Lazy loading
        std::string pluginDir = getPluginDirectory(processName);
        if (pluginDir.empty() == false)
        {
            try
            {
                loadPlugin(pluginDir);
                taskPtr = m_processRegistrator.createProcessObject(processName, nullptr);
            }
            catch (CException& e)
            {
                std::string msg = std::string("Failed to lazy load algorithm while creating instance:") + e.what();
                Utils::print(msg, QtMsgType::QtWarningMsg);
            }
        }
    }
    return taskPtr;
}

WorkflowTaskPtr CIkomiaRegistry::createInstance(const std::string &processName, const WorkflowTaskParamPtr &paramPtr)
{
    return m_processRegistrator.createProcessObject(processName, paramPtr);
}

WorkflowTaskPtr CIkomiaRegistry::createInstance(const std::string &processName, const UMapString &paramValues)
{
    auto paramPtr = m_processRegistrator.createParamObject(processName);
    if (paramPtr)
        paramPtr->merge(paramValues);
    else
    {
        std::string msg = processName + " does not implement parameter factory class. Given parameters will be ignored.";
        Utils::print(msg, QtMsgType::QtWarningMsg);
    }
    return m_processRegistrator.createProcessObject(processName, paramPtr);
}

WorkflowTaskWidgetPtr CIkomiaRegistry::createWidgetInstance(const std::string &processName, const WorkflowTaskParamPtr &paramPtr)
{
    return m_processRegistrator.createWidgetObject(processName, paramPtr);
}

void CIkomiaRegistry::registerTask(const TaskFactoryPtr &taskFactoryPtr, const TaskParamFactoryPtr& paramFactoryPtr)
{
    m_processRegistrator.registerProcess(taskFactoryPtr, nullptr, paramFactoryPtr);
}

void CIkomiaRegistry::registerTaskAndWidget(const TaskFactoryPtr &taskFactoryPtr, WidgetFactoryPtr &widgetFactoryPtr, const TaskParamFactoryPtr& paramFactoryPtr)
{
    m_processRegistrator.registerProcess(taskFactoryPtr, widgetFactoryPtr, paramFactoryPtr);
}

void CIkomiaRegistry::registerIO(const TaskIOFactoryPtr &factoryPtr)
{
    m_ioRegistrator.registerIO(factoryPtr);
}

void CIkomiaRegistry::loadPlugins()
{
    loadCppPlugins();
    loadPythonPlugins();
    m_bAllLoaded = true;
}

void CIkomiaRegistry::loadCppPlugins()
{
    QString pluginRootPath = QString::fromStdString(m_pluginsDir + "/C++");
    QDir pluginsDir(pluginRootPath);
    CDllSearchPathAdder dllSearch(QDir::toNativeSeparators(pluginRootPath).toStdString());

    //Load plugins placed directly in the root folder
    foreach (QString fileName, pluginsDir.entryList(QDir::Files|QDir::NoSymLinks))
    {
        try
        {
            _loadCppPlugin(pluginsDir.absoluteFilePath(fileName));
        }
        catch(CException& e)
        {
            Utils::print(e.getMessage(), QtWarningMsg);
        }
    }

    //Scan sub-directories
    foreach (QString directory, pluginsDir.entryList(QDir::Dirs|QDir::NoDotAndDotDot))
    {
        try
        {
            loadCppPlugin(pluginsDir.absoluteFilePath(directory).toStdString());
        }
        catch(CException& e)
        {
            Utils::print(e.getMessage(), QtWarningMsg);
        }
    }
}

void CIkomiaRegistry::loadCppPlugin(const std::string& directory)
{
    QString qDirectory = QString::fromStdString(directory);
    QDir pluginDir(qDirectory);
    CDllSearchPathAdder dllSearch(QDir::toNativeSeparators(qDirectory).toStdString());

    try
    {
        CPyEnsureGIL gil;
        foreach (QString fileName, pluginDir.entryList(QDir::Files|QDir::NoSymLinks))
            _loadCppPlugin(pluginDir.absoluteFilePath(fileName));
    }
    catch(CException& e)
    {
        std::string msg = "Loading failed: " + e.getMessage();
        throw CException(CoreExCode::INVALID_FILE, msg, __func__, __FILE__, __LINE__);
    }
}

void CIkomiaRegistry::_loadCppPlugin(const QString &fileName)
{
    if(!QLibrary::isLibrary(fileName))
    {
        // File is not a library, so not a plugin -> skip file
        return;
    }

    QPluginLoader* pLoader;
    if(m_loaders.contains(fileName))
        pLoader = m_loaders[fileName];
    else
        pLoader = m_loaders.insert(fileName, new QPluginLoader(fileName)).value();

    // Get root component object of the plugin
    QObject* pObject = pLoader->instance();

    // Check if plugin is loaded or root component instantiated (ie library is a Qt plugin)
    if(pObject == nullptr)
    {
        // Skip library
        Utils::print(QString("Library %1 is not a vaid Qt plugin: %2.").arg(fileName).arg(pLoader->errorString()).toStdString(), QtDebugMsg);
        return;
    }

    // Check if plugin is a CPluginProcessInterface (ie plugin is an Ikomia algorithm)
    CPluginProcessInterface* pPlugin = qobject_cast<CPluginProcessInterface*>(pObject);
    if(pPlugin == nullptr)
    {
        // Skip plugin
        Utils::print(QString("Plugin interface of %1 is not valid.").arg(fileName).toStdString(), QtDebugMsg);
        return;
    }

    // At this point, library file is an Ikomia plugin so we raise exception if error occurs
    auto taskFactoryPtr = pPlugin->getProcessFactory();
    if(taskFactoryPtr == nullptr)
    {
        std::string msg = QString("Algorithm %1 has no valid task factory.").arg(fileName).toStdString();
        throw CException(CoreExCode::INVALID_FILE, msg, __func__, __FILE__, __LINE__);
    }

    taskFactoryPtr->getInfo().m_bInternal = false;
    taskFactoryPtr->getInfo().m_language = ApiLanguage::CPP;
    taskFactoryPtr->getInfo().m_os = Utils::OS::getCurrent();
    // Check compatibility -> throw if incompatible
    checkCompatibility(taskFactoryPtr->getInfo());

    // Task parameters factory -> could be absent, introduced in 0.13.0
    auto paramFactoryPtr = pPlugin->getParamFactory();

    auto widgetFactoryPtr = pPlugin->getWidgetFactory();
    if(widgetFactoryPtr == nullptr)
    {
        std::string msg = QString("Algorithm %1 has no valid widget factory.")
                .arg(QString::fromStdString(taskFactoryPtr->getInfo().getName())).toStdString();
        throw CException(CoreExCode::INVALID_FILE, msg, __func__, __FILE__, __LINE__);
    }

    m_processRegistrator.registerProcess(taskFactoryPtr, widgetFactoryPtr, paramFactoryPtr);
    Utils::print(QString("Algorithm %1 is loaded.").arg(fileName).toStdString(), QtDebugMsg);
}

void CIkomiaRegistry::checkCompatibility(const CTaskInfo& info)
{
    PluginState state = Utils::Plugin::getApiCompatibilityState(info.m_minIkomiaVersion, info.m_maxIkomiaVersion, info.m_language);
    if(state != PluginState::VALID)
    {
        std::string msg;
        if (state == PluginState::DEPRECATED)
        {
            msg = "Algorithm " + info.getName() +
                    " is incompatible due to break changes in Ikomia API. Current Ikomia API version is " +
                    Utils::IkomiaApp::getCurrentVersionNumber() + ".";
        }
        else
        {
            std::string apiCondition = ">=" + info.m_minIkomiaVersion;
            if (info.m_maxIkomiaVersion.empty() == false)
                apiCondition += ", <" + info.m_maxIkomiaVersion;

            msg = "Algorithm " + info.getName() + " is incompatible. " +
                    "Ikomia API condition is " + apiCondition +
                    "while current Ikomia API version is " + Utils::IkomiaApp::getCurrentVersionNumber() + ".";
        }
        throw CException(CoreExCode::INVALID_VERSION, msg, __func__, __FILE__, __LINE__);
    }
}

void CIkomiaRegistry::loadPythonPlugins()
{
    QString pluginRootPath = QString::fromStdString(m_pluginsDir + "/Python");
    QDir pluginsDir(pluginRootPath);

    foreach (QString directory, pluginsDir.entryList(QDir::Dirs|QDir::NoDotAndDotDot))
    {
        try
        {
            loadPythonPlugin(pluginsDir.absoluteFilePath(directory).toStdString());
        }
        catch(CException& e)
        {
            std::string msg = "Loading failed: " + e.getMessage();
            Utils::print(msg, QtWarningMsg);
        }
    }
}

void CIkomiaRegistry::loadPythonPlugin(const std::string &directory)
{
    QString pluginName;

    try
    {
        CPyEnsureGIL gil;
        QDir pluginDir(QString::fromStdString(directory));

        foreach (QString fileName, pluginDir.entryList(QDir::Files))
        {
            auto pluginDirName = pluginDir.dirName();
            if(fileName == pluginDirName + ".py")
            {
                //Module names
                pluginName = pluginDirName;
                boost::python::object mainModule = loadPythonMainModule(directory, pluginName.toStdString());

                //Instantiate plugin factories
                auto pluginFactoryName = boost::python::str("IkomiaPlugin");
                boost::python::object pyFactory = mainModule.attr(pluginFactoryName)();
                boost::python::extract<CPluginProcessInterface*> exFactory(pyFactory);

                if(exFactory.check())
                {
                    auto plugin = exFactory();
                    auto taskFactoryPtr = plugin->getProcessFactory();
                    taskFactoryPtr->getInfo().m_bInternal = false;
                    taskFactoryPtr->getInfo().setLanguage(ApiLanguage::PYTHON);
                    // Check compatibility -> throw if incompatible
                    checkCompatibility(taskFactoryPtr->getInfo());

                    // Task parameters factory -> could be absent, introduced in 0.13.0
                    TaskParamFactoryPtr paramFactoryPtr = nullptr;
                    try
                    {
                        paramFactoryPtr = plugin->getParamFactory();
                    }
                    catch(CException& e)
                    {
                        Utils::print(e.getMessage(), QtDebugMsg);
                    }

                    // Plugin registration
                    if (Utils::IkomiaApp::isAppStarted())
                    {
                        auto widgetFactoryPtr = plugin->getWidgetFactory();
                        registerTaskAndWidget(taskFactoryPtr, widgetFactoryPtr, paramFactoryPtr);
                    }
                    else
                    {
                        registerTask(taskFactoryPtr, paramFactoryPtr);
                    }
                    Utils::print(QString("Algorithm %1 is loaded.").arg(fileName).toStdString(), QtDebugMsg);
                }
            }
        }
    }
    catch(boost::python::error_already_set&)
    {
        QString msg = QObject::tr("Algorithm %1 could not be loaded:").arg(pluginName);
        msg += QString::fromStdString(Utils::Python::handlePythonException());
        throw CException(CoreExCode::INVALID_FILE, msg.toStdString(), __func__, __FILE__, __LINE__);
    }
    catch(std::exception& e)
    {
        QString msg = QObject::tr("Algorithm %1 could not be loaded:").arg(pluginName);
        msg += QString::fromStdString(e.what());
        throw CException(CoreExCode::INVALID_FILE, msg.toStdString(), __func__, __FILE__, __LINE__);
    }
}

boost::python::object CIkomiaRegistry::loadPythonMainModule(const std::string& folder, const std::string &name)
{
    std::string mainModuleName = name + "." + name;
    std::string processName = mainModuleName + "_process";
    std::string widgetName = mainModuleName + "_widget";
    std::string moduleInit = name + "." + "__init__";

    if(Utils::Python::isModuleImported(mainModuleName))
    {
        boost::filesystem::directory_iterator iter(folder), end;
        for(; iter != end; ++iter)
        {
            if(boost::filesystem::is_directory(iter->path()) == true)
            {
                //Unload subfolder modules (need file __init__.py)
                if(Utils::Python::isFolderModule(iter->path().string()))
                    Utils::Python::unloadModule(iter->path().stem().string(), false);
            }
            else
            {
                //Unload sibling modules
                auto currentFile = iter->path().string();
                auto moduleName = name + "." + Utils::File::getFileNameWithoutExtension(currentFile);

                if(Utils::File::extension(currentFile) == ".py" &&
                        moduleName != processName &&
                        moduleName != widgetName &&
                        moduleName != mainModuleName &&
                        moduleName != moduleInit)
                {
                    Utils::Python::unloadModule(moduleName, true);
                }
            }
        }
    }

    //Load mandatory plugin interface modules - order matters
    Utils::CPluginTools::loadPythonModule(processName, true);
    if (Utils::IkomiaApp::isAppStarted())
        Utils::CPluginTools::loadPythonModule(widgetName, true);

    auto mainModule = Utils::CPluginTools::loadPythonModule(mainModuleName, true);
    Utils::CPluginTools::loadPythonModule(name, true);
    return mainModule;
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

void CIkomiaRegistry::loadPlugin(const std::string &directory)
{
    std::string pythonDir = "/Python/";
    if (directory.find(pythonDir) != std::string::npos)
        loadPythonPlugin(directory);
    else
    {
        std::string cppDir = "/C++/";
        if (directory.find(cppDir) != std::string::npos)
            loadCppPlugin(directory);
    }
}
