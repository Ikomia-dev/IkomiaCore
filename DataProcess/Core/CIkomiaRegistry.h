#ifndef CIKOMIAREGISTRY_H
#define CIKOMIAREGISTRY_H

#include "DataProcessGlobal.hpp"
#include "CProcessRegistration.h"
#include "IO/CTaskIORegistration.h"


class CDllSearchPathAdder
{
    public:

        CDllSearchPathAdder(const std::string& directory)
        {
#ifdef Q_OS_WIN64
            //Add directory to the search path of the DLL loader
            SetDllDirectoryA(directory.c_str());
#endif
        }
        ~CDllSearchPathAdder()
        {
#ifdef Q_OS_WIN64
            //Restore standard DLL search path
            SetDllDirectoryA(NULL);
#endif
        }
        void addDirectory(const std::string& directory)
        {
#ifdef Q_OS_WIN64
            //Add directory to the search path of the DLL loader
            SetDllDirectoryA(directory.c_str());
#endif
        }
};

class DATAPROCESSSHARED_EXPORT CIkomiaRegistry
{
    public:

        CIkomiaRegistry();

        ~CIkomiaRegistry();

        void                        setPluginsDirectory(const std::string& dir);

        std::vector<std::string>    getAlgorithms() const;
        std::string                 getPluginsDirectory() const;
        std::string                 getPluginDirectory(const std::string& name) const;
        CTaskInfo                   getAlgorithmInfo(const std::string& name) const;
        CProcessRegistration*       getTaskRegistrator();
        CTaskIORegistration*        getIORegistrator();
        TaskFactoryPtr              getTaskFactory(const std::string& name) const;

        bool                        isAllLoaded() const;

        WorkflowTaskPtr             createInstance(const std::string& processName);
        WorkflowTaskPtr             createInstance(const std::string& processName, const WorkflowTaskParamPtr& paramPtr);
        WorkflowTaskWidgetPtr       createWidgetInstance(const std::string& processName, const WorkflowTaskParamPtr& paramPtr);

        void                        registerTask(const TaskFactoryPtr& factoryPtr);
        void                        registerTaskAndWidget(const TaskFactoryPtr& factoryPtr, WidgetFactoryPtr& widgetFactoryPtr);
        void                        registerIO(const TaskIOFactoryPtr& factoryPtr);

        void                        loadPlugins();
        void                        loadCppPlugins();
        void                        loadCppPlugin(const std::string &directory);
        void                        loadPythonPlugins();
        void                        loadPythonPlugin(const std::string &directory);
        boost::python::object       loadPythonMainModule(const std::string& folder, const std::string& name);

        static std::vector<std::string> getBlackListedPackages();

        void                        clear();

    private:

        void                        _loadCppPlugin(const QString &fileName);

    private:

        CProcessRegistration            m_processRegistrator;
        CTaskIORegistration             m_ioRegistrator;
        std::string                     m_pluginsDir;
        QMap<QString, QPluginLoader*>   m_loaders;
        bool                            m_bAllLoaded = false;
};

#endif // CIKOMIAREGISTRY_H
