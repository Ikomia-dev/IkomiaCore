#ifndef CIKOMIAREGISTRY_H
#define CIKOMIAREGISTRY_H

#include "DataProcessGlobal.hpp"
#include "CProcessRegistration.h"
#include "IO/CTaskIORegistration.h"

//-------------------------------//
//----- CDllSearchPathAdder -----//
//-------------------------------//
class CDllSearchPathAdder
{
    public:

        CDllSearchPathAdder(const std::string& directory);

        ~CDllSearchPathAdder();

        void addDirectory(const std::string& directory);
};

//---------------------------//
//----- CIkomiaRegistry -----//
//---------------------------//
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

        WorkflowTaskPtr             createInstance(const std::string& taskName);
        WorkflowTaskPtr             createInstance(const std::string& taskName, const WorkflowTaskParamPtr& paramPtr);
        WorkflowTaskPtr             createInstance(const std::string& taskName, const UMapString& paramValues);
        WorkflowTaskWidgetPtr       createWidgetInstance(const std::string& taskName, const WorkflowTaskParamPtr &paramPtr);

        void                        registerTask(const TaskFactoryPtr& taskFactoryPtr, const TaskParamFactoryPtr &paramFactoryPtr=nullptr);
        void                        registerTaskAndWidget(const TaskFactoryPtr& taskFactoryPtr, WidgetFactoryPtr& widgetFactoryPtr, const TaskParamFactoryPtr &paramFactoryPtr=nullptr);
        void                        registerIO(const TaskIOFactoryPtr& factoryPtr);

        void                        loadPlugins();
        void                        loadCppPlugins();
        void                        loadCppPlugin(const std::string &directory);
        void                        loadPythonPlugins();
        void                        loadPythonPlugin(const std::string &directory);

        static std::vector<std::string> getBlackListedPackages();

        void                        clear();

    private:

        void                        loadPlugin(const std::string& directory);
        void                        _loadCppPlugin(const QString &fileName);
        boost::python::object       loadPythonMainModule(const std::string& folder, const std::string& name);

        void                        unloadPythonMainModel(const std::string& folder, const std::string& name);
        void                        checkCompatibility(const CTaskInfo &info);

    private:

        CProcessRegistration            m_processRegistrator;
        CTaskIORegistration             m_ioRegistrator;
        std::string                     m_pluginsDir;
        QMap<QString, QPluginLoader*>   m_loaders;
        bool                            m_bAllLoaded = false;
};

#endif // CIKOMIAREGISTRY_H
