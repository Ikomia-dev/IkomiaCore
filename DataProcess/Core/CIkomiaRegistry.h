#ifndef CIKOMIAREGISTRY_H
#define CIKOMIAREGISTRY_H

#include "DataProcessGlobal.hpp"
#include "CProcessRegistration.h"
#include "IO/CTaskIORegistration.h"

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

        WorkflowTaskPtr             createInstance(const std::string& processName);
        WorkflowTaskPtr             createInstance(const std::string& processName, const WorkflowTaskParamPtr& paramPtr);
        WorkflowTaskWidgetPtr       createWidgetInstance(const std::string& processName, const WorkflowTaskParamPtr& paramPtr);

        void                        registerTask(const TaskFactoryPtr& factoryPtr);
        void                        registerTaskAndWidget(const TaskFactoryPtr& factoryPtr, WidgetFactoryPtr& widgetFactoryPtr);
        void                        registerIO(const TaskIOFactoryPtr& factoryPtr);

        void                        loadCppPlugins();
        void                        loadCppPlugin(const std::string &directory);

        static std::vector<std::string> getBlackListedPackages();

        void                        clear();

    private:

        void                        _loadCppPlugin(const QString &fileName);

    private:

        CProcessRegistration            m_processRegistrator;
        CTaskIORegistration             m_ioRegistrator;
        std::string                     m_pluginsDir;
        QMap<QString, QPluginLoader*>   m_loaders;
};

#endif // CIKOMIAREGISTRY_H
