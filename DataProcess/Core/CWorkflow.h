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

#ifndef CWORKFLOW_H
#define CWORKFLOW_H

#ifndef Q_MOC_RUN
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/depth_first_search.hpp>
#endif

#include <iostream>
#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTask.h"
#include "Workflow/CWorkflowEdge.hpp"
#include "Workflow/CWorkflowTaskWidget.h"
#include "Workflow/CWorkflowParam.h"
#include "Workflow/CWorkflowOutput.h"
#include "IO/CGraphicsInput.h"
#include "CRunTaskManager.h"

using WorkflowGraph = boost::adjacency_list<boost::listS,
                                            boost::listS,
                                            boost::bidirectionalS,
                                            std::shared_ptr<CWorkflowTask>,
                                            std::shared_ptr<CWorkflowEdge>>;

using WorkflowVertex = WorkflowGraph::vertex_descriptor;

using WorkflowEdge = WorkflowGraph::edge_descriptor;

using VertexIndexMap = std::map<WorkflowVertex, size_t>;

using BfsPropertyMap = boost::associative_property_map<VertexIndexMap>;

using WorkflowVertexRangeIt = std::pair<boost::graph_traits<WorkflowGraph>::vertex_iterator,
                                        boost::graph_traits<WorkflowGraph>::vertex_iterator>;

using WorkflowEdgeRangeIt = std::pair<boost::graph_traits<WorkflowGraph>::edge_iterator,
                                      boost::graph_traits<WorkflowGraph>::edge_iterator>;

using WorkflowInEdgeRangeIt = std::pair<boost::graph_traits<WorkflowGraph>::in_edge_iterator,
                                        boost::graph_traits<WorkflowGraph>::in_edge_iterator>;

using WorkflowOutEdgeRangeIt = std::pair<boost::graph_traits<WorkflowGraph>::out_edge_iterator,
                                         boost::graph_traits<WorkflowGraph>::out_edge_iterator>;

class CIkomiaRegistry;

class DATAPROCESSSHARED_EXPORT CWorkflowSignalHandler : public CSignalHandler
{
    Q_OBJECT

    signals:

        void    doFinishTask(const WorkflowVertex& id, CWorkflowTask::State status, const QString& msg = QString());
        void    doUpdateTaskItemView(const WorkflowTaskPtr& taskPtr, const WorkflowVertex& id);
        void    doUpdateTaskGraphicsInput(const GraphicsInputPtr& input);
        void    doSetTaskState(const WorkflowVertex& id, CWorkflowTask::State status, const QString& msg);
        void    doDeleteConnection(const WorkflowEdge& id);
        void    doDfsTaskVisited(const WorkflowVertex& id);
};

class DATAPROCESSSHARED_EXPORT CWorkflow : public CWorkflowTask
{
    public:

        using ExposedParams = std::map<std::string, CWorkflowParam>;

        //Constructors
        CWorkflow();
        CWorkflow(const std::string& name);
        CWorkflow(const std::string& name, CIkomiaRegistry* pRegistry, const GraphicsContextPtr &contextPtr);
        CWorkflow(const CWorkflow& workflow);
        CWorkflow(const CWorkflow&& workflow);

        //Destructor
        ~CWorkflow();

        //Operators
        CWorkflow&                      operator=(const CWorkflow& workflow);
        CWorkflow&                      operator=(const CWorkflow&& workflow);
        WorkflowTaskPtr                 operator[](WorkflowVertex v);
        WorkflowEdgePtr                 operator[](WorkflowEdge e);

        std::string                     repr() const;

        //Setters
        void                            setDescription(const std::string& description);
        void                            setKeywords(const std::string& keywords);
        void                            setInput(const WorkflowTaskIOPtr& pInput, size_t index) override;
        void                            setInput(const WorkflowTaskIOPtr& pInput, size_t index, bool bNewSequence);
        void                            setInputs(const InputOutputVect& inputs) override;
        void                            setInputs(const InputOutputVect& inputs, bool bNewSequence);
        void                            setOutputFolder(const std::string& folder) override;
        void                            setActiveTask(const WorkflowVertex& id);
        void                            setRunningTask(const WorkflowVertex& id);
        void                            setTaskActionFlag(const WorkflowVertex &id, ActionFlag action, bool bEnable);
        void                            setInputBatchState(size_t index, bool bBatch);
        void                            setCfgEntry(const std::string& key, const std::string& value);
        void                            setConfig(const MapString& conf);
        void                            setAutoSave(bool bEnable);
        void                            setExposedParameter(const std::string& name, const std::string& value);
        void                            setExposedOutputDescription(const WorkflowVertex& id, int outputIndex, const std::string& description);
        void                            setTaskEnabled(const WorkflowVertex& id, bool bEnable);

        //Getters
        std::string                     getDescription() const;
        std::string                     getKeywords() const;
        WorkflowVertex                  getRootId() const;
        size_t                          getTaskCount() const;
        size_t                          getChildCount(const WorkflowVertex& parent) const;
        size_t                          getChildIndex(const WorkflowVertex& parent, const WorkflowVertex& child) const;
        WorkflowVertex                  getLastTaskId() const;
        WorkflowVertex                  getActiveTaskId() const;
        WorkflowVertex                  getRunningTaskId() const;
        std::vector<WorkflowVertex>     getTaskIdList(const std::string& name) const;
        WorkflowTaskPtr                 getTask(const WorkflowVertex& id) const;
        std::vector<WorkflowVertex>     getParents(const WorkflowVertex& id) const;
        void                            getAllParents(const WorkflowVertex& id, std::vector<WorkflowVertex>& parents) const;
        std::vector<WorkflowVertex>     getChilds(const WorkflowVertex &parent) const;
        std::vector<WorkflowVertex>     getAllChilds(const WorkflowVertex &parent) const;
        std::vector<WorkflowTaskPtr>    getFinalTasks();
        WorkflowVertexRangeIt           getVertices();
        WorkflowEdgeRangeIt             getEdges();
        WorkflowInEdgeRangeIt           getInEdges(const WorkflowVertex& id);
        size_t                          getInEdgesCount(const WorkflowVertex& id) const;
        WorkflowOutEdgeRangeIt          getOutEdges(const WorkflowVertex& id);
        size_t                          getOutEdgeCount(const WorkflowVertex& id) const;
        WorkflowEdgePtr                 getEdge(const WorkflowEdge& id) const;
        WorkflowVertex                  getEdgeSource(const WorkflowEdge& id);
        WorkflowVertex                  getEdgeTarget(const WorkflowEdge& id);
        uint                            getHashValue() const;
        size_t                          getProgressSteps() override;
        size_t                          getProgressSteps(bool bTaskStepsOnly);
        size_t                          getProgressStepsFrom(const WorkflowVertex& idFrom) const;
        size_t                          getProgressStepsTo(const WorkflowVertex& idTo) const;
        GraphicsInputPtr                getGraphicsInput(const WorkflowTaskPtr& pTask);
        double                          getElapsedTimeTo(const WorkflowVertex& id);
        double                          getTotalElapsedTime() const;
        CDataInfoPtr                    getIOInfo(const WorkflowVertex& id, size_t index, bool bInput);
        std::vector<IODataType>         getRootTargetTypes() const;
        std::vector<std::string>        getRequiredTasks(const std::string& path);
        MapString                       getConfig() const;
        std::string                     getLastRunFolder() const;
        ExposedParams                   getExposedParameters() const;
        size_t                          getOutputCount() const override;
        WorkflowTaskIOPtr               getOutput(size_t index) const override;
        InputOutputVect                 getOutputs() const override;
        std::vector<CWorkflowOutput>    getExposedOutputs() const;
        IODataType                      getOutputDataType(size_t index) const override;

        bool                            hasOutput(const IODataType& type) const override;
        bool                            hasOutputData() const override;

        bool                            isRoot(const WorkflowVertex& id) const;
        bool                            isModified() const;
        bool                            isStopped() const;
        bool                            isInputConnected(size_t index) const;
        bool                            isConnectedToRoot(const WorkflowVertex& id) const;
        bool                            isValid(const WorkflowVertex& id);
        bool                            isLeafTask(const WorkflowVertex& id) const;
        bool                            isBatchMode() const;

        void                            addInput(const WorkflowTaskIOPtr& pInput) override;
        void                            addInput(const WorkflowTaskIOPtr&& pInput) override;
        void                            addInputs(const InputOutputVect &inputs) override;
        void                            addExposedParameter(const std::string& name, const std::string& description, const WorkflowVertex& taskId, const std::string& targetParamName);
        void                            addOutput(const std::string& description, const WorkflowVertex& taskId, int taskOutputIndex);

        void                            removeInput(size_t index) override;
        void                            removeOutput(const WorkflowVertex& taskId, int outputIndex);
        void                            removeExposedParameter(const std::string& name);

        WorkflowVertex                  addTask(const WorkflowTaskPtr& pNewTask);

        void                            replaceTask(const WorkflowTaskPtr& pNewTask, const WorkflowVertex& id);

        std::vector<WorkflowEdge>       connect(const WorkflowVertex& srcId, const WorkflowVertex& dstId);
        WorkflowEdge                    connect(const WorkflowVertex& srcId, size_t srcIndex, const WorkflowVertex& dstId, size_t dstIndex);

        void                            deleteTask(const WorkflowVertex& id);
        void                            deleteEdge(const WorkflowEdge& id);
        void                            deleteOutEdges(const WorkflowVertex& taskId, size_t srcIndex);

        void                            clear();
        void                            clearInputs() override;
        void                            clearOutputs() override;
        void                            clearAllOutputData();
        void                            clearExposedParameters();

        void                            run() override;
        void                            runFrom(const WorkflowVertex& id);
        void                            runTo(const WorkflowVertex& id);
        void                            runLastTask();
        void                            runTasks(const std::vector<WorkflowVertex>& taskToExecute);
        void                            runTask(const WorkflowVertex &id);

        void                            startIOAnalysis(const WorkflowVertex &idFrom);
        void                            startDepthFirstSearch();

        void                            stop();

        void                            analyzeTaskIO(const WorkflowVertex& id);

        void                            updateHash();
        void                            updateStartTime();

        void                            manageOutputs(const WorkflowVertex &taskId);

        void                            writeGraphviz(const std::string& path);

        void                            save(const std::string& path);

        void                            load(const std::string& path);

        void                            notifyGraphicsChanged();
        void                            notifyVideoStart(int frameCount) override;

        void                            workflowStarted() override;
        void                            workflowFinished() override;

        // Tells compiler (clang) we want all getProgressSteps functions
        using CWorkflowTask::getProgressSteps;

    protected:

        void                            to_ostream(std::ostream& os) const override;

    private:

        void                            initDefaultConfig();

        size_t                          getProgressSteps(const std::vector<WorkflowVertex>& tasks) const;
        std::vector<WorkflowVertex>     getForwardPassTasks(const WorkflowVertex& startVertex);
        std::vector<WorkflowVertex>     getSelfInputTasks() const;
        std::vector<WorkflowVertex>     getOrphanTasks() const;

        bool                            isVertexExists(const WorkflowVertex& id) const;

        void                            createRoot();
        BfsPropertyMap                  createBfsPropertyMap(VertexIndexMap &mapIndex) const;

        bool                            checkConnection(const WorkflowVertex& src, size_t srcIndex, const WorkflowVertex& target, size_t targetIndex);
        void                            checkBatchModeState();
        void                            checkOrphans() const;

        void                            connectSignals(const WorkflowTaskPtr& pNewTask);

        void                            findTaskToExecute(std::vector<WorkflowVertex>& list, const WorkflowVertex& id) const;
        void                            findPreviousTasks(std::vector<WorkflowVertex>& list, const WorkflowVertex& id);
        std::vector<std::pair<size_t, size_t> > findConnectionPorts(const WorkflowVertex& srcId, const WorkflowVertex& dstId);

        void                            resetTaskInput(WorkflowTaskPtr& taskPtr, size_t index);

        void                            runTasksSimple(const std::vector<WorkflowVertex>& taskToExecute);
        void                            runTasksVideo(const std::vector<WorkflowVertex>& taskToExecute);

        void                            stopVideoRead(const InputOutputVect& ioVect);
        void                            stopVideoWrite(const InputOutputVect& ioVect, int timeout);

        void                            clearOutputDataFrom(const WorkflowVertex& id);
        void                            clearOutputDataTo(const WorkflowVertex& id);
        void                            clearOutputData(const std::vector<WorkflowVertex>& tasks);

        void                            onTaskInputRemoved(size_t index);
        void                            onTaskOutputRemoved(size_t index);

        void                            decrementOutEdgesSrcIndex(const WorkflowVertex &taskId, size_t indexFrom);

        void                            updateCompositeInputName();

        void                            finalizeVideoRun(const std::vector<WorkflowVertex>& taskToExecute);

        void                            saveJSON(const std::string& path);

        void                            loadJSON(const std::string& path);

    private:

        std::mutex                      m_mutex;
        uint                            m_hashValue = 0;
        std::string                     m_description;
        std::string                     m_keywords;
        std::string                     m_compositeInputName;
        std::string                     m_startDate;
        std::string                     m_folder;
        WorkflowGraph                   m_graph;
        WorkflowVertex                  m_root;
        WorkflowVertex                  m_lastTaskAdded;
        WorkflowVertex                  m_activeTask;
        WorkflowVertex                  m_runningTask;
        std::vector<bool>               m_inputBatchState;
        std::atomic<bool>               m_bStopped{false};
        CIkomiaRegistry*                m_pRegistry = nullptr;
        GraphicsContextPtr              m_graphicsContextPtr = nullptr;
        CRunTaskManager                 m_runMgr;
        MapString                       m_cfg;
        // List of task parameters to be exposed at workflow level
        ExposedParams                   m_exposedParams;
        // List of task outputs to be exposed at workflow level
        std::vector<CWorkflowOutput>    m_exposedOutputs;
};

//---------------------------//
//- Class CFindChildVisitor -//
//---------------------------//
class CFindChildVisitor: public boost::default_bfs_visitor
{
    public:

        CFindChildVisitor(std::shared_ptr<std::vector<WorkflowVertex>> pChilds);
        void    discover_vertex(WorkflowVertex vertexId, const WorkflowGraph& graph);

    private:

        bool                                            m_bFirst = true;
        std::shared_ptr<std::vector<WorkflowVertex>>    m_pChilds = nullptr;
};

//------------------------------//
//- Class CProgressStepVisitor -//
//------------------------------//
class CProgressStepVisitor: public boost::default_bfs_visitor
{
    public:

        CProgressStepVisitor(size_t& stepCount, bool isBatchMode, bool isWholeVideo);
        CProgressStepVisitor(size_t&&) = delete; //prevent rvalue binding

        void    discover_vertex(WorkflowVertex vertexId, const WorkflowGraph& graph);

    private:

        bool    m_isBatchMode = false;
        bool    m_isWholeVideo = false;
        size_t& m_stepcount;
};

//------------------------------//
//- Class CElapsedTimeVisitor -//
//------------------------------//
class CElapsedTimeVisitor: public boost::default_bfs_visitor
{
    public:

        CElapsedTimeVisitor(double& elapsedTime);

        void    discover_vertex(WorkflowVertex vertexId, const WorkflowGraph& graph);

    private:

        double& m_elapsedTime;
};

//------------------------------//
//- Class CNotifyVideoVisitor -//
//------------------------------//
class CNotifyVideoVisitor: public boost::default_bfs_visitor
{
    public:

        CNotifyVideoVisitor(CWorkflow* pWorkflow, int frameCount);

        void    discover_vertex(WorkflowVertex vertexId, const WorkflowGraph& graph);

    private:

        CWorkflow*  m_pWorkflow = nullptr;
        int         m_frameCount = 0;
};

//----------------------------//
//- Class CGenericDfsVisitor -//
//----------------------------//
class CGenericDfsVisitor : public boost::default_dfs_visitor
{
    public:

        CGenericDfsVisitor(std::function<void(const WorkflowVertex& vertexId)> applyToVertex);

        void    discover_vertex(WorkflowVertex vertexId, const WorkflowGraph& graph);

    private:

        using VertexFunc = std::function<void(const WorkflowVertex& vertexId)>;
        VertexFunc  m_applyFunc = nullptr;
};

using WorkflowUPtr = std::unique_ptr<CWorkflow>;
using WorkflowPtr = std::shared_ptr<CWorkflow>;

#endif // CWORKFLOW_H
