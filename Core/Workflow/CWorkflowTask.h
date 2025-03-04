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

#ifndef CWORKFLOWTASK_H
#define CWORKFLOWTASK_H

#include <QObject>
#ifdef Q_OS_WIN
#include <guiddef.h>
#endif
#include <QThread>
#include <QReadWriteLock>
#include <memory>
#include <mutex>
#include "CTimer.hpp"
#include "Main/CoreGlobal.hpp"
#include "CWorkflowTaskIO.h"
#include "Graphics/CGraphicsContext.h"
#include "CProgressSignalHandler.h"
#include "CViewPropertyIO.h"

class CGraphicsLayer;

class CORESHARED_EXPORT CSignalHandler : public CProgressSignalHandler
{
    Q_OBJECT

    public:

        CSignalHandler();

    signals:

        void doFinishWorkflow();

        void doInputRemoved(size_t index);
        void doOutputRemoved(size_t index);
        void doOutputChanged();

        void doAddGraphicsLayer(CGraphicsLayer* pLayer);
        void doRemoveGraphicsLayer(CGraphicsLayer* pLayer);
        void doGraphicsContextChanged();

        void doLog(const QString& msg);

        void doParametersChanged();

    public slots:

        void onOutputChanged();

        void onAddGraphicsLayer(CGraphicsLayer* pLayer);
        void onRemoveGraphicsLayer(CGraphicsLayer* pLayer);
        void onGraphicsContextChanged();

        void onFinishWorkflow();
};

/**
 * @ingroup groupCore
 * @brief The CWorkflowTask class is the base class for all process or task that aims to be executed in a workflow.
 * @details This class is the base class for all process or tasks that has to be executed in a workflow into the Ikomia environment.
 * It provides all basics mechanism to handle inputs and outputs, task parameters and progress feedbacks.
 * It also provides an interface throug virtual methods to set a formal scope to design custom task.
 * This interface allows the implementation of various image-based process. One can also use the derived class of the API which cover basic needs for image, video or volume processing task.
 */
class CORESHARED_EXPORT CWorkflowTask
{
    public:

        /**
         * @enum Type
         * @brief The Type enum defines the data types for which a process is dedicated.
         */
        enum class Type : size_t
        {
            GENERIC,            /**< Generic process */
            IMAGE_PROCESS_2D,   /**< Process or task dedicated to standard image (2D) */
            IMAGE_PROCESS_3D,   /**< Process or task dedicated to volume */
            VIDEO,              /**< Process or task dedicated to video */
            DNN_TRAIN           /**< Process or task dedicated to DNN training */
        };

        /**
         * @enum State
         * @brief The State enum defines the execution states of a process.
         */
        enum class State : size_t
        {
            UNDONE,     /**< Not executed */
            VALIDATE,   /**< Execution succeed */
            _ERROR      /**< Execution failed */
        };

        /**
         * @enum ActionFlag
         * @brief The ActionFlag enum defines specific behaviors or actions that can be enable/disable for a process.
         */
        enum class ActionFlag : size_t
        {
            APPLY_VOLUME,       /**< Action to execute a task dedicated to standard image to a volume */
            OUTPUT_AUTO_EXPORT  /**< Action to export automatically task outputs */
        };

        /**
         * @brief Default constructor.
         */
        CWorkflowTask();
        /**
         * @brief Constructor with the given task name.
         */
        CWorkflowTask(const std::string& name);
        /**
         * @brief Copy constructor.
         */
        CWorkflowTask(const CWorkflowTask& task);
        /**
         * @brief Universal reference copy constructor.
         */
        CWorkflowTask(const CWorkflowTask&& task);

        /**
         * @brief Assignement operator.
         */
        CWorkflowTask& operator=(const CWorkflowTask& task);
        /**
         * @brief Universal reference assignement operator.
         */
        CWorkflowTask& operator=(const CWorkflowTask&& task);

        friend CORESHARED_EXPORT std::ostream& operator<<(std::ostream& os, const CWorkflowTask& task);

        virtual ~CWorkflowTask();

        virtual std::string         repr() const;

        /**
         * @brief Performs heavy initialization of the task.
         * This method can be overriden to put custom initialization steps.
         * It is well suited for models donwloading, models loading or models compiling.
         * The function will be automatically called in beginTaskRun if it has not been called before.
         * Don't forget to call the base class method.
         */
        virtual void                initLongProcess();

        //Setters
        /**
         * @brief Sets the data type for the input at position index.
         * If the input at position index does not exist, the function creates as many generic inputs to reach the number of index+1 and sets the data type for the input at position index.
         * @param dataType: see ::IODataType for details.
         * @param index: zero-based index of the input.
         */
        virtual void                setInputDataType(const IODataType& dataType, size_t index = 0);
        /**
         * @brief Sets input at position index with the given one.
         * * If the input at position index does not exist, the function creates as many generic inputs to reach the number of index+1 and sets the input at position index.
         * @param pInput: CWorkflowTaskIO based shared pointer.
         * @param index: zero-based index of the input.
         * @param bNewSequence: unused
         */
        virtual void                setInput(const WorkflowTaskIOPtr& pInput, size_t index = 0);
        void                        setInputNoCheck(const WorkflowTaskIOPtr& pInput, size_t index = 0);
        /**
         * @brief Sets the whole list of inputs with the given one.
         * @param inputs: vector of CWorkflowTaskIO based shared pointer.
         * @param bNewSequence: unused
         */
        virtual void                setInputs(const InputOutputVect& inputs);
        /**
         * @brief Sets the data type for the output at position index.
         * If the output at position index does not exist, the function creates as many generic outputs to reach the number of index+1 and sets the data type for the output at position index.
         * @param dataType: see ::IODataType for details.
         * @param index: zero-based index of the output.
         */
        virtual void                setOutputDataType(const IODataType& dataType, size_t index = 0);
        /**
         * @brief Sets output at position index with the given one.
         * * If the output at position index does not exist, the function creates as many generic outputs to reach the number of index+1 and sets the output at position index.
         * @param pInput: CWorkflowTaskIO based shared pointer.
         * @param index: zero-based index of the input.
         * @param bNewSequence: unused
         */
        virtual void                setOutput(const WorkflowTaskIOPtr& pOutput, size_t index = 0);
        /**
         * @brief Sets the whole list of outputs with the given one.
         * @param outputs: vector of CWorkflowTaskIO based shared pointer.
         */
        virtual void                setOutputs(const InputOutputVect &outputs);
        /**
         * @brief Sets the task parameters object.
         * @param pParam: CWorkflowTaskParam based shared pointer.
         */
        void                        setParam(const WorkflowTaskParamPtr& pParam);
        /**
         * @brief Sets the map structure storing the parameters definition (name and value for each).
         * This map is used to have persistent storage of task parameters when user saves a workflow.
         * @param paramMap: map containing pairs of std::string: key=parameter name, value=parameter value.
         */
        void                        setParamValues(const UMapString& paramMap);
        /**
         * @brief Sets the active state of the task.
         * The active task is the one selected in the workflow, thus, user has access to parameters and can visualize results associated with the task.
         * @param bActive: true or false.
         */
        virtual void                setActive(bool bActive);
        /**
         * @brief Sets the task name.
         * @param name: new name of the task.
         */
        void                        setName(const std::string& name);
        /**
         * @brief Set the output folder.
         * @param name: full path of output folder.
         */
        virtual void                setOutputFolder(const std::string& folder);
        /**
         * @brief Enables or disables the given action.
         * If the action does not exist, it is added to the list.
         * @param flag: see ::ActionFlag for more details.
         * @param bEnable: true to enable, false to disable.
         */
        void                        setActionFlag(ActionFlag flag, bool bEnable);
        /**
         * @brief Sets the graphics context of the task.
         * A task can have output of type Graphics, which means that some graphical information will be overlaid on result image.
         * The graphics context contains all display properties (color, thickness...) of each graphics tools.
         * So you can set this context to customize the way the task outputs grapĥics objects.
         * @param contextPtr: shared pointer to CGraphicsContext instance.
         */
        void                        setGraphicsContext(const GraphicsContextPtr& contextPtr);
        /**
         * @brief Set batch input mode.
         * @param bBatch: True in case of batch input, False otherwise.
         */
        void                        setBatchInput(bool bBatch);
        /**
         * @brief Enable/disable auto-export for all outputs.
         * @param bEnable: True or False.
         */
        void                        setAutoSave(bool bEnable);
        /**
         * @brief Enable/disable task for running.
         * @param bEnable: True or False.
         */
        void                        setEnabled(bool bEnable);

        //Getters
        /**
         * @brief Gets the data type for which the task is dedicated.
         * @return Type can be one those values ::Type.
         */
        Type                        getType() const;
        /**
         * @brief Gets the task uuid.
         * @return UUID of the task.
         */
        std::string                 getUUID() const;
        /**
         * @brief Gets the task name.
         * @return Name of the task.
         */
        std::string                 getName() const;
        /**
         * @brief Gets the number of inputs (even if input pointer is null).
         * @return Inputs count.
         */
        size_t                      getInputCount() const;
        /**
         * @brief Gets the number of non-null inputs.
         * @return Non-null inputs count.
         */
        size_t                      getValidInputCount() const;
        /**
         * @brief Gets the whole list of inputs.
         * @return Vector of CWorkflowTaskIO based shared pointer.
         */
        InputOutputVect             getInputs() const;
        /**
         * @brief Gets the list of inputs of given types.
         * @param types: set of IO types to return.
         * @return Vector of CWorkflowTaskIO based shared pointer.
         */
        InputOutputVect             getInputs(const std::set<IODataType>& types) const;
        /**
         * @brief Gets the input at position index.
         * @param index: zero-based index.
         * @return CWorkflowTaskIO based shared pointer.
         */
        WorkflowTaskIOPtr           getInput(size_t index) const;
        /**
         * @brief Gets data type of input at position index.
         * This data type can differ from the original type because it can change at runtime according to the data source.
         * @param index: zero-based index.
         * @return data type ::IODataType.
         */
        IODataType                  getInputDataType(size_t index) const;
        /**
         * @brief Gets original data type of input at position index as it was defined at design time.
         * It's not dependent on the dynamically defined type of the current data source.
         * @param index: zero-based index.
         * @return
         */
        IODataType                  getOriginalInputDataType(size_t index) const;
        /**
         * @brief Gets the number of outputs (even if pointer is null).
         * @return Outputs count.
         */
        virtual size_t              getOutputCount() const;
        /**
         * @brief Gets the whole list of outputs.
         * @return Vector of CWorkflowTaskIO based shared pointer.
         */
        virtual InputOutputVect     getOutputs() const;
        /**
         * @brief Gets the list of outputs of the given data type.
         * @param types: set of IO types to return.
         * @return Vector of CWorkflowTaskIO based shared pointer.
         */
        InputOutputVect             getOutputs(const std::set<IODataType> &dataTypes) const;
        /**
         * @brief Gets the output at position index.
         * @param index: zero-based index.
         * @return CWorkflowTaskIO based shared pointer.
         */
        virtual WorkflowTaskIOPtr   getOutput(size_t index) const;
        /**
         * @brief Gets data type of output at position index.
         * @param index: zero-based index.
         * @return data type ::IODataType.
         */
        virtual IODataType          getOutputDataType(size_t index) const;
        /**
         * @brief Get output folder.
         * @return full path to the output folder.
         */
        std::string                 getOutputFolder() const;
        /**
         * @brief Gets parameter object instance.
         * @return CWorkflowTaskParam based shared pointer.
         */
        WorkflowTaskParamPtr        getParam() const;
        /**
         * @brief Gets parameters values.
         * @return Unordered map of std::string pair (key, value).
         */
        UMapString                  getParamValues() const;

        /**
         * @brief Gets unique hash value identifying the task.
         * This hash value is computed from multiple elements to ensure uniqueness.
         * Internally, the system use it to know if the task has been modified.
         * @return Hash value.
         */
        uint                        getHashValue() const;
        /**
         * @brief Gets the number of progress steps when the system runs the task.
         * @return Number of steps.
         */
        virtual size_t              getProgressSteps();
        /**
         * @brief Gets the time of the last execution.
         * @return Elapsed time in milliseconds.
         */
        double                      getElapsedTime() const;
        /**
         * @brief Gets custom information of current task.
         * @return List of properties with corresponding value (vector of pair< string, string >)
         */
        VectorPairString            getCustomInfo() const;
        /**
         * @brief Gets the first output of the given data type.
         * @param type: see ::IODataType for more details.
         * @param index: relative zero-based index of output in case multiple outputs of the given type exist.
         * @return CWorkflowTaskIO based shared pointer.
         */
        WorkflowTaskIOPtr           getOutputFromType(const IODataType& type, size_t index=0) const;
        /**
         * @brief Gets specific behaviors list and their activation state.
         * @return A map containing pairs of std::string where key=::ActionFlag and value=state.
         */
        std::map<ActionFlag,bool>   getActionFlags() const;
        /**
         * @brief Gets the view properties of input at position index.
         * The view properties contain information about how inputs have to be displayed (maximized state, zoom...).
         * @param index: zero-based index
         * @return Pointer to a CViewPropertyIO instance.
         */
        CViewPropertyIO*            getInputViewProperty(size_t index);
        /**
         * @brief Gets the view properties of output at position index.
         * The view properties contain information about how outputs have to be displayed (maximized state, zoom...).
         * @param index: zero-based index
         * @return Pointer to a CViewPropertyIO instance.
         */
        CViewPropertyIO*            getOutputViewProperty(size_t index);

        // Signal raw pointer for connecting things
        /** @cond INTERNAL */
        CSignalHandler*             getSignalRawPtr();
        /** @endcond */

        /**
         * @brief Checks if the given specific behavior is enabled.
         * @param flag: see ::ActionFlag.
         * @return True is behavior is enabled, False otherwise.
         */
        bool                        isActionFlagEnable(ActionFlag flag) const;
        /**
         * @brief Checks if the task is running.
         * @return True if it's running, False otherwise.
         */
        bool                        isRunning() const;

        /**
         * @brief Checks if the task is listening to graphics changed event.
         * @return  True or False.
         */
        virtual bool                isGraphicsChangedListening() const;

        /**
         * @brief Checks if the task does not need external inputs.
         * @return  True or False.
         */
        bool                        isSelfInput() const;
        /**
         * @brief Check if the task has at least one output with auto-save enabled.
         * @return  True or False.
         */
        bool                        isAutoSave() const;
        /**
         * @brief Check if the task is enabled for running.
         * @return  True or False.
         */
        bool                        isEnabled() const;

        /**
         * @brief Checks if the task has output of the given data type.
         * @param type: see IODataType.
         * @return True if it has an output of the given data type, False otherwise.
         */
        virtual bool                hasOutput(const IODataType& type) const;
        /**
         * @brief Checks if the task has output of the given data types.
         * @param types: set of IO data types (see IODataType).
         * @return True if it has an output of the given data type, False otherwise.
         */
        bool                        hasOutput(const std::set<IODataType>& types) const;
        /**
         * @brief Checks if the task has at least one output with valid data.
         * Output data are filled after a successful run of the task.
         * @return True if at least one output has valid data, False otherwise.
         */
        virtual bool                hasOutputData() const;

        /**
         * @brief Checks if the task has input of the given data type.
         * @param type: see IODataType.
         * @return True if it has an input of the given data type, False otherwise.
         */
        bool                        hasInput(const IODataType& type) const;
        /**
         * @brief Checks if the task has input of the given data types.
         * @param types: set of IO data types (see IODataType).
         * @return True if it has an input of the given data type, False otherwise.
         */
        bool                        hasInput(const std::set<IODataType>& types) const;

        //Methods
        /**
         * @brief Appends an input.
         * @param pInput: CWorkflowTaskIO based shared pointer. If null is passed, an empty input is added.
         */
        virtual void                addInput(const WorkflowTaskIOPtr& pInput);
        /**
         * @brief Appends an input. Universal reference version.
         * @param pInput: CWorkflowTaskIO based shared pointer. If null is passed, an empty input is added.
         */
        virtual void                addInput(const WorkflowTaskIOPtr&& pInput);
        /**
         * @brief Appends an output.
         * @param pInput: CWorkflowTaskIO based shared pointer.
         */
        virtual void                addOutput(const WorkflowTaskIOPtr& pOutput);
        /**
         * @brief Appends an output. Universal reference version.
         * @param pInput: CWorkflowTaskIO based shared pointer.
         */
        virtual void                addOutput(const WorkflowTaskIOPtr&& pOutput);
        /**
         * @brief Appends the given list of inputs.
         * @param inputs: vector of CWorkflowTaskIO based shared pointers.
         */
        virtual void                addInputs(const InputOutputVect& inputs);
        /**
         * @brief Appends the given list of outputs.
         * @param outputs: vector of CWorkflowTaskIO based shared pointers.
         */
        virtual void                addOutputs(const InputOutputVect& outputs);

        /**
         * @brief Insert given input at position index.
         * If index exceeds the input list size, the given input is appended.
         * @param pInput: CWorkflowTaskIO based shared pointer.
         * @param index: zero-based index.
         */
        void                        insertInput(const WorkflowTaskIOPtr& pInput, size_t index);
        /**
         * @brief Insert given output at position index.
         * If index exceeds the output list size, the given output is appended.
         * @param pOutput: CWorkflowTaskIO based shared pointer.
         * @param index: zero-based index.
         */
        void                        insertOutput(const WorkflowTaskIOPtr& pOutput, size_t index);

        /**
         * @brief Removes all inputs.
         */
        virtual void                clearInputs();
        /**
         * @brief Removes all outputs.
         */
        virtual void                clearOutputs();
        /**
         * @brief Clears the data of input at position index.
         * @param index: zero-based index.
         */
        void                        clearInputData(size_t index);
        /**
         * @brief Clears the data of all outputs.
         */
        void                        clearOutputData();

        /**
         * @brief Replaces the input at position index by the given one.
         * @param index: zero-based index.
         * @param io: CWorkflowTaskIO based shared pointer.
         */
        void                        resetInput(size_t index, WorkflowTaskIOPtr &io);

        /**
         * @brief Removes input at position index.
         * @param index: zero-based index.
         */
        virtual void                removeInput(size_t index);
        /**
         * @brief Removes output at position index.
         * @param index: zero-based index.
         */
        void                        removeOutput(size_t index);
        /**
         * @brief Removes specific task behavior corresponding to the given action flag.
         * @param flag: see ::ActionFlag.
         */
        void                        removeActionFlag(ActionFlag flag);

        /**
         * @brief Create locks for all task inputs in multi-threaded context.
         * The lock is limited to the scope of returned vector instance and can thus protect access at object level.
         */
        std::vector<TaskIOLockerUPtr> createInputScopedLocks() const;
        /**
         * @brief Create locks for all task outputs in multi-threaded context.
         * The lock is limited to the scope of returned vector instance and can thus protect access at object level.
         */
        std::vector<TaskIOLockerUPtr> createOutputScopedLocks() const;

        /**
         * @brief Runs the task.
         * It's where the main process of the task has to be implemented.
         * In this base class, the methods just forwards the inputs to outputs.
         * It has to be overriden in derived class.
         */
        virtual void                run();

        /**
         * @brief Executes actions according to the defined specific behavior.
         * In this base class, this method does nothing.
         * @param flags: unused.
         */
        virtual void                executeActions(int flags);

        /**
         * @brief Updates the static information deduced from inputs.
         * The static data corresponds to all data that can be deduced without the runtime context.
         * This method is called each time inputs changed.
         */
        virtual void                updateStaticOutputs();

        /**
         * @brief Performs all initialization stuff before running the task.
         * This method can be overriden to put custom initialization steps. In this case, don't forget to call the base class method.
         * This method must be the first call of the run method.
         */
        virtual void                beginTaskRun();
        /**
         * @brief Performs all finalization stuff after running the task.
         * This method can be overriden to put custom finalization steps. In this case, don't forget to call the base class method.
         * This method must be the last call of the run method.
         */
        virtual void                endTaskRun();

        /**
         * @brief Notifies that the parameters have changed.
         * This method can be overriden to implement custom actions when this event happens.
         * The method does nothing in this base class.
         */
        virtual void                parametersModified();
        /**
         * @brief Notifies that graphics layers of input images have changed.
         * This method can be overriden to implement custom actions when this event happens.
         * The method does nothing in this base class.
         */
        virtual void                graphicsChanged();
        /**
         * @brief Notifies that the inputs of the workflow have changed.
         * This method can be overriden to implement custom actions when this event happens.
         * The method does nothing in this base class.
         * @param bNewSequence: unused
         */
        virtual void                globalInputChanged(bool bNewSequence);

        /**
         * @brief Notifies that the workflow video inputs have started.
         * This method can be overriden to implement custom actions when this event happens.
         * The method does nothing in this base class.
         */
        virtual void                notifyVideoStart(int);
        /**
         * @brief Notifies that the workflow video inputs have stopped.
         * This method can be overriden to implement custom actions when this event happens.
         * The method does nothing in this base class.
         */
        virtual void                notifyVideoEnd();

        /**
         * @brief Notifies that the task is requested to stop.
         * It is higly recommended to manage this stop event and override the method for time-consuming task.
         * Base class implementation must be called before any other instructions.
         */
        virtual void                stop();

        /**
         * @brief Notify that the workflow is started.
         * This method can be overriden to implement custom actions when a workflow is started (before the task is run)
         */
        virtual void                workflowStarted();
        /**
         * @brief Notify that the workflow is finished.
         * This method can be overriden to implement custom actions when a workflow is finished (after the task is run)
         */
        virtual void                workflowFinished();

        virtual void                saveOutputs(const std::string &baseName) const;

        QJsonObject                 toJson() const;

        void                        download(const std::string& url, const std::string& to);

    protected:

        virtual void                to_ostream(std::ostream& os) const;

    private:

        std::string                 generateUUID() const;

    /** @cond INTERNAL */
    protected:

        Type                                m_type = Type::GENERIC;
        std::unique_ptr<CSignalHandler>     m_signalHandler;
        std::string                         m_uuid = "";
        std::string                         m_name = "";
        std::string                         m_outputFolder = "";
        WorkflowTaskParamPtr                m_pParam = nullptr;
        double                              m_elapsedTime = 0.0;
        std::map<ActionFlag,bool>           m_actionFlags;
        std::mutex                          m_mutex;
        Utils::CTimer                       m_timer;
        std::atomic<bool>                   m_bRunning{false};
        std::atomic<bool>                   m_bStop{false};
        bool                                m_bActive = false;
        bool                                m_bEnabled = true;
        bool                                m_bInitLongProcess = false;
        GraphicsContextPtr                  m_graphicsContextPtr = nullptr;
        ViewPropertiesIO                    m_inputViewProps;
        ViewPropertiesIO                    m_outputViewProps;
        VectorPairString                    m_customInfo;
    /** @endcond */

        InputOutputVect                     m_inputs;
        InputOutputVect                     m_outputs;
        std::vector<IODataType>             m_originalInputTypes;
};

using WorkflowTaskPtr = std::shared_ptr<CWorkflowTask>;

#endif // CWORKFLOWTASK_H

