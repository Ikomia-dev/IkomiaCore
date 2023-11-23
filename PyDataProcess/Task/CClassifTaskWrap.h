#ifndef CCLASSIFTASKWRAP_H
#define CCLASSIFTASKWRAP_H

#include "PyDataProcessGlobal.h"
#include "Task/CClassificationTask.h"

class CClassifTaskWrap: public CClassificationTask, public wrapper<CClassificationTask>
{
    public:

        CClassifTaskWrap();
        CClassifTaskWrap(const std::string& name);
        CClassifTaskWrap(const CClassificationTask& task);

        ~CClassifTaskWrap() = default;

        std::string     repr() const override;
        std::string     default_repr() const;

        size_t          getProgressSteps() override;
        size_t          default_getProgressSteps();

        void            setActive(bool bActive) override;
        void            default_setActive(bool bActive);

        void            updateStaticOutputs() override;
        void            default_updateStaticOutputs();

        void            beginTaskRun() override;
        void            default_beginTaskRun();

        void            endTaskRun() override;
        void            default_endTaskRun();

        void            executeActions(int flags) override;
        void            default_executeActions(int flags);

        void            run() override;
        void            default_run();

        void            stop() override;
        void            default_stop();

        void            graphicsChanged() override;
        void            default_graphicsChanged();

        void            globalInputChanged(bool bNewSequence) override;
        void            default_globalInputChanged(bool bNewSequence);

        void            emitAddSubProgressSteps(int count);
        void            emitStepProgress();
        void            emitGraphicsContextChanged();
        void            emitOutputChanged();
        void            emitParametersModified();
};

#endif // CCLASSIFTASKWRAP_H
