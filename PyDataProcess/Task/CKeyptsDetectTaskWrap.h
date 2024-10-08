#ifndef CKEYPTSDETECTTASKWRAP_H
#define CKEYPTSDETECTTASKWRAP_H

#include "PyDataProcessGlobal.h"
#include "Task/CKeypointDetectionTask.h"


class CKeyptsDetectTaskWrap: public CKeypointDetectionTask, public wrapper<CKeypointDetectionTask>
{
    public:

        CKeyptsDetectTaskWrap();
        CKeyptsDetectTaskWrap(const std::string& name);
        CKeyptsDetectTaskWrap(const CKeypointDetectionTask& task);

        ~CKeyptsDetectTaskWrap() = default;

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
        void            emitParametersChanged();
};

#endif // CKEYPTSDETECTTASKWRAP_H
