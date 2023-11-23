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

#ifndef CVIDEOOFTASKWRAP_H
#define CVIDEOOFTASKWRAP_H

#include "PyDataProcessGlobal.h"
#include "Task/CVideoOFTask.h"

//----------------------------------------------------------//
//-----------      CVideoOFTaskWrap         -------------//
//- Wrapping class to handle virtual functions and signals -//
//----------------------------------------------------------//
class CVideoOFTaskWrap: public CVideoOFTask, public wrapper<CVideoOFTask>
{
    public:

        CVideoOFTaskWrap();
        CVideoOFTaskWrap(const std::string& name);
        CVideoOFTaskWrap(const CVideoOFTask &process);

        size_t  getProgressSteps() override;
        size_t  default_getProgressSteps();

        void    setActive(bool bActive) override;
        void    default_setActive(bool bActive);

        void    updateStaticOutputs() override;
        void    default_updateStaticOutputs();

        void    beginTaskRun() override;
        void    default_beginTaskRun();

        void    endTaskRun() override;
        void    default_endTaskRun();

        CMat    drawOptFlowMapWrap(const CMat &flow, const CMat &cflowmap, int step);

        void    run() override;
        void    default_run();

        void    stop() override;
        void    default_stop();

        void    graphicsChanged() override;
        void    default_graphicsChanged();

        void    globalInputChanged(bool bNewSequence) override;
        void    default_globalInputChanged(bool bNewSequence);

        void    executeActions(int flags) override;
        void    default_executeActions(int flags);

        void    notifyVideoStart(int frameCount) override;
        void    default_notifyVideoStart(int frameCount);

        void    notifyVideoEnd() override;
        void    default_notifyVideoEnd();

        void    emitAddSubProgressSteps(int count);
        void    emitStepProgress();
        void    emitGraphicsContextChanged();
        void    emitOutputChanged();
        void    emitParametersModified();
};

#endif // CVIDEOOFTASKWRAP_H
