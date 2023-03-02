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

#ifndef C2DIMAGEINTERACTIVETASKWRAP_H
#define C2DIMAGEINTERACTIVETASKWRAP_H

#include "PyDataProcessGlobal.h"
#include "Task/C2dImageInteractiveTask.h"

//----------------------------------------------------------//
//-----------  C2dImageInteractiveTaskWrap  -------------//
//- Wrapping class to handle virtual functions and signals -//
//----------------------------------------------------------//
class C2dImageInteractiveTaskWrap: public C2dImageInteractiveTask, public wrapper<C2dImageInteractiveTask>
{
    public:

        C2dImageInteractiveTaskWrap();
        C2dImageInteractiveTaskWrap(const std::string& name);
        C2dImageInteractiveTaskWrap(const C2dImageInteractiveTask& process);

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

        void    executeActions(int flags) override;
        void    default_executeActions(int flags);

        void    run() override;
        void    default_run();

        void    stop() override;
        void    default_stop();

        void    graphicsChanged() override;
        void    default_graphicsChanged();

        void    globalInputChanged(bool bNewSequence) override;
        void    default_globalInputChanged(bool bNewSequence);

        void    emitAddSubProgressSteps(int count);
        void    emitStepProgress();
        void    emitGraphicsContextChanged();
        void    emitOutputChanged();
};

#endif // C2DIMAGEINTERACTIVETASKWRAP_H
