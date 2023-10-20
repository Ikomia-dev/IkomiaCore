// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CLogManager.h"
#include <iostream>
#include <QRegularExpression>

#ifdef Q_OS_WIN64
#include <io.h>
#else
#include <unistd.h>
#endif

CLogManager::CLogManager()
{
    m_pFileWatcher = new QFileSystemWatcher(this);
    connect(m_pFileWatcher, &QFileSystemWatcher::fileChanged, this, &CLogManager::onFileChanged);
}

CLogManager& CLogManager::instance()
{
    static CLogManager instance;
    return instance;
}

void CLogManager::handleMessage(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    QString fullMsg;

    //Do not display special character
    if(msg == "\n" || msg == "\t")
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    switch(type)
    {
        case QtDebugMsg:
            fullMsg = "Debug:";
            break;

        case QtInfoMsg:
            fullMsg = "Info:";
            break;

        case QtWarningMsg:
            fullMsg = "Warning:";
            break;

        case QtCriticalMsg:            
            fullMsg = "Critical:";
            break;

        case QtFatalMsg:            
            fullMsg = "Fatal:";
            break;
    }

    QString category;
    if (context.category && strcmp(context.category, "Python") == 0 && type == QtCriticalMsg)
    {
        // Python redirects almost all message to stderr, even no error message.
        // We apply here simple rules to qualify real message status
        processPythonMessage(msg, fullMsg, type);
    }
    else if (context.category && strcmp(context.category, "default") != 0)
    {
        category = QString(context.category);
        fullMsg += category + ": " + msg;
    }
    else
    {
        category = QString("Default");
        fullMsg += msg;
    }

    if(m_outputFunctions.size() > 0)
    {
        for(size_t i=0; i<m_outputFunctions.size(); ++i)
            if(m_outputFunctions[i] != nullptr)
                m_outputFunctions[i](type, fullMsg, category);
    }
    else
        std::cout << fullMsg.toStdString() << std::endl;
}

void CLogManager::processPythonMessage(const QString &msg, QString &fullMsg, QtMsgType& type)
{
    // Very simple rules (surely too simple) to check real status: info, warning or error.
    QRegularExpression errorExpr("(?i)error|exception");
    QRegularExpressionMatch match = errorExpr.match(msg);

    if(match.hasMatch())
    {
        type = QtCriticalMsg;
        fullMsg = "Critical:" + msg;
    }
    else
    {
        QRegularExpression warningExpr("(?i)warning");
        match = warningExpr.match(msg);

        if(match.hasMatch())
        {
            type = QtWarningMsg;
            fullMsg = "Critical:" + msg;
        }
        else
        {
            type = QtInfoMsg;
            fullMsg = "Info:" + msg;
        }
    }
}

void CLogManager::clearOutputManagers()
{
    m_outputFunctions.clear();
}

void CLogManager::setStdRedirection(const QString &path)
{
    freopen(path.toStdString().c_str(), "w", stdout);

#ifdef Q_OS_WIN64
    _dup2(fileno(stdout), fileno(stderr));
#else
    dup2(fileno(stdout), fileno(stderr));
#endif

    m_stdout_err.setFileName(path);
    m_stdout_err.open(QFile::ReadOnly | QFile::Text);
    m_stdStream.setDevice(&m_stdout_err);
    m_pFileWatcher->addPath(path);
}

void CLogManager::onFileChanged(const QString &filePath)
{
    if(m_outputFunctions.size() > 0)
    {
        if(filePath == m_stdout_err.fileName())
        {
            QString str;
            fflush(stdout);

            while(!m_stdStream.atEnd())
                str += m_stdStream.readLine() + "\n";

            if(!str.isEmpty())
                handleMessage(QtInfoMsg, QMessageLogContext(), str);
        }
    }
}

void CLogManager::addOutputManager(std::function<void(int, const QString&, const QString&)> outputFunc)
{
    m_outputFunctions.push_back(outputFunc);
}
