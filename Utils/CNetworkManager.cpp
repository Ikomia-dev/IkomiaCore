#include "CNetworkManager.h"
#include <QCoreApplication>
#include <QThread>
#include <QtNetwork/QNetworkAccessManager>
#include <QFile>
#include <QFileInfo>
#include "CException.h"
#include "UtilsTools.hpp"

CNetworkManager::CNetworkManager(QObject *parent): QObject(parent)
{
    // QThread is required, otherwise QEventLoop will block
    m_pSocketThread = new QThread(this);
    moveToThread(m_pSocketThread);
    m_pSocketThread->start(QThread::HighestPriority);
}

CNetworkManager::~CNetworkManager()
{
    // Ensure we exit all loops
    m_loop.exit(1);

    if (m_pSocketThread != nullptr)
    {
        m_pSocketThread->quit();
        m_pSocketThread->wait();
        m_pSocketThread->deleteLater();
        m_pSocketThread = nullptr;
    }
}

bool CNetworkManager::isQtAppStarted() const
{
    QCoreApplication *pCoreApp = QCoreApplication::instance();
    return pCoreApp != nullptr;
}

bool CNetworkManager::isGuiThread() const
{
    QCoreApplication *pCoreApp = QCoreApplication::instance();
    if (pCoreApp == nullptr)
        return false;

    return (QThread::currentThread() == pCoreApp->thread());
}

void CNetworkManager::download(const std::string &url, const std::string &to)
{
    if (isQtAppStarted())
    {
        QUrl qurl(QString::fromStdString(url));
        if(qurl.isValid() == false)
            throw CException(CoreExCode::INVALID_PARAMETER, "Invalid url", __func__, __FILE__, __LINE__);

        QString pathTo = QString::fromStdString(to);
        QFileInfo info(pathTo);

        if (info.isDir())
            pathTo = QString::fromStdString(to) + "/" + qurl.fileName();

        if (isGuiThread())
        {
            // For GUI threads, we use the non-blocking call and use QEventLoop to wait and yet keep the GUI alive
            bool bOk = QMetaObject::invokeMethod(this, "slotDownload", Qt::QueuedConnection,
                                                 Q_ARG(void*, &m_loop),
                                                 Q_ARG(QString, qurl.url()),
                                                 Q_ARG(QString, pathTo));
            assert(bOk);
            m_loop.exec();
        }
        else
        {
            // For non-GUI threads, QEventLoop would cause a deadlock, so we simply use a blocking call.
            // (Does not hurt as no messages need to be processed either during the open operation).
            bool bOk = QMetaObject::invokeMethod(this, "slotDownload", Qt::BlockingQueuedConnection,
                                                 Q_ARG(void*, nullptr),
                                                 Q_ARG(QString, qurl.url()),
                                                 Q_ARG(QString, pathTo));
            assert(bOk);
        }
    }
    else
    {
        // From Python -> use Python
        downloadWithPython(url, to);
    }
}

void CNetworkManager::slotDownload(void* pLoop, QString url, QString to)
{
    workerDownload(url, to);

    if (pLoop != nullptr)
    {
        ((QEventLoop*)pLoop)->wakeUp();
        QMetaObject::invokeMethod((QEventLoop*)pLoop, "quit", Qt::QueuedConnection);
    }
}

void CNetworkManager::workerDownload(const QString& url, const QString& to)
{
    QNetworkAccessManager* pNetAccessMgr = new QNetworkAccessManager(this);
    auto pReply = pNetAccessMgr->get(QNetworkRequest(url));

    QEventLoop loop;
    connect(pReply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    QNetworkReply::NetworkError err = pReply->error();
    if (err != QNetworkReply::NoError)
    {
        std::cout << pReply->errorString().toStdString() << std::endl;
        return;
    }

    QByteArray data = pReply->readAll();
    QFile file(to);
    file.open(QIODevice::WriteOnly);
    file.write(data);
    file.close();

    pReply->close();
    pReply->deleteLater();
}

void CNetworkManager::downloadWithPython(const std::string &url, const std::string &to)
{
    std::string formattedTo = to;
#ifdef Q_OS_WIN64
    Utils::String::replace(formattedTo, "\\", "\\\\");
#endif

    QString script = QString(
                "import requests\n"
                "import shutil\n\n"
                "with requests.get('%1', stream=True) as r:\n"
                "    with open('%2', 'wb') as file:\n"
                "        shutil.copyfileobj(r.raw, file)\n")
            .arg(QString::fromStdString(url))
            .arg(QString::fromStdString(formattedTo));
    Utils::Python::runScript(script.toStdString());
}
