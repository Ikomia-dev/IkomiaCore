#include "CNetworkManager.h"
#include <QCoreApplication>
#include <QThread>
#include <QtNetwork/QNetworkAccessManager>
#include <QFile>
#include <QFileInfo>
#include "CException.h"

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
        delete m_pSocketThread;
        m_pSocketThread = nullptr;
    }
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
        QMetaObject::invokeMethod(this, "slotDownload", Qt::QueuedConnection,
                                  Q_ARG(void*, &m_loop),
                                  Q_ARG(QString, qurl.url()),
                                  Q_ARG(QString, pathTo));
        m_loop.exec();
    }
    else
    {
        // For non-GUI threads, QEventLoop would cause a deadlock, so we simply use a blocking call.
        // (Does not hurt as no messages need to be processed either during the open operation).
        QMetaObject::invokeMethod(this, "slotDownload", Qt::BlockingQueuedConnection,
                                  Q_ARG(void*, nullptr),
                                  Q_ARG(QString, qurl.url()),
                                  Q_ARG(QString, pathTo));
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
