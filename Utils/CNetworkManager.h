#ifndef CNETWORKMANAGER_H
#define CNETWORKMANAGER_H

#include <string>
#include <QObject>
#include <QEventLoop>
#include <QtNetwork/QNetworkReply>
#include "UtilsGlobal.hpp"

class UTILSSHARED_EXPORT CNetworkManager: public QObject
{
    Q_OBJECT

    public:

        CNetworkManager(QObject* parent=nullptr);
        ~CNetworkManager();

        void    download(const std::string& url, const std::string& to);

    public slots:

        void    slotDownload(void* pLoop, QString url, QString to);

    private:

        bool    isGuiThread() const;
        void    workerDownload(const QString& url, const QString& to);

    private:

        QEventLoop  m_loop;
        QThread*    m_pSocketThread = nullptr;
};

#endif // CNETWORKMANAGER_H
