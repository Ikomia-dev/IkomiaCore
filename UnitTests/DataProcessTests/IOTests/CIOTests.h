#ifndef CIOTESTS_H
#define CIOTESTS_H

#include <QObject>
#include "DataProcess.hpp"

class CIOTests: public QObject
{
    Q_OBJECT

    public:

        CIOTests(QObject* parent=nullptr);

    private slots:

        void    initTestCase();

        void    blobMeasureIOSave();
        void    blobMeasureIOLoad();

    private:

        void    fillBlobMeasureIO(CBlobMeasureIO& io);
};

#endif // CIOTESTS_H
