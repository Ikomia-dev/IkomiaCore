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

        void    graphicsInputSave();
        void    graphicsInputLoad();

        void    graphicsOutputSave();
        void    graphicsOutputLoad();

        void    numericIODoubleSave();
        void    numericIODoubleLoad();

        void    numericIOStringSave();
        void    numericIOStringLoad();

    private:

        void    fillBlobMeasureIO(CBlobMeasureIO& io);
        void    fillNumericIO(CNumericIO<double>& io);
        void    fillNumericIO(CNumericIO<std::string>& io);

        std::vector<ProxyGraphicsItemPtr> createGraphics();
};

#endif // CIOTESTS_H
