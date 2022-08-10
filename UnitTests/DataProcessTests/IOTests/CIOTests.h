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
        void    blobMeasureIOToJson();
        void    blobMeasureIOFromJson();

        void    graphicsInputSave();
        void    graphicsInputLoad();
        void    graphicsInputToJson();
        void    graphicsInputFromJson();

        void    graphicsOutputSave();
        void    graphicsOutputLoad();
        void    graphicsOutputToJson();
        void    graphicsOutputFromJson();

        void    numericIODoubleSave();
        void    numericIODoubleLoad();
        void    numericIODoubleToJson();
        void    numericIODoubleFromJson();

        void    numericIOStringSave();
        void    numericIOStringLoad();
        void    numericIOStringToJson();
        void    numericIOStringFromJson();

        void    imageIOToJson();
        void    imageIOFromJson();

    private:

        void    fillBlobMeasureIO(CBlobMeasureIO& io);
        void    fillNumericIO(CNumericIO<double>& io);
        void    fillNumericIO(CNumericIO<std::string>& io);

        std::vector<ProxyGraphicsItemPtr> createGraphics();
};

#endif // CIOTESTS_H
