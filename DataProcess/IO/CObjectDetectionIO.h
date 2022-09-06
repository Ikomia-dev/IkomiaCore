#ifndef COBJECTDETECTIONIO_H
#define COBJECTDETECTIONIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CBlobMeasureIO.h"
#include "Main/CoreDefine.hpp"

//----------------------------//
//----- CObjectDetection -----//
//----------------------------//
class DATAPROCESSSHARED_EXPORT CObjectDetection
{
    public:

        std::string         getLabel() const;
        double              getConfidence() const;
        std::vector<double> getBox() const;
        CColor              getColor() const;

        void                setLabel(const std::string& label);
        void                setConfidence(double confidence);
        void                setBox(const std::vector<double>& box);
        void                setColor(const CColor& color);

    public:

        std::string         m_label = "";
        double              m_confidence = 0;
        std::vector<double> m_box;
        CColor              m_color = {255,0,0};
};


//------------------------------//
//----- CObjectDetectionIO -----//
//------------------------------//
class DATAPROCESSSHARED_EXPORT CObjectDetectionIO: public CWorkflowTaskIO
{
    public:

        CObjectDetectionIO();
        CObjectDetectionIO(const CObjectDetectionIO& io);
        CObjectDetectionIO(const CObjectDetectionIO&& io);

        ~CObjectDetectionIO() = default;

        CObjectDetectionIO& operator=(const CObjectDetectionIO& io);
        CObjectDetectionIO& operator=(const CObjectDetectionIO&& io);

        size_t                              getObjectCount() const;
        CObjectDetection                    getObject(size_t index) const;
        std::vector<CObjectDetection>       getObjects() const;
        CDataInfoPtr                        getDataInfo() override;
        std::shared_ptr<CGraphicsOutput>    getGraphicsIO() const;
        std::shared_ptr<CBlobMeasureIO>     getBlobMeasureIO() const;

        bool                                isDataAvailable() const override;

        void                                init(const std::string& taskName, int imageIndex);

        void                                addObject(const std::string& label, double confidence,
                                                      double boxX, double boxY, double boxWidth, double boxHeight,
                                                      const CColor& color);

        void                                clearData() override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CObjectDetectionIO> clone() const;

    private:

        std::shared_ptr<CWorkflowTaskIO>    cloneImp() const override;
        QJsonObject                         toJson() const;
        void                                fromJson(const QJsonDocument& doc);

    private:

        std::vector<CObjectDetection>       m_objects;
        std::shared_ptr<CGraphicsOutput>    m_graphicsIOPtr = nullptr;
        std::shared_ptr<CBlobMeasureIO>     m_blobMeasureIOPtr = nullptr;
};

#endif // COBJECTDETECTIONIO_H
