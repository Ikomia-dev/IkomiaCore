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

        int                 getId() const;
        std::string         getLabel() const;
        double              getConfidence() const;
        std::vector<double> getBox() const;
        CColor              getColor() const;

        void                setId(int id);
        void                setLabel(const std::string& label);
        void                setConfidence(double confidence);
        void                setBox(const std::vector<double>& box);
        void                setColor(const CColor& color);

    public:

        int                 m_id;
        std::string         m_label = "";
        double              m_confidence = 0;
        std::vector<double> m_box;
        CColor              m_color = {255, 0, 0};
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

        std::string                         repr() const override;

        size_t                              getObjectCount() const;
        CObjectDetection                    getObject(size_t index) const;
        std::vector<CObjectDetection>       getObjects() const;
        CDataInfoPtr                        getDataInfo() override;
        GraphicsOutputPtr                   getGraphicsIO() const;
        BlobMeasureIOPtr                    getBlobMeasureIO() const;

        bool                                isDataAvailable() const override;
        bool                                isComposite() const override;

        void                                init(const std::string& taskName, int imageIndex);

        void                                addObject(int id, const std::string& label, double confidence,
                                                      double boxX, double boxY, double boxWidth, double boxHeight,
                                                      const CColor& color);

        void                                addObject(int id, const std::string& label, double confidence,
                                                      double cx, double cy, double width, double height,
                                                      double angle, const CColor& color);

        void                                clearData() override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson() const override;
        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CObjectDetectionIO> clone() const;

        void                                copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr) override;

    private:

        WorkflowTaskIOPtr                   cloneImp() const override;
        QJsonObject                         toJsonInternal() const;
        void                                fromJsonInternal(const QJsonDocument& doc);

    private:

        std::vector<CObjectDetection>   m_objects;
        GraphicsOutputPtr               m_graphicsIOPtr = nullptr;
        BlobMeasureIOPtr                m_blobMeasureIOPtr = nullptr;
};

using ObjectDetectionIOPtr = std::shared_ptr<CObjectDetectionIO>;

class DATAPROCESSSHARED_EXPORT CObjectDetectionIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CObjectDetectionIOFactory()
        {
            m_name = "CObjectDetectionIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CObjectDetectionIO>();
        }
};

#endif // COBJECTDETECTIONIO_H
