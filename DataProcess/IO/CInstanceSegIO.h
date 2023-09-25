#ifndef CINSTANCESEGIO_H
#define CINSTANCESEGIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CObjectDetectionIO.h"
#include "IO/CImageIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CBlobMeasureIO.h"

//---------------------------------//
//----- CInstanceSegmentation -----//
//---------------------------------//
class DATAPROCESSSHARED_EXPORT CInstanceSegmentation : public CObjectDetection
{
    public:

        enum ObjectType
        {
            THING = 0,
            STUFF = 1
        };

        int         getType() const;
        int         getClassIndex() const;
        CMat        getMask() const;
        std::vector<ProxyGraphicsItemPtr>   getPolygons() const;

        void        setType(int type);
        void        setClassIndex(int index);
        void        setMask(const CMat& mask);

        void        computePolygons();

        std::string repr() const;

        friend DATAPROCESSSHARED_EXPORT std::ostream& operator<<(std::ostream& os, const CInstanceSegmentation& io);

    public:

        int                                 m_type = ObjectType::THING;
        int                                 m_classIndex = 0;
        CMat                                m_mask;
        std::vector<ProxyGraphicsItemPtr>   m_polygons;
};


//--------------------------//
//----- CInstanceSegIO -----//
//--------------------------//
class DATAPROCESSSHARED_EXPORT CInstanceSegIO: public CWorkflowTaskIO
{
    public:

        CInstanceSegIO();
        CInstanceSegIO(const CInstanceSegIO& io);
        CInstanceSegIO(const CInstanceSegIO&& io);

        ~CInstanceSegIO() = default;

        CInstanceSegIO& operator=(const CInstanceSegIO& io);
        CInstanceSegIO& operator=(const CInstanceSegIO&& io);

        std::string                         repr() const override;

        size_t                              getObjectCount() const;
        CInstanceSegmentation               getObject(size_t index) const;
        std::vector<CInstanceSegmentation>  getObjects() const;
        std::vector<std::string>            getClassNames() const;
        CDataInfoPtr                        getDataInfo() override;
        ImageIOPtr                          getMaskImageIO() const;
        GraphicsOutputPtr                   getGraphicsIO() const;
        BlobMeasureIOPtr                    getBlobMeasureIO() const;
        CMat                                getMergeMask() const;
        InputOutputVect                     getSubIOList(const std::set<IODataType> &dataTypes) const override;
        CMat                                getImageWithGraphics(const CMat &image) const override;
        CMat                                getImageWithMask(const CMat &image) const override;
        CMat                                getImageWithMaskAndGraphics(const CMat &image) const override;
        std::vector<CColor>                 getColors() const;

        bool                                isDataAvailable() const override;
        bool                                isComposite() const override;

        void                                init(const std::string& taskName, int refImageIndex, int imageWidth, int imageHeight);

        void                                addObject(int id, int type, int classIndex, const std::string& label, double confidence,
                                                      double boxX, double boxY, double boxWidth, double boxHeight,
                                                      const CMat& mask, const CColor &color);

        void                                clearData() override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson() const override;
        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CInstanceSegIO>     clone() const;

    private:

        WorkflowTaskIOPtr                   cloneImp() const override;
        QJsonObject                         toJsonInternal(const std::vector<std::string> &options) const;
        void                                fromJsonInternal(const QJsonDocument& doc);

    private:

        std::vector<CInstanceSegmentation>  m_instances;
        ImageIOPtr                          m_imgIOPtr = nullptr;
        GraphicsOutputPtr                   m_graphicsIOPtr = nullptr;
        BlobMeasureIOPtr                    m_blobMeasureIOPtr = nullptr;
};

using InstanceSegIOPtr = std::shared_ptr<CInstanceSegIO>;


class DATAPROCESSSHARED_EXPORT CInstanceSegIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CInstanceSegIOFactory()
        {
            m_name = "CInstanceSegIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CInstanceSegIO>();
        }
};

#endif // CINSTANCESEGIO_H
