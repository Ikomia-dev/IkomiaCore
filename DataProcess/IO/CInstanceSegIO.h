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

        int     getType() const;
        int     getClassIndex() const;
        CMat    getMask() const;

        void    setType(int type);
        void    setClassIndex(int index);
        void    setMask(const CMat& mask);

    public:

        int     m_type = ObjectType::THING;
        int     m_classIndex = 0;
        CMat    m_mask;
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

        size_t                              getInstanceCount() const;
        CInstanceSegmentation               getInstance(size_t index) const;
        std::vector<CInstanceSegmentation>  getInstances() const;
        CDataInfoPtr                        getDataInfo() override;
        std::shared_ptr<CImageIO>           getMaskImageIO() const;
        std::shared_ptr<CGraphicsOutput>    getGraphicsIO() const;
        std::shared_ptr<CBlobMeasureIO>     getBlobMeasureIO() const;
        CMat                                getMergeMask() const;

        bool                                isDataAvailable() const override;

        void                                init(const std::string& taskName, int refImageIndex, int imageWidth, int imageHeight);

        void                                addInstance(int id, int type, int classIndex, const std::string& label, double confidence,
                                                        double boxX, double boxY, double boxWidth, double boxHeight,
                                                        const CMat& mask, const CColor& color);

        void                                clearData() override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson() const override;
        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CInstanceSegIO>     clone() const;

    private:

        std::shared_ptr<CWorkflowTaskIO>    cloneImp() const override;
        QJsonObject                         toJsonInternal(const std::vector<std::string> &options) const;
        void                                fromJsonInternal(const QJsonDocument& doc);

    private:

        CMat                                m_mergeMask;
        std::vector<CInstanceSegmentation>  m_instances;
        std::shared_ptr<CImageIO>           m_imgIOPtr = nullptr;
        std::shared_ptr<CGraphicsOutput>    m_graphicsIOPtr = nullptr;
        std::shared_ptr<CBlobMeasureIO>     m_blobMeasureIOPtr = nullptr;
};

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
