#ifndef CKEYPOINTSIO_H
#define CKEYPOINTSIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CBlobMeasureIO.h"
#include "IO/CNumericIO.h"

//----------------------------//
//----- CObjectKeypoints -----//
//----------------------------//
class DATAPROCESSSHARED_EXPORT CObjectKeypoints
{
    public:

        int                     getId() const;
        std::string             getLabel() const;
        double                  getConfidence() const;
        std::vector<double>     getBox() const;
        CColor                  getColor() const;
        std::vector<Keypoint>   getKeypoints() const;
        CPointF                 getKeypoint(int index) const;

        void                    setId(int id);
        void                    setLabel(const std::string& label);
        void                    setConfidence(double conf);
        void                    setBox(const std::vector<double>& box);
        void                    setColor(const CColor& color);
        void                    setKeypoints(const std::vector<Keypoint>& pts);

        QJsonObject             toJson() const;

    public:

        int                     m_id = 0;
        std::string             m_label;
        double                  m_confidence = 0.0;
        std::vector<double>     m_box;
        CColor                  m_color;
        std::vector<Keypoint>   m_keypts;
};

//-------------------------//
//----- CKeypointLink -----//
//-------------------------//
class DATAPROCESSSHARED_EXPORT CKeypointLink
{
    public:

        int         getStartPointIndex() const;
        int         getEndPointIndex() const;
        std::string getLabel() const;
        CColor      getColor() const;

        void        setStartPointIndex(int index);
        void        setEndPointIndex(int index);
        void        setLabel(const std::string& label);
        void        setColor(const CColor& color);

        void        fromJson(const QJsonObject &jsonObj);
        QJsonObject toJson() const;

    public:

        int         m_ptIndex1 = -1;
        int         m_ptIndex2 = -1;
        std::string m_label;
        CColor      m_color;
};

//------------------------//
//----- CKeypointsIO -----//
//------------------------//
class DATAPROCESSSHARED_EXPORT CKeypointsIO: public CWorkflowTaskIO
{
    using DataStringIOPtr = std::shared_ptr<CNumericIO<std::string>>;

    public:

        CKeypointsIO();
        CKeypointsIO(const CKeypointsIO& io);
        CKeypointsIO(const CKeypointsIO&& io);

        ~CKeypointsIO() = default;

        CKeypointsIO& operator=(const CKeypointsIO& io);
        CKeypointsIO& operator=(const CKeypointsIO&& io);

        std::string                     repr() const override;

        void                            addObject(int id, const std::string& label, double confidence,
                                                  double x, double y, double width, double height,
                                                  const std::vector<Keypoint> keypts, CColor color);

        void                            clearData() override;

        std::shared_ptr<CKeypointsIO>   clone() const;

        BlobMeasureIOPtr                getBlobMeasureIO() const;
        CDataInfoPtr                    getDataInfo() override;
        DataStringIOPtr                 getDataStringIO() const;
        GraphicsOutputPtr               getGraphicsIO() const;
        size_t                          getObjectCount() const;
        CObjectKeypoints                getObject(size_t index) const;
        std::vector<CObjectKeypoints>   getObjects() const;
        std::vector<CKeypointLink>      getKeypointLinks() const;
        std::vector<std::string>        getKeypointNames() const;

        void                            init(const std::string& taskName, int imageIndex);

        bool                            isComposite() const override;
        bool                            isDataAvailable() const override;

        void                            load(const std::string &path) override;
        void                            save(const std::string &path) override;

        std::string                     toJson() const override;
        std::string                     toJson(const std::vector<std::string>& options) const override;
        void                            fromJson(const std::string &jsonStr) override;

        void                            setKeypointNames(const std::vector<std::string>& names);
        void                            setKeypointLinks(const std::vector<CKeypointLink>& links);

    private:

        WorkflowTaskIOPtr               cloneImp() const override;
        QJsonObject                     toJsonInternal() const;
        void                            fromJsonInternal(const QJsonDocument& doc);


    private:

        std::vector<CObjectKeypoints>   m_objects;
        std::vector<std::string>        m_keyptsNames;
        std::vector<CKeypointLink>      m_links;

        GraphicsOutputPtr               m_graphicsIOPtr = nullptr;
        BlobMeasureIOPtr                m_objMeasureIOPtr = nullptr;
        DataStringIOPtr                 m_keyptsLinkIOPtr = nullptr;
};

using KeypointsIOPtr = std::shared_ptr<CKeypointsIO>;


class DATAPROCESSSHARED_EXPORT CKeypointsIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CKeypointsIOFactory()
        {
            m_name = "CKeypointsIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CKeypointsIO>();
        }
};

#endif // CKEYPOINTSIO_H
