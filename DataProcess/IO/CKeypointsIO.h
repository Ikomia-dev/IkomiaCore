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
class CObjectKeypoints
{
    public:

        QJsonObject toJson() const;

    public:

        int                     m_id = 0;
        std::string             m_label;
        double                  m_confidence = 0.0;
        std::vector<double>     m_box;
        CColor                  m_color;
        std::vector<CPointF>    m_pts;
};

//-------------------------//
//----- CKeypointLink -----//
//-------------------------//
class CKeypointLink
{
    public:

        QJsonObject toJson() const;
        void        fromJson(const QJsonObject &jsonObj);
    public:

        int         m_ptIndex1 = -1;
        int         m_ptIndex2 = -1;
        std::string m_label;
        CColor      m_color;
};

//------------------------//
//----- CKeypointsIO -----//
//------------------------//
class CKeypointsIO: public CWorkflowTaskIO
{
    using DataStringIOPtr = std::shared_ptr<CNumericIO<std::string>>;

    public:

        CKeypointsIO();
        CKeypointsIO(const CKeypointsIO& io);
        CKeypointsIO(const CKeypointsIO&& io);

        ~CKeypointsIO() = default;

        CKeypointsIO& operator=(const CKeypointsIO& io);
        CKeypointsIO& operator=(const CKeypointsIO&& io);

        void                                addObject(int id, const std::string& label, double confidence,
                                                      double x, double y, double width, double height,
                                                      const std::vector<CPointF> keypts, CColor color);

        void                                clearData() override;

        std::shared_ptr<CKeypointsIO>       clone() const;

        std::shared_ptr<CBlobMeasureIO>     getBlobMeasureIO() const;
        CDataInfoPtr                        getDataInfo() override;
        DataStringIOPtr                     getDataStringIO() const;
        std::shared_ptr<CGraphicsOutput>    getGraphicsIO() const;
        size_t                              getObjectCount() const;
        CObjectKeypoints                    getObject(size_t index) const;
        std::vector<CObjectKeypoints>       getObjects() const;
        std::vector<CKeypointLink>          getKeypointLinks() const;
        std::vector<std::string>            getKeypointNames() const;

        bool                                isComposite() const override;
        bool                                isDataAvailable() const override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson() const override;
        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        void                                setKeypointNames(const std::vector<std::string>& names);
        void                                setKeypointLinks(const std::vector<CKeypointLink>& links);

    private:

        std::shared_ptr<CWorkflowTaskIO>    cloneImp() const override;
        QJsonObject                         toJsonInternal() const;
        void                                fromJsonInternal(const QJsonDocument& doc);


    private:

        std::vector<CObjectKeypoints>       m_objects;
        std::vector<std::string>            m_keyptsNames;
        std::vector<CKeypointLink>          m_links;

        std::shared_ptr<CGraphicsOutput>    m_graphicsIOPtr = nullptr;
        std::shared_ptr<CBlobMeasureIO>     m_objMeasureIOPtr = nullptr;
        DataStringIOPtr                     m_keyptsLinkIOPtr = nullptr;
};

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
