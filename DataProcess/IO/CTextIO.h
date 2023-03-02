#ifndef CTEXTIO_H
#define CTEXTIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CNumericIO.h"

//----------------------//
//----- CTextField -----//
//----------------------//
class DATAPROCESSSHARED_EXPORT CTextField
{
    public:

        int                 getId() const;
        std::string         getLabel() const;
        std::string         getText() const;
        double              getConfidence() const;
        PolygonF            getPolygon() const;
        CColor              getColor() const;

        void                setId(int id);
        void                setLabel(const std::string& label);
        void                setText(const std::string& text);
        void                setConfidence(double confidence);
        void                setPolygon(const PolygonF& poly);
        void                setColor(const CColor& color);

        QJsonObject         toJson() const;

    public:

        int         m_id;
        std::string m_label = "";
        std::string m_text = "";
        double      m_confidence = 0;
        PolygonF    m_polygon;
        CColor      m_color = {255, 0, 0};
};

//-------------------//
//----- CTextIO -----//
//-------------------//
class DATAPROCESSSHARED_EXPORT CTextIO: public CWorkflowTaskIO
{
    using DataStringIOPtr = std::shared_ptr<CNumericIO<std::string>>;

    public:

        CTextIO();
        CTextIO(const CTextIO& io);
        CTextIO(const CTextIO&& io);

        ~CTextIO() = default;

        CTextIO& operator=(const CTextIO& io);
        CTextIO& operator=(const CTextIO&& io);

        size_t                      getTextFieldCount() const;
        CTextField                  getTextField(size_t index) const;
        std::vector<CTextField>     getTextFields() const;
        CDataInfoPtr                getDataInfo() override;
        GraphicsOutputPtr           getGraphicsIO() const;
        DataStringIOPtr             getDataStringIO() const;

        bool                        isDataAvailable() const override;
        bool                        isComposite() const override;

        void                        init(const std::string& taskName, int imageIndex);
        void                        finalize();

        void                        addTextField(int id, const std::string& label, const std::string& text,
                                                 double confidence, double x, double y, double width, double height,
                                                 const CColor& color);

        void                        addTextField(int id, const std::string& label, const std::string& text,
                                                 double confidence, const PolygonF& polygon, const CColor& color);

        void                        clearData() override;

        void                        load(const std::string &path) override;
        void                        save(const std::string &path) override;

        std::string                 toJson() const override;
        std::string                 toJson(const std::vector<std::string>& options) const override;
        void                        fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CTextIO>    clone() const;

    private:

        WorkflowTaskIOPtr           cloneImp() const override;
        QJsonObject                 toJsonInternal() const;
        void                        fromJsonInternal(const QJsonDocument& doc);

    private:

        std::vector<CTextField>     m_fields;
        GraphicsOutputPtr           m_graphicsIOPtr = nullptr;
        DataStringIOPtr             m_textDataIOPtr = nullptr;
};

#endif // CTEXTIO_H
