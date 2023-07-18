#ifndef CSEMANTICSEGIO_H
#define CSEMANTICSEGIO_H

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CImageIO.h"


class DATAPROCESSSHARED_EXPORT CSemanticSegIO: public CWorkflowTaskIO
{
    public:

        CSemanticSegIO();
        CSemanticSegIO(const CSemanticSegIO& io);
        CSemanticSegIO(const CSemanticSegIO&& io);

        ~CSemanticSegIO() = default;

        CSemanticSegIO& operator=(const CSemanticSegIO& io);
        CSemanticSegIO& operator=(const CSemanticSegIO&& io);

        std::string                     repr() const override;

        CMat                            getMask() const;
        CMat                            getLegend() const;
        std::vector<std::string>        getClassNames() const;
        std::vector<CColor>             getColors() const;
        ImageIOPtr                      getMaskImageIO() const;
        ImageIOPtr                      getLegendImageIO() const;
        InputOutputVect                 getSubIOList(const std::set<IODataType> &dataTypes) const override;
        int                             getReferenceImageIndex() const;

        void                            setMask(const CMat& mask);
        void                            setClassNames(const std::vector<std::string>& names);
        void                            setClassColors(const std::vector<CColor>& colors);
        void                            setReferenceImageIndex(int index);

        bool                            isDataAvailable() const override;
        bool                            isComposite() const override;

        void                            clearData() override;

        void                            load(const std::string &path) override;
        void                            save(const std::string &path) override;

        std::string                     toJson() const override;
        std::string                     toJson(const std::vector<std::string>& options) const override;
        void                            fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CSemanticSegIO> clone() const;

        void                            copy(const std::shared_ptr<CWorkflowTaskIO> &ioPtr) override;

    private:

        WorkflowTaskIOPtr               cloneImp() const override;

        QJsonObject                     toJsonInternal(const std::vector<std::string> &options) const;
        void                            fromJsonInternal(const QJsonDocument& doc);

        void                            generateLegend();
        void                            generateRandomColors();

    private:

        std::vector<std::string>    m_classes;
        std::vector<CColor>         m_colors;
        cv::Mat                     m_histo;
        std::shared_ptr<CImageIO>   m_imgMaskIOPtr = nullptr;
        std::shared_ptr<CImageIO>   m_imgLegendIOPtr = nullptr;
        int                         m_refImageIndex = 0;
};

using SemanticSegIOPtr = std::shared_ptr<CSemanticSegIO>;


class DATAPROCESSSHARED_EXPORT CSemanticSegIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CSemanticSegIOFactory()
        {
            m_name = "CSemanticSegIO";
        }

        virtual WorkflowTaskIOPtr   create(IODataType dataType)
        {
            Q_UNUSED(dataType);
            return std::make_shared<CSemanticSegIO>();
        }
};

#endif // CSEMANTICSEGIO_H
