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

        CMat                                getMask() const;
        std::vector<std::string>            getClassNames() const;
        std::vector<cv::Vec3b>              getColors() const;
        std::shared_ptr<CImageIO>           getMaskImageIO() const;
        std::shared_ptr<CImageIO>           getLegendImageIO() const;

        void                                setMask(const CMat& mask);
        void                                setClassNames(const std::vector<std::string>& names, const std::vector<cv::Vec3b>& colors);

        bool                                isDataAvailable() const override;

        void                                clearData() override;

        void                                load(const std::string &path) override;
        void                                save(const std::string &path) override;

        std::string                         toJson(const std::vector<std::string>& options) const override;
        void                                fromJson(const std::string &jsonStr) override;

        std::shared_ptr<CSemanticSegIO>     clone() const;

    private:

        std::shared_ptr<CWorkflowTaskIO>    cloneImp() const override;

        QJsonObject                         toJsonInternal(const std::vector<std::string> &options) const;
        void                                fromJson(const QJsonDocument& doc);

        void                                generateLegend();

    private:

        std::vector<std::string>    m_classes;
        std::vector<cv::Vec3b>      m_colors;
        std::shared_ptr<CImageIO>   m_imgMaskIOPtr = nullptr;
        std::shared_ptr<CImageIO>   m_imgLegendIOPtr = nullptr;
};

#endif // CSEMANTICSEGIO_H
