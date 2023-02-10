#ifndef CSEMANTICSEGIOWRAP_H
#define CSEMANTICSEGIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CSemanticSegIO.h"

class CSemanticSegIOWrap: public CSemanticSegIO, public wrapper<CSemanticSegIO>
{
    public:

        CSemanticSegIOWrap();
        CSemanticSegIOWrap(const CSemanticSegIO& io);

        void        setClassColors(const std::vector<std::vector<uchar>>& colors);

        std::vector<std::vector<uchar>> getColorsWrap() const;

        bool        isDataAvailable() const override;
        bool        default_isDataAvailable() const;

        void        clearData() override;
        void        default_clearData();

        void        load(const std::string &path) override;
        void        default_load(const std::string &path);

        void        save(const std::string &path) override;
        void        default_save(const std::string &path);

        std::string toJson() const override;
        std::string default_toJsonNoOpt() const;

        std::string toJson(const std::vector<std::string>& options) const override;
        std::string default_toJson(const std::vector<std::string>& options) const;

        void        fromJson(const std::string& jsonStr) override;
        void        default_fromJson(const std::string& jsonStr);
};

#endif // CSEMANTICSEGIOWRAP_H
