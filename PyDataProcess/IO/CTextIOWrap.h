#ifndef CTEXTIOWRAP_H
#define CTEXTIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CTextIO.h"

class CTextIOWrap: public CTextIO, public wrapper<CTextIO>
{
    public:

        CTextIOWrap();
        CTextIOWrap(const CTextIO& io);

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

#endif // CTEXTIOWRAP_H
