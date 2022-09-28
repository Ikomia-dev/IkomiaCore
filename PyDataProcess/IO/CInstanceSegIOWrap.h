#ifndef CINSTANCESEGIOWRAP_H
#define CINSTANCESEGIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CInstanceSegIO.h"

class CInstanceSegIOWrap: public CInstanceSegIO, public wrapper<CInstanceSegIO>
{
    public:

        CInstanceSegIOWrap();
        CInstanceSegIOWrap(const CInstanceSegIO& io);

        bool        isDataAvailable() const override;
        bool        default_isDataAvailable() const;

        void        clearData() override;
        void        default_clearData();

        void        load(const std::string &path) override;
        void        default_load(const std::string &path);

        void        save(const std::string &path) override;
        void        default_save(const std::string &path);

        std::string toJson() const;
        std::string toJson(const std::vector<std::string>& options) const override;
        std::string default_toJson(const std::vector<std::string>& options) const;

        void        fromJson(const std::string& jsonStr) override;
        void        default_fromJson(const std::string& jsonStr);
};

#endif // CINSTANCESEGIOWRAP_H
