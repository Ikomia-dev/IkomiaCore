#ifndef CKEYPTSIOWRAP_H
#define CKEYPTSIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CKeypointsIO.h"

class CKeyptsIOWrap: public CKeypointsIO, public wrapper<CKeypointsIO>
{
    public:

        CKeyptsIOWrap();
        CKeyptsIOWrap(const CKeypointsIO& io);

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

#endif // CKEYPTSIOWRAP_H
