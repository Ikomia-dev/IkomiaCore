#ifndef CGRAPHICSOUTPUTWRAP_H
#define CGRAPHICSOUTPUTWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CGraphicsOutput.h"

class CGraphicsOutputWrap: public CGraphicsOutput, public wrapper<CGraphicsOutput>
{
    public:

        CGraphicsOutputWrap();
        CGraphicsOutputWrap(const std::string& name);
        CGraphicsOutputWrap(const CGraphicsOutput &io);

        virtual bool    isDataAvailable() const;
        bool            default_isDataAvailable() const;

        virtual void    clearData();
        void            default_clearData();

        void            load(const std::string& path) override;
        void            default_load(const std::string& path);

        void            save(const std::string& path) override;
        void            default_save(const std::string& path);

        std::string     toJson() const override;
        std::string     default_toJsonNoOpt() const;

        std::string     toJson(const std::vector<std::string>& options) const override;
        std::string     default_toJson(const std::vector<std::string>& options) const;

        void            fromJson(const std::string& jsonStr) override;
        void            default_fromJson(const std::string& jsonStr);
};

#endif // CGRAPHICSOUTPUTWRAP_H
