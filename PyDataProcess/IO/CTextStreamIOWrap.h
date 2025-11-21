#ifndef CTEXTSTREAMIOWRAP_H
#define CTEXTSTREAMIOWRAP_H

#include <boost/asio.hpp>
#include "PyDataProcessGlobal.h"
#include "Workflow/CWorkflowTaskIO.h"
#include "IO/CTextStreamIO.h"


//--------------------------------------------------------
//- Python wrapper to handle io_context in its own thread
//--------------------------------------------------------
class CPyTextStreamIO: public CWorkflowTaskIO
{
    public:

        CPyTextStreamIO(int maxBufferSize=1e8);
        ~CPyTextStreamIO();

        CTextStreamIO& stream();
        const CTextStreamIO& stream() const;

    private:

        boost::asio::io_context m_io;
        boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_workGuard;
        CTextStreamIO           m_stream;
        std::thread             m_thread;
};


//------------------------------------------------
//- Python binding wrapper to handle polymorphism
//------------------------------------------------
class CTextStreamIOWrap: public CPyTextStreamIO, public wrapper<CPyTextStreamIO>, public std::enable_shared_from_this<CTextStreamIOWrap>
{
    public:

        CTextStreamIOWrap();
        CTextStreamIOWrap(int maxBufferSize);

        std::string repr() const override;

        bool        isDataAvailable() const override;
        bool        default_isDataAvailable() const;

        bool        isFeedFinished() const;
        bool        isReadFinished() const;

        void        feed(const std::string& chunk);

        void        readNext(int minBytes, int timeout, api::object py_callback);
        std::string readFull();
        
        void        close();

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

#endif // CTEXTSTREAMIOWRAP_H
