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

        bool                    isFeedFinished() const;
        bool                    isReadFinished() const;

        void                    feed(const std::string& chunk);

        CTextStreamIO&          stream();
        const CTextStreamIO&    stream() const;

        std::string             readNext(float timeout);
        std::string             readFull();

        void                    close();

        void                    shutdown();

    protected:

        void                    setMinBytes(int minBytes);

    private:

        void                    queueProcessingLoop();

    private:

        boost::asio::io_context m_io;
        boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_workGuard;
        CTextStreamIO           m_stream;
        std::thread             m_ioThread;
        std::thread             m_queueThread;
        std::mutex              m_queueMutex;
        std::deque<std::string> m_chunkQueue;
        std::condition_variable m_queueCv;
        std::atomic_bool        m_bStopQueueThread = false;
        std::atomic_int         m_minBytes = 1;
};


//------------------------------------------------
//- Python binding wrapper to handle polymorphism
//------------------------------------------------
class CTextStreamIOWrap: public CPyTextStreamIO, public wrapper<CPyTextStreamIO>
{
    public:

        CTextStreamIOWrap();
        CTextStreamIOWrap(int maxBufferSize);
        ~CTextStreamIOWrap() = default;

        std::string repr() const override;

        bool        isDataAvailable() const override;
        bool        default_isDataAvailable() const;

        std::string readNext(int minBytes, float timeout);

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

