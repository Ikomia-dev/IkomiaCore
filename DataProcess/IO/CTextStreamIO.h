#ifndef CTEXTSTREAMIO_H
#define CTEXTSTREAMIO_H

#include <boost/asio.hpp>
#include <boost/thread/future.hpp>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <functional>
#include <chrono>
#include <condition_variable>
#include <thread>

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"


class DATAPROCESSSHARED_EXPORT CTextStreamIO: public CWorkflowTaskIO
{
    public:

        using Handler = std::function<void(const std::string&, const boost::system::error_code&)>;

        explicit CTextStreamIO(int maxBufferSize = 1e8);
        explicit CTextStreamIO(boost::asio::io_context& io, int maxBufferSize = 1e8);
        CTextStreamIO(const CTextStreamIO&) = delete;

        ~CTextStreamIO();

        CTextStreamIO& operator=(const CTextStreamIO& io) = delete;

        std::string                 repr() const override;

        bool                        isDataAvailable() const override;
        bool                        isFeedFinished() const;
        bool                        isReadFinished() const;

        // -------------------------------------------------------
        // Feed incoming text chunks (like network input)
        // -------------------------------------------------------
        void                        feed(const std::string& chunk);

        // -------------------------------------------------------
        // Async chunk read with optional timeout
        // -------------------------------------------------------
        void                        readNextAsync(int minBytes, float timeout, Handler handler);
        std::future<std::string>    readNextAsync(int minBytes, float timeout);
        void                        readFullAsync(float timeout, Handler handler);
        std::future<std::string>    readFullAsync(float timeout);
        std::string                 readFull();

        void                        close();

        void                        shutdown();

        void                        clearData() override;

        void                        load(const std::string &path) override;
        void                        save(const std::string &path) override;

        std::string                 toJson() const override;
        std::string                 toJson(const std::vector<std::string>& options) const override;
        void                        fromJson(const std::string &jsonStr) override;

    private:

        std::string                 extract(std::size_t n);
        void                        notifyWaiters();
        void                        notifyWaitersFull();
        void                        sendData(Handler handler, const std::string &data, const boost::system::error_code &err);
        QJsonObject                 toJsonInternal() const;
        void                        fromJsonInternal(const QJsonDocument& doc);

    private:

        struct WaitRequest
        {
            std::size_t id;
            int         minBytes;
            Handler     handler;
            std::shared_ptr<boost::asio::steady_timer> timer; // optional timeout
        };

        // Owned io_context (only when constructed without an external one)
        std::optional<boost::asio::io_context>                                                    m_ownedIo;
        std::optional<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>>   m_workGuard;
        std::optional<std::thread>                                                                m_ioThread;
        boost::asio::io_context*    m_ioPtr;    // always valid; points to m_ownedIo or external io

        std::size_t                 m_nextId = 0;
        int                         m_maxBufferSize;
        std::atomic_bool            m_bFeedFinished{false};
        std::atomic_bool            m_bReadFinished{false};
        boost::asio::steady_timer   m_timer;
        std::string                 m_buffer;
        std::string                 m_fullText;
        std::deque<WaitRequest>     m_waiters;
        std::vector<WaitRequest>    m_waitersFull;
        std::mutex                  m_mutex;
        std::condition_variable     m_feedFinishedCv;
};


class DATAPROCESSSHARED_EXPORT CTextStreamIOFactory: public CWorkflowTaskIOFactory
{
    public:

        CTextStreamIOFactory()
        {
            m_name = "CTextStreamIO";
        }

        WorkflowTaskIOPtr   create(IODataType dataType) override
        {
            Q_UNUSED(dataType);
            return std::make_shared<CTextStreamIO>();
        }

        std::vector<IODataType> getValidDataTypes() const override
        {
            return { IODataType::TEXT_STREAM };
        }
};

#endif // CTEXTSTREAMIO_H
