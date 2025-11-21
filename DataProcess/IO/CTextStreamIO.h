#ifndef CTEXTSTREAMIO_H
#define CTEXTSTREAMIO_H

#include <boost/asio.hpp>
#include <boost/thread/future.hpp>
#include <deque>
#include <mutex>
#include <string>
#include <functional>
#include <chrono>

#include "DataProcessGlobal.hpp"
#include "Workflow/CWorkflowTaskIO.h"


class DATAPROCESSSHARED_EXPORT CTextStreamIO: public CWorkflowTaskIO
{
    public:

        using Handler = std::function<void(const std::string&, const boost::system::error_code&)>;

        explicit CTextStreamIO(boost::asio::io_context& io, int maxBufferSize = 1e8);
        CTextStreamIO(const CTextStreamIO&) = delete;

        ~CTextStreamIO() = default;

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
        void                        readNext(int minBytes, int timeout, Handler handler);
        std::future<std::string>    readNext(int minBytes, int timeout);
        std::string                 readFull() const;

        void                        close();

        void                        clearData() override;

        void                        load(const std::string &path) override;
        void                        save(const std::string &path) override;

        std::string                 toJson() const override;
        std::string                 toJson(const std::vector<std::string>& options) const override;
        void                        fromJson(const std::string &jsonStr) override;

    private:

        std::string                 extract(std::size_t n);
        void                        notifyWaiters();
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

        boost::asio::io_context&    m_io;
        std::size_t                 m_nextId = 0;
        int                         m_maxBufferSize;
        bool                        m_bFeedFinished = false;
        bool                        m_bReadFinished = false;
        boost::asio::steady_timer   m_timer;
        std::string                 m_buffer;
        std::string                 m_fullText;
        std::deque<WaitRequest>     m_waiters;
        std::mutex                  m_mutex;
};

#endif // CTEXTSTREAMIO_H
