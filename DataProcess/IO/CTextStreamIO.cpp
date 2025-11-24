#include "CTextStreamIO.h"
#include <thread>
#include "Main/CoreTools.hpp"


CTextStreamIO::CTextStreamIO(boost::asio::io_context &io, int maxBufferSize)
    : m_io(io), m_maxBufferSize(maxBufferSize), m_timer(io)
{
    m_description = QObject::tr("Text I/O with streaming capabilities (in and out).").toStdString();
    m_saveFormat = DataFileFormat::JSON;
}

std::string CTextStreamIO::repr() const
{
    std::stringstream s;
    s << "CTextStreamIO()";
    return s.str();
}

bool CTextStreamIO::isDataAvailable() const
{
    return !m_fullText.empty();
}

bool CTextStreamIO::isFeedFinished() const
{
    return m_bFeedFinished;
}

bool CTextStreamIO::isReadFinished() const
{
    return m_bReadFinished;
}

// -------------------------------------------------------
// Feed incoming text chunks (like network input)
// -------------------------------------------------------
void CTextStreamIO::feed(const std::string& chunk)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_bFeedFinished = false;
    m_bReadFinished = false;

    if (m_buffer.size() + chunk.size() > m_maxBufferSize)
    {
        // drop extra data to enforce max buffer
        std::size_t allowed = m_maxBufferSize - m_buffer.size();
        std::string truncated_chunk = chunk.substr(0, allowed);
        m_buffer += truncated_chunk;
        m_fullText += truncated_chunk;
    }
    else
    {
        m_buffer += chunk;
        m_fullText += chunk;
    }
    notifyWaiters();
}

// -------------------------------------------------------
// Async chunk read with optional timeout
// -------------------------------------------------------
void CTextStreamIO::readNextAsync(int minBytes, int timeout, Handler handler)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto timeoutChrono = std::chrono::seconds(timeout);

    // Immediate completion if enough data
    if (m_buffer.size() >= minBytes || m_bFeedFinished)
    {
        boost::system::error_code ec(static_cast<int>(boost::system::errc::success), boost::system::generic_category());
        sendData(handler, extract(m_buffer.size()), ec);
        return;
    }

    // Otherwise, enqueue a pending read
    WaitRequest w;
    w.id = m_nextId++;
    w.minBytes = minBytes;
    w.handler = handler;

    if (timeoutChrono.count() > 0)
    {
        w.timer = std::make_shared<boost::asio::steady_timer>(m_io);
        w.timer->expires_after(timeoutChrono);
        w.timer->async_wait([this, w](auto ec){
            if (ec == boost::asio::error::operation_aborted)
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            // remove from waiters if still pending
            auto it = std::find_if(m_waiters.begin(), m_waiters.end(), [w](auto& wr){
                return wr.id == w.id;
            });

            if (it != m_waiters.end())
            {
                m_waiters.erase(it);
                boost::system::error_code ec(static_cast<int>(boost::system::errc::timed_out), boost::system::generic_category());
                sendData(w.handler, "", ec);
            }
        });
    }
    m_waiters.push_back(std::move(w));
}

// Future-based version
std::future<std::string> CTextStreamIO::readNextAsync(int n, int timeout)
{
    auto promise = std::make_shared<std::promise<std::string>>();

    readNextAsync(n, timeout, [this, promise](const std::string& text, const boost::system::error_code& ec){
        if (ec == boost::system::errc::timed_out)
            promise->set_exception(std::make_exception_ptr(std::runtime_error("timeout")));
        else
            promise->set_value(text);
    });

    return promise->get_future();
}

void CTextStreamIO::readFullAsync(int timeout, Handler handler)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto timeoutChrono = std::chrono::seconds(timeout);

    // Immediate completion if feed operation is finished
    if (m_bFeedFinished)
    {
        boost::system::error_code ec(static_cast<int>(boost::system::errc::success), boost::system::generic_category());
        m_buffer.clear();
        sendData(handler, m_fullText, ec);
        return;
    }

    // Otherwise, enqueue a pending read
    WaitRequest w;
    w.id = m_nextId++;
    w.handler = handler;

    if (timeoutChrono.count() > 0)
    {
        w.timer = std::make_shared<boost::asio::steady_timer>(m_io);
        w.timer->expires_after(timeoutChrono);
        w.timer->async_wait([this, w](auto ec){
            if (ec == boost::asio::error::operation_aborted)
                return;

            std::lock_guard<std::mutex> lock(m_mutex);
            // remove from waiters if still pending
            auto it = std::find_if(m_waiters.begin(), m_waiters.end(), [w](auto& wr){
                return wr.id == w.id;
            });

            if (it != m_waiters.end())
            {
                m_waiters.erase(it);
                boost::system::error_code ec(static_cast<int>(boost::system::errc::timed_out), boost::system::generic_category());
                sendData(w.handler, "", ec);
            }
        });
    }
    m_waitersFull.push_back(std::move(w));
}

std::future<std::string> CTextStreamIO::readFullAsync(int timeout)
{
    auto promise = std::make_shared<std::promise<std::string>>();

    readFullAsync(timeout, [this, promise](const std::string& text, const boost::system::error_code& ec){
        if (ec == boost::system::errc::timed_out)
            promise->set_exception(std::make_exception_ptr(std::runtime_error("timeout")));
        else
            promise->set_value(text);
    });

    return promise->get_future();
}

std::string CTextStreamIO::readFull() const
{
    return m_fullText;
}

void CTextStreamIO::close()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_bFeedFinished = true;
    notifyWaiters();
    notifyWaitersFull();
}

void CTextStreamIO::clearData()
{
    m_waiters.clear();
    m_waitersFull.clear();
    m_buffer.clear();
    m_fullText.clear();
}

std::string CTextStreamIO::extract(std::size_t n)
{
    if (m_buffer.empty())
        return "";

    std::size_t take = std::min(n, m_buffer.size());
    std::string out = m_buffer.substr(0, take);
    m_buffer.erase(0, n);
    return out;
}

void CTextStreamIO::notifyWaiters()
{
    for (;;)
    {
        if (m_waiters.empty())
            return;

        auto& w = m_waiters.front();
        if (m_buffer.size() < w.minBytes && m_bFeedFinished == false)
            return;

        // Cancel timeout timer if exists
        if (w.timer)
            w.timer->cancel();

        auto chunk = extract(m_buffer.size());
        m_waiters.pop_front();
        boost::system::error_code ec(static_cast<int>(boost::system::errc::success), boost::system::generic_category());
        sendData(w.handler, chunk, ec);
    }
}

void CTextStreamIO::notifyWaitersFull()
{
    for (size_t i=0; i<m_waitersFull.size(); ++i)
    {
        auto& w = m_waitersFull[i];

        // Cancel timeout timer if exists
        if (w.timer)
            w.timer->cancel();

        boost::system::error_code ec(static_cast<int>(boost::system::errc::success), boost::system::generic_category());
        sendData(w.handler, m_fullText, ec);
    }
    m_waitersFull.clear();
}

void CTextStreamIO::sendData(Handler handler, const std::string& data, const boost::system::error_code& ec)
{
    boost::asio::post(m_io, [this, handler, data, ec](){
        handler(data, ec);

        if (m_buffer.empty() && m_bFeedFinished)
            m_bReadFinished = true;
    });
}

void CTextStreamIO::load(const std::string &path)
{
    auto extension = Utils::File::extension(path);
    if (extension != ".json")
        throw CException(CoreExCode::NOT_IMPLEMENTED, "File format not available yet, please use .json files.", __func__, __FILE__, __LINE__);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::ReadOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't read file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonFile.readAll()));
    if(jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading text: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CTextStreamIO::save(const std::string &path)
{
    CWorkflowTaskIO::save(path);

    QFile jsonFile(QString::fromStdString(path));
    if(!jsonFile.open(QFile::WriteOnly | QFile::Text))
        throw CException(CoreExCode::INVALID_FILE, "Couldn't write file:" + path, __func__, __FILE__, __LINE__);

    QJsonDocument jsonDoc(toJsonInternal());
    jsonFile.write(jsonDoc.toJson(QJsonDocument::Compact));
}

std::string CTextStreamIO::toJson() const
{
    std::vector<std::string> options = {"json_format", "compact"};
    return toJson(options);
}

std::string CTextStreamIO::toJson(const std::vector<std::string> &options) const
{
    QJsonDocument doc(toJsonInternal());
    return toFormattedJson(doc, options);
}

QJsonObject CTextStreamIO::toJsonInternal() const
{
    QJsonObject root;
    root["text"] =  QString::fromStdString(m_fullText);
    return root;
}

void CTextStreamIO::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading text detections: invalid JSON structure", __func__, __FILE__, __LINE__);

    fromJsonInternal(jsonDoc);
}

void CTextStreamIO::fromJsonInternal(const QJsonDocument &doc)
{
    clearData();
    QJsonObject root = doc.object();
    m_fullText = root["text"].toString().toStdString();
}
