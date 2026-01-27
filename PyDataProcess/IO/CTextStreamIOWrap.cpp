#include "CTextStreamIOWrap.h"
#include <atomic>
#include <thread>
#include <chrono>


//--------------------------------------------------------
//- Python wrapper to handle io_context in its own thread
//--------------------------------------------------------
CPyTextStreamIO::CPyTextStreamIO(int maxBufferSize):
    m_workGuard(boost::asio::make_work_guard(m_io)),
    m_stream(m_io, maxBufferSize)
{
    m_ioThread = std::thread([this](){ m_io.run(); });
}

CPyTextStreamIO::~CPyTextStreamIO()
{
    // First, close the stream to signal no more data will be fed
    // This will notify any waiting async operations to complete
    m_stream.close();
    // Now clear all data to break reference cycles
    m_stream.shutdown();

    m_bStopQueueProcess = true;
    m_queueCv.notify_all();
    m_workGuard.reset();
    m_io.stop();

    if (m_ioThread.joinable())
        m_ioThread.join();
}

bool CPyTextStreamIO::isFeedFinished() const
{
    return m_stream.isFeedFinished();
}

bool CPyTextStreamIO::isReadFinished() const
{
    return m_chunkQueue.empty() && m_stream.isReadFinished();
}

void CPyTextStreamIO::feed(const std::string &chunk)
{
    m_stream.feed(chunk);
}

CTextStreamIO& CPyTextStreamIO::stream()
{
    return m_stream;
}

const CTextStreamIO& CPyTextStreamIO::stream() const
{
    return m_stream;
}

std::string CPyTextStreamIO::readNext(float timeout)
{
    if (!m_bQueueThreadActive)
        startQueueProcessing();

    std::unique_lock<std::mutex> lock(m_queueMutex);

    // Otherwise, wait for data or timeout
    if (timeout > 0)
    {
        m_queueCv.wait_for(
                    lock,
                    std::chrono::milliseconds(int(timeout*1000)),
                    [this]() { return !m_chunkQueue.empty() || m_bStopQueueProcess; }
        );
    }
    else
    {
        m_queueCv.wait(
                    lock,
                    [this]() { return !m_chunkQueue.empty() || m_bStopQueueProcess; }
        );
    }

    // Timeout or stopped
    if (m_chunkQueue.empty())
        return "";

    std::string chunk = std::move(m_chunkQueue.front());
    m_chunkQueue.pop_front();
    return chunk;
}

void CPyTextStreamIO::startQueueProcessing()
{
    m_bStopQueueProcess = false;
    m_bQueueThreadActive = true;

    boost::asio::post(m_io, [this]() {
        scheduleNextRead();
    });
}

void CPyTextStreamIO::scheduleNextRead()
{
    // Fast exit if stopping
    if (m_bStopQueueProcess)
    {
        m_bQueueThreadActive = false;
        return;
    }

    m_stream.readNextAsync(
        1,
        0,
        [this](const std::string& data, const boost::system::error_code& ec)
        {
            // Check stop / error first
            if (m_bStopQueueProcess || ec)
            {
                m_bQueueThreadActive = false;
                return;
            }

            if (!data.empty())
            {
                {
                    std::lock_guard<std::mutex> lock(m_queueMutex);
                    m_chunkQueue.push_back(data);
                }
                m_queueCv.notify_one();
            }

            // Chain the next async read
            if (!m_stream.isReadFinished())
                scheduleNextRead();
            else
                m_bQueueThreadActive = false;
        }
    );
}

std::string CPyTextStreamIO::readFull()
{
    return m_stream.readFull();
}

void CPyTextStreamIO::close()
{
    m_stream.close();
}

void CPyTextStreamIO::clearData()
{
    stream().clearData();
    std::unique_lock<std::mutex> lock(m_queueMutex);
    m_chunkQueue.clear();
}

void CPyTextStreamIO::shutdown()
{
    m_bStopQueueProcess = true;
    m_stream.shutdown();
}

//------------------------------------------------
//- Python binding wrapper to handle polymorphism
//------------------------------------------------
CTextStreamIOWrap::CTextStreamIOWrap(): CPyTextStreamIO()
{
}

CTextStreamIOWrap::CTextStreamIOWrap(int maxBufferSize): CPyTextStreamIO(maxBufferSize)
{
}

std::string CTextStreamIOWrap::repr() const
{
    return stream().repr();
}

bool CTextStreamIOWrap::isDataAvailable() const
{
    CPyEnsureGIL gil;
    try
    {
        if(override isDataOver = this->get_override("is_data_available"))
            return isDataOver();

        return stream().isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CTextStreamIOWrap::default_isDataAvailable() const
{
    try
    {
        return stream().isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            CPyTextStreamIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::default_clearData()
{
    try
    {
        CPyTextStreamIO::clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::load(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override loadOver = this->get_override("load"))
            loadOver(path);
        else
            stream().load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::default_load(const std::string &path)
{
    try
    {
        stream().load(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::save(const std::string &path)
{
    CPyEnsureGIL gil;
    try
    {
        if(override saveOver = this->get_override("save"))
            saveOver(path);
        else
            stream().save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::default_save(const std::string &path)
{
    try
    {
        stream().save(path);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CTextStreamIOWrap::toJson() const
{
    CPyEnsureGIL gil;
    try
    {
        return stream().toJson(std::vector<std::string>());
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CTextStreamIOWrap::default_toJsonNoOpt() const
{
    try
    {
        return stream().toJson();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CTextStreamIOWrap::toJson(const std::vector<std::string> &options) const
{
    CPyEnsureGIL gil;
    try
    {
        if(override toJsonOver = this->get_override("to_json"))
            return toJsonOver(options);
        else
            return stream().toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

std::string CTextStreamIOWrap::default_toJson(const std::vector<std::string> &options) const
{
    try
    {
        return stream().toJson(options);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::fromJson(const std::string &jsonStr)
{
    CPyEnsureGIL gil;
    try
    {
        if(override fromJsonOver = this->get_override("from_json"))
            fromJsonOver(jsonStr);
        else
            stream().fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::default_fromJson(const std::string &jsonStr)
{
    try
    {
        stream().fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}


