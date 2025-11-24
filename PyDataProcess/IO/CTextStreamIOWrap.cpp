#include "CTextStreamIOWrap.h"


//--------------------------------------------------------
//- Python wrapper to handle io_context in its own thread
//--------------------------------------------------------
CPyTextStreamIO::CPyTextStreamIO(int maxBufferSize):
    m_workGuard(boost::asio::make_work_guard(m_io)),
    m_stream(m_io, maxBufferSize),
    m_thread([this](){ m_io.run(); })
{
}

CPyTextStreamIO::~CPyTextStreamIO()
{
    m_workGuard.reset();
    m_io.stop();

    if (m_thread.joinable())
        m_thread.join();
}

CTextStreamIO& CPyTextStreamIO::stream()
{
    return m_stream;
}

const CTextStreamIO& CPyTextStreamIO::stream() const
{
    return m_stream;
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
    CPyEnsureGIL gil;
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
    CPyEnsureGIL gil;
    try
    {
        return stream().isDataAvailable();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

bool CTextStreamIOWrap::isFeedFinished() const
{
    CPyEnsureGIL gil;
    return stream().isFeedFinished();
}

bool CTextStreamIOWrap::isReadFinished() const
{
    CPyEnsureGIL gil;
    return stream().isReadFinished();
}

void CTextStreamIOWrap::feed(const std::string &chunk)
{
    stream().feed(chunk);
}

void CTextStreamIOWrap::readNextAsync(int minBytes, int timeout, boost::python::object py_callback)
{
    // Keep Python callback alive
    auto py_cb = std::make_shared<boost::python::object>(py_callback);

    // Keep "this" alive as long as async op is alive
    auto self = shared_from_this();

    CTextStreamIO::Handler handler = [self, py_cb](const std::string &data, const boost::system::error_code &ec)
    {
        // Acquire GIL safely
        CPyEnsureGIL gil;

        try
        {
            if (ec == boost::system::errc::timed_out)
                (*py_cb)(boost::python::object());   // pass None
            else
                (*py_cb)(data);
        }
        catch (const boost::python::error_already_set &)
        {
            Utils::print(Utils::Python::handlePythonException(), QtCriticalMsg);
        }
    };

    // Register async read
    stream().readNextAsync(minBytes, timeout, handler);
}

void CTextStreamIOWrap::readFullAsync(int timeout, api::object py_callback)
{
    // Keep Python callback alive
    auto py_cb = std::make_shared<boost::python::object>(py_callback);

    // Keep "this" alive as long as async op is alive
    auto self = shared_from_this();

    CTextStreamIO::Handler handler = [self, py_cb](const std::string &data, const boost::system::error_code &ec)
    {
        // Acquire GIL safely
        CPyEnsureGIL gil;

        try
        {
            if (ec == boost::system::errc::timed_out)
                (*py_cb)(boost::python::object());   // pass None
            else
                (*py_cb)(data);
        }
        catch (const boost::python::error_already_set &)
        {
            Utils::print(Utils::Python::handlePythonException(), QtCriticalMsg);
        }
    };

    // Register async read
    stream().readFullAsync(timeout, handler);
}

std::string CTextStreamIOWrap::readFull()
{
    CPyEnsureGIL gil;
    return stream().readFull();
}

void CTextStreamIOWrap::close()
{
    CPyEnsureGIL gil;
    stream().close();
}

void CTextStreamIOWrap::clearData()
{
    CPyEnsureGIL gil;
    try
    {
        if(override clearDataOver = this->get_override("clear_data"))
            clearDataOver();
        else
            stream().clearData();
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}

void CTextStreamIOWrap::default_clearData()
{
    CPyEnsureGIL gil;
    try
    {
        stream().clearData();
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
    CPyEnsureGIL gil;
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
    CPyEnsureGIL gil;
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
    CPyEnsureGIL gil;
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
    CPyEnsureGIL gil;
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
    CPyEnsureGIL gil;
    try
    {
        stream().fromJson(jsonStr);
    }
    catch(boost::python::error_already_set&)
    {
        throw CException(CoreExCode::PYTHON_EXCEPTION, Utils::Python::handlePythonException(), __func__, __FILE__, __LINE__);
    }
}


