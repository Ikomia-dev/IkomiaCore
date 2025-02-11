# include "CNumericIO.h"

template <>
std::string CNumericIO<std::string>::repr() const
{
    std::stringstream s;
    s << "CDataStringIO(" << m_name << ")";
    return s.str();
}

template <>
std::vector<std::vector<std::string> > CNumericIO<std::string>::getAllValuesAsString() const
{
    return m_values;
}

template <>
void CNumericIO<std::string>::saveCSV(const std::string &path) const
{
    std::ofstream file;
    file.open(path, std::ios::out | std::ios::trunc);

    //Output type
    file << "Output type" << ";" << std::to_string(static_cast<int>(m_outputType)) << std::endl;

    //Plot type
    file << "Plot type" << ";" << std::to_string(static_cast<int>(m_plotType)) << std::endl;

    //Write header labels
    if(m_headerLabels.size() > 0)
    {
        if(m_valueLabels.size() > 0)
            file << ";";

        for(size_t i=0; i<m_headerLabels.size(); ++i)
            file << m_headerLabels[i] + ";";
        file << "\n";
    }
    else
        file << "No headers" << std::endl;

    //Count rows
    size_t nbRow = 0;
    for(size_t i=0; i<m_values.size(); ++i)
        nbRow = std::max(nbRow, m_values[i].size());

    //Write values and associated labels (if there are some)
    if(m_values.size() == m_valueLabels.size())
    {
        for(size_t i=0; i<nbRow; ++i)
        {
            for(size_t j=0; j<m_values.size(); ++j)
            {
                file << m_valueLabels[j][i] + ";";
                file << m_values[j][i] + ";";
            }
            file << "\n";
        }
    }
    else if(m_valueLabels.size() == 1)
    {
        for(size_t i=0; i<nbRow; ++i)
        {
            file << m_valueLabels[0][i] + ";";
            for(size_t j=0; j<m_values.size(); ++j)
                file << m_values[j][i] + ";";

            file << "\n";
        }
    }
    else
    {
        for(size_t i=0; i<nbRow; ++i)
        {
            for(size_t j=0; j<m_values.size(); ++j)
                file << m_values[j][i] + ";";

            file << "\n";
        }
    }
    file.close();
}

template <>
void CNumericIO<std::string>::loadCSV(const std::string &path)
{
    std::string line;
    std::ifstream file(path);
    std::vector<std::string> lineValues;
    std::vector<std::vector<std::string>> rowValues;

    //Output type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_outputType = static_cast<NumericOutputType>(std::stoi(lineValues[1]));

    //Plot type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_plotType = static_cast<PlotType>(std::stoi(lineValues[1]));

    // Header labels
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() > 0 && lineValues[0] != "No headers")
        m_headerLabels = lineValues;

    // Values
    while (std::getline(file, line))
    {
        Utils::String::tokenize(line, lineValues, ";");
        rowValues.push_back(lineValues);
    }

    // Transpose row/column
    size_t cols = 0;
    for(size_t i=0; i<rowValues.size(); ++i)
        cols = std::max(cols, rowValues[i].size());

    m_values.resize(cols);
    for (size_t i=0; i<m_values.size(); ++i)
        m_values[i].resize(rowValues.size());

    for (size_t i=0; i<rowValues.size(); ++i)
        for (size_t j=0; j<rowValues[i].size(); ++j)
            m_values[j][i] = rowValues[i][j];
}

template <>
void CNumericIO<double>::loadCSV(const std::string &path)
{
    std::string line;
    std::ifstream file(path);
    std::vector<std::string> lineValues;
    double val;
    std::vector<std::vector<double>> rowValues;
    std::vector<std::vector<std::string>> rowLabels;

    //Output type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_outputType = static_cast<NumericOutputType>(std::stoi(lineValues[1]));

    //Plot type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_plotType = static_cast<PlotType>(std::stoi(lineValues[1]));

    // Header labels
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() > 0 && lineValues[0] != "No headers")
        m_headerLabels = lineValues;

    while (std::getline(file, line))
    {
        std::vector<double> values;
        std::vector<std::string> labels;
        Utils::String::tokenize(line, lineValues, ";");

        for (size_t i=0; i<lineValues.size(); ++i)
        {
            try
            {
                val = std::stod(lineValues[i]);
                values.push_back(val);
            }
            catch(std::exception& e)
            {
                //It's a row label
                labels.push_back(lineValues[i]);
            }
        }
        rowValues.push_back(values);
        rowLabels.push_back(labels);
    }

    // Transpose row/column for values
    size_t cols = 0;
    for(size_t i=0; i<rowValues.size(); ++i)
        cols = std::max(cols, rowValues[i].size());

    m_values.resize(cols);
    for (size_t i=0; i<m_values.size(); ++i)
        m_values[i].resize(rowValues.size());

    for (size_t i=0; i<rowValues.size(); ++i)
        for (size_t j=0; j<rowValues[i].size(); ++j)
            m_values[j][i] = rowValues[i][j];

    // Transpose row/column for labels
    cols = 0;
    for(size_t i=0; i<rowLabels.size(); ++i)
        cols = std::max(cols, rowLabels[i].size());

    m_valueLabels.resize(cols);
    for (size_t i=0; i<m_valueLabels.size(); ++i)
        m_valueLabels[i].resize(rowLabels.size());

    for (size_t i=0; i<rowLabels.size(); ++i)
        for (size_t j=0; j<rowLabels[i].size(); ++j)
            m_valueLabels[j][i] = rowLabels[i][j];
}

template <>
void CNumericIO<int>::loadCSV(const std::string &path)
{
    std::string line;
    std::ifstream file(path);
    std::vector<std::string> lineValues;
    int val;
    std::vector<std::vector<int>> rowValues;
    std::vector<std::vector<std::string>> rowLabels;

    //Output type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_outputType = static_cast<NumericOutputType>(std::stoi(lineValues[1]));

    //Plot type
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() == 2)
        m_plotType = static_cast<PlotType>(std::stoi(lineValues[1]));

    // Header labels
    std::getline(file, line);
    Utils::String::tokenize(line, lineValues, ";");

    if (lineValues.size() > 0 && lineValues[0] != "No headers")
        m_headerLabels = lineValues;

    while (std::getline(file, line))
    {
        std::vector<int> values;
        std::vector<std::string> labels;
        Utils::String::tokenize(line, lineValues, ";");

        for (size_t i=0; i<lineValues.size(); ++i)
        {
            try
            {
                val = std::stoi(lineValues[i]);
                values.push_back(val);
            }
            catch(std::exception& e)
            {
                //It's a row label
                labels.push_back(lineValues[i]);
            }
        }
        rowValues.push_back(values);
        rowLabels.push_back(labels);
    }

    // Transpose row/column for values
    size_t cols = 0;
    for(size_t i=0; i<rowValues.size(); ++i)
        cols = std::max(cols, rowValues[i].size());

    m_values.resize(cols);
    for (size_t i=0; i<m_values.size(); ++i)
        m_values[i].resize(rowValues.size());

    for (size_t i=0; i<rowValues.size(); ++i)
        for (size_t j=0; j<rowValues[i].size(); ++j)
            m_values[j][i] = rowValues[i][j];

    // Transpose row/column for labels
    cols = 0;
    for(size_t i=0; i<rowLabels.size(); ++i)
        cols = std::max(cols, rowLabels[i].size());

    m_valueLabels.resize(cols);
    for (size_t i=0; i<m_valueLabels.size(); ++i)
        m_valueLabels[i].resize(rowLabels.size());

    for (size_t i=0; i<rowLabels.size(); ++i)
        for (size_t j=0; j<rowLabels[i].size(); ++j)
            m_valueLabels[j][i] = rowLabels[i][j];
}

template <>
QJsonObject CNumericIO<std::string>::toJsonInternal() const
{
    QJsonObject root;
    toJsonCommon(root);

    QJsonArray values;
    for (size_t i=0; i<m_values.size(); ++i)
    {
        QJsonArray colValues;
        for (size_t j=0; j<m_values[i].size(); ++j)
            colValues.append(QString::fromStdString(m_values[i][j]));

        values.append(colValues);
    }
    root["values"] = values;
    return root;
}

template <>
void CNumericIO<std::string>::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading string data: invalid JSON structure", __func__, __FILE__, __LINE__);

    QJsonObject root = jsonDoc.object();
    if (root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading string data: empty JSON structure", __func__, __FILE__, __LINE__);

     fromJsonCommon(root);

     m_values.clear();
     QJsonArray valueArray = root["values"].toArray();

     for (int i=0; i<valueArray.size(); ++i)
     {
         std::vector<std::string> values;
         QJsonArray colValues = valueArray[i].toArray();

         for (int j=0; j<colValues.size(); ++j)
             values.push_back(colValues[i].toString().toStdString());

         m_values.push_back(values);
     }
}

template <>
void CNumericIO<double>::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading numeric data: invalid JSON structure", __func__, __FILE__, __LINE__);

    QJsonObject root = jsonDoc.object();
    if (root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading numeric data: empty JSON structure", __func__, __FILE__, __LINE__);

     fromJsonCommon(root);

     m_values.clear();
     QJsonArray valueArray = root["values"].toArray();

     for (int i=0; i<valueArray.size(); ++i)
     {
         std::vector<double> values;
         QJsonArray colValues = valueArray[i].toArray();

         for (int j=0; j<colValues.size(); ++j)
             values.push_back(colValues[i].toDouble());

         m_values.push_back(values);
     }
}

template <>
void CNumericIO<int>::fromJson(const std::string &jsonStr)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(QString::fromStdString(jsonStr).toUtf8());
    if (jsonDoc.isNull() || jsonDoc.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading numeric data: invalid JSON structure", __func__, __FILE__, __LINE__);

    QJsonObject root = jsonDoc.object();
    if (root.isEmpty())
        throw CException(CoreExCode::INVALID_JSON_FORMAT, "Error while loading numeric data: empty JSON structure", __func__, __FILE__, __LINE__);

     fromJsonCommon(root);

     m_values.clear();
     QJsonArray valueArray = root["values"].toArray();

     for (int i=0; i<valueArray.size(); ++i)
     {
         std::vector<int> values;
         QJsonArray colValues = valueArray[i].toArray();

         for (int j=0; j<colValues.size(); ++j)
             values.push_back(colValues[i].toInt());

         m_values.push_back(values);
     }
}
