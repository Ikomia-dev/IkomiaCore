#include "CClassificationTask.h"
#include "IO/CObjectDetectionIO.h"
#include "IO/CGraphicsOutput.h"
#include "IO/CNumericIO.h"

CClassificationTask::CClassificationTask(): C2dImageTask(true)
{
    initIO();
}

CClassificationTask::CClassificationTask(const std::string &name): C2dImageTask(name, true)
{
    initIO();
}

std::vector<std::string> CClassificationTask::getNames() const
{
    return m_classNames;
}

void CClassificationTask::initIO()
{
    addOutput(std::make_shared<CObjectDetectionIO>());
    addOutput(std::make_shared<CGraphicsOutput>());
    addOutput(std::make_shared<CNumericIO<std::string>>());
}

void CClassificationTask::generateRandomColors()
{
    std::srand(RANDOM_COLOR_SEED);
    double factor = 255.0 / (double)RAND_MAX;

    for (size_t i=0; i<m_classNames.size(); ++i)
    {
        CColor color = {
            (int)((double)std::rand() * factor),
            (int)((double)std::rand() * factor),
            (int)((double)std::rand() * factor)
        };
        m_classColors.push_back(color);
    }
}

std::vector<ProxyGraphicsItemPtr> CClassificationTask::getInputObjects() const
{
    auto graphicsInPtr = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
    if (graphicsInPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid graphics input", __func__, __FILE__, __LINE__);

    std::vector<ProxyGraphicsItemPtr> objects;
    std::vector<ProxyGraphicsItemPtr> items = graphicsInPtr->getItems();

    for (size_t i=0; i<items.size(); ++i)
    {
        if (items[i]->isTextItem() == false)
            objects.push_back(items[i]);
    }
    return objects;
}

CMat CClassificationTask::getObjectSubImage(const ProxyGraphicsItemPtr &objectPtr) const
{
    if (objectPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid input object", __func__, __FILE__, __LINE__);

    auto imageInPtr = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    if (imageInPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid source image", __func__, __FILE__, __LINE__);

    cv::Mat srcImg = imageInPtr->getImage();
    std::vector<float> rc = objectPtr->getBoundingRect();

    if (rc[0] < 0)
        rc[0] = 0;
    if (rc[0] + rc[2] >= srcImg.cols)
        rc[2] = srcImg.cols - rc[0] - 1;
    if (rc[1] < 0)
        rc[1] = 0;
    if (rc[1] + rc[3] >= srcImg.rows)
        rc[3] = srcImg.rows - rc[1] - 1;

    if (rc[2] < 2 || rc[3] < 2)
        return CMat();

    CMat subImg = srcImg(cv::Range(rc[1], rc[1] + rc[3]), cv::Range(rc[0], rc[0] + rc[2]));
    return subImg;
}

std::shared_ptr<CObjectDetectionIO> CClassificationTask::getObjectsResults() const
{
    auto objDetectIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetectIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    return objDetectIOPtr;
}

std::vector<PairString> CClassificationTask::getWholeImageResults() const
{
    auto numericOutPtr =  std::dynamic_pointer_cast<CNumericIO<std::string>>(getOutput(3));
    if (numericOutPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid data output", __func__, __FILE__, __LINE__);

    std::vector<PairString> results;
    std::vector<std::string> names = numericOutPtr->getValueLabelList(0);
    std::vector<std::string> confidences = numericOutPtr->getValueList(0);

    if (names.size() != confidences.size())
        throw CException(CoreExCode::INVALID_SIZE, "Size mismatch between class names and confidences", __func__, __FILE__, __LINE__);

    for (size_t i=0; i<names.size(); ++i)
        results.push_back(std::make_pair(names[i], confidences[i]));

    return results;
}

bool CClassificationTask::isWholeImageClassification() const
{
    auto graphicsIn = getInput(1);
    return graphicsIn->isDataAvailable() == false;
}

void CClassificationTask::setNames(const std::vector<std::string> &names)
{
    m_classNames = names;
    if (m_classColors.empty())
        generateRandomColors();
}

void CClassificationTask::setColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_classColors = colors;
}

void CClassificationTask::readClassNames(const std::string& path)
{
    if(path.empty())
        throw CException(CoreExCode::INVALID_FILE, "Path to class names file is empty", __func__, __FILE__, __LINE__);

    std::ifstream file(path);
    if(file.is_open() == false)
        throw CException(CoreExCode::INVALID_FILE, "Failed to open labels file: " + path, __func__, __FILE__, __LINE__);

    std::string name;
    m_classNames.clear();

    while(!file.eof())
    {
        std::getline(file, name);
        if(name.empty() == false)
            m_classNames.push_back(name);
    }
    file.close();

    if (m_classColors.empty())
        generateRandomColors();
}

void CClassificationTask::setWholeImageResults(const std::vector<std::string> &sortedNames, const std::vector<std::string> &sortedConfidences)
{
    if (sortedNames.size() != sortedConfidences.size())
        throw CException(CoreExCode::INVALID_SIZE, "Size of class names and class confidences must be equal", __func__, __FILE__, __LINE__);

    auto graphicsOutPtr =  std::dynamic_pointer_cast<CGraphicsOutput>(getOutput(2));
    if (graphicsOutPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid graphics output", __func__, __FILE__, __LINE__);

    // Init graphics layer
    graphicsOutPtr->setNewLayer(getName());
    graphicsOutPtr->setImageIndex(0);

    // Add text graphics to display top-1 result
    std::stringstream streamConf;
    streamConf << std::fixed << std::setprecision(3) << sortedConfidences[0];
    graphicsOutPtr->addText(sortedNames[0] + ": " + streamConf.str(), 20, 20);

    // Add table results
    auto numericOutPtr =  std::dynamic_pointer_cast<CNumericIO<std::string>>(getOutput(3));
    if (numericOutPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid numeric output", __func__, __FILE__, __LINE__);

    numericOutPtr->addValueList(sortedConfidences, "Confidence", sortedNames);
}

void CClassificationTask::addObject(const ProxyGraphicsItemPtr &objectPtr, int classIndex, double confidence)
{
    auto objDetectIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetectIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    if (classIndex >= m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Invalid class index, index overflows class names list", __func__, __FILE__, __LINE__);

    std::vector<float> rc = objectPtr->getBoundingRect();
    objDetectIOPtr->addObject(objectPtr->getId(), m_classNames[classIndex], confidence, rc[0], rc[1], rc[2], rc[3], m_classColors[classIndex]);
}

void CClassificationTask::endTaskRun()
{
    forwardInputImage(0, 0);
    C2dImageTask::endTaskRun();
}
