#include "CObjectDetectionTask.h"

CObjectDetectionTask::CObjectDetectionTask(): C2dImageTask()
{
    initIO();
}

CObjectDetectionTask::CObjectDetectionTask(const std::string &name): C2dImageTask(name)
{
    initIO();
}

void CObjectDetectionTask::addObject(int id, int classIndex, double confidence, double boxX, double boxY, double boxWidth, double boxHeight)
{
    auto objDetIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    if (classIndex >= m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Invalid class index, index overflows class names list", __func__, __FILE__, __LINE__);

    if (objDetIOPtr->getObjectCount() == 0)
        objDetIOPtr->init(getName(), 0);

    objDetIOPtr->addObject(id, m_classNames[classIndex], confidence, boxX, boxY, boxWidth, boxHeight, m_classColors[classIndex]);
}

void CObjectDetectionTask::addObject(int id, int classIndex, double confidence, double cx, double cy, double width, double height, double angle)
{
    auto objDetIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    if (classIndex >= m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Invalid class index, index overflows class names list", __func__, __FILE__, __LINE__);

    if (objDetIOPtr->getObjectCount() == 0)
        objDetIOPtr->init(getName(), 0);

    objDetIOPtr->addObject(id, m_classNames[classIndex], confidence, cx, cy, width, height, angle, m_classColors[classIndex]);
}

void CObjectDetectionTask::endTaskRun()
{
    forwardInputImage(0, 0);
    C2dImageTask::endTaskRun();
}

std::vector<std::string> CObjectDetectionTask::getNames() const
{
    return m_classNames;
}

ObjectDetectionIOPtr CObjectDetectionTask::getResults() const
{
    auto objDetIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    return objDetIOPtr;
}

CMat CObjectDetectionTask::getImageWithGraphics() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto objDetIOPtr = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(1));
    if (objDetIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    auto graphicsIOPtr = objDetIOPtr->getGraphicsIO();
    return imgIOPtr->getImageWithGraphics(graphicsIOPtr);
}

void CObjectDetectionTask::initIO()
{
    addOutput(std::make_shared<CObjectDetectionIO>());
}

void CObjectDetectionTask::generateRandomColors()
{
    std::srand(RANDOM_COLOR_SEED);
    double factor = 255.0 / (double)RAND_MAX;

    for (size_t i=0; i<m_classNames.size(); ++i)
    {
        CColor color = {
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor)
        };
        m_classColors.push_back(color);
    }
}

void CObjectDetectionTask::readClassNames(const std::string &path)
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

void CObjectDetectionTask::setNames(const std::vector<std::string> &names)
{
    m_classNames = names;
    if (m_classColors.empty())
        generateRandomColors();
}

void CObjectDetectionTask::setColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_classColors = colors;
}
