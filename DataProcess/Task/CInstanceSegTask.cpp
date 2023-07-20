#include "CInstanceSegTask.h"
#include "DataProcessTools.hpp"

CInstanceSegTask::CInstanceSegTask(): C2dImageTask()
{
    init();
}

CInstanceSegTask::CInstanceSegTask(const std::string &name): C2dImageTask(name)
{
    init();
}

std::string CInstanceSegTask::repr() const
{
    std::stringstream s;
    s << "CInstanceSegmentationTask(" << m_name <<  ")";
    return s.str();
}

void CInstanceSegTask::addObject(int id, int type, int classIndex, double confidence, double x, double y, double width, double height, const CMat &mask)
{
    auto instanceSegIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid instance segmentation output", __func__, __FILE__, __LINE__);

    if (classIndex >= m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Invalid class index, index overflows class names list", __func__, __FILE__, __LINE__);

    if (instanceSegIOPtr->getObjectCount() == 0)
    {
        auto imgInPtr = std::dynamic_pointer_cast<CImageIO>(getInput(0));
        CMat imgSrc = imgInPtr->getImage();
        instanceSegIOPtr->init(getName(), 0, imgSrc.cols, imgSrc.rows);
    }

    instanceSegIOPtr->addObject(id, type, classIndex, m_classNames[classIndex], confidence, x, y, width, height, mask, m_classColors[classIndex]);
}

void CInstanceSegTask::init()
{
    addOutput(std::make_shared<CInstanceSegIO>());
}

void CInstanceSegTask::endTaskRun()
{
    forwardInputImage(0, 0);
    setOutputColorMap(0, 1, m_classColors, true);
    C2dImageTask::endTaskRun();
}

void CInstanceSegTask::generateRandomColors()
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

std::vector<std::string> CInstanceSegTask::getNames() const
{
    return m_classNames;
}

InstanceSegIOPtr CInstanceSegTask::getResults() const
{
    auto instanceSegIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    return instanceSegIOPtr;
}

CMat CInstanceSegTask::getImageWithMask() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto instanceSegIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    return Utils::Image::mergeColorMask(imgIOPtr->getImage(), instanceSegIOPtr->getMergeMask(), getColorMap(0), 0.7, false);
}

CMat CInstanceSegTask::getImageWithGraphics() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto instanceSegIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    auto graphicsIOPtr = instanceSegIOPtr->getGraphicsIO();
    return imgIOPtr->getImageWithGraphics(graphicsIOPtr);
}

CMat CInstanceSegTask::getImageWithMaskAndGraphics() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto instanceSegIOPtr = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid object detection output", __func__, __FILE__, __LINE__);

    auto graphicsIOPtr = instanceSegIOPtr->getGraphicsIO();
    CMat imgWithGraphics = imgIOPtr->getImageWithGraphics(graphicsIOPtr);
    return Utils::Image::mergeColorMask(imgWithGraphics, instanceSegIOPtr->getMergeMask(), getColorMap(0), 0.7, true);
}

void CInstanceSegTask::readClassNames(const std::string &path)
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

void CInstanceSegTask::setColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_classColors = colors;
}

void CInstanceSegTask::setNames(const std::vector<std::string> &names)
{
    m_classNames = names;
    if (m_classColors.empty())
        generateRandomColors();
}
