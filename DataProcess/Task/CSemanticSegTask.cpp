#include "CSemanticSegTask.h"
#include "DataProcessTools.hpp"

CSemanticSegTask::CSemanticSegTask(): C2dImageTask()
{
    init();
}

CSemanticSegTask::CSemanticSegTask(const std::string &name): C2dImageTask(name)
{
    init();
}

std::string CSemanticSegTask::repr() const
{
    std::stringstream s;
    s << "CSemanticSegmentationTask(" << m_name <<  ")";
    return s.str();
}

void CSemanticSegTask::init()
{
    addOutput(std::make_shared<CSemanticSegIO>());
}

void CSemanticSegTask::endTaskRun()
{
    auto segIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (segIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid segmentation output", __func__, __FILE__, __LINE__);

    forwardInputImage(0, 0);
    setOutputColorMap(0, 1, segIOPtr->getColors());
    segIOPtr->setReferenceImageIndex(0);
    C2dImageTask::endTaskRun();
}

std::vector<std::string> CSemanticSegTask::getNames() const
{
    auto segIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (segIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid segmentation output", __func__, __FILE__, __LINE__);

    return segIOPtr->getClassNames();
}

SemanticSegIOPtr CSemanticSegTask::getResults() const
{
    auto segIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (segIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid segmentation output", __func__, __FILE__, __LINE__);

    return segIOPtr;
}

CMat CSemanticSegTask::getImageWithMask() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image input", __func__, __FILE__, __LINE__);

    auto segIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (segIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid segmentation output", __func__, __FILE__, __LINE__);

    return imgIOPtr->getImageWithMask(segIOPtr);
}

CMat CSemanticSegTask::getImageWithGraphics() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto semanticSegIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (semanticSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid semantic segmentation output", __func__, __FILE__, __LINE__);

    return imgIOPtr->getImageWithGraphics(semanticSegIOPtr);
}

CMat CSemanticSegTask::getImageWithMaskAndGraphics() const
{
    auto imgIOPtr = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
    if (imgIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid image output", __func__, __FILE__, __LINE__);

    auto semanticSegIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (semanticSegIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid semantic segmentation output", __func__, __FILE__, __LINE__);

    return imgIOPtr->getImageWithMaskAndGraphics(semanticSegIOPtr);
}

void CSemanticSegTask::readClassNames(const std::string &path)
{
    if(path.empty())
        throw CException(CoreExCode::INVALID_FILE, "Path to class names file is empty", __func__, __FILE__, __LINE__);

    std::ifstream file(path);
    if(file.is_open() == false)
        throw CException(CoreExCode::INVALID_FILE, "Failed to open labels file: " + path, __func__, __FILE__, __LINE__);

    m_classNames.clear();
    std::string name;

    while(!file.eof())
    {
        std::getline(file, name);
        if(name.empty() == false)
            m_classNames.push_back(name);
    }
    file.close();
}

void CSemanticSegTask::setColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_classColors = colors;
}

void CSemanticSegTask::setNames(const std::vector<std::string> &names)
{
    if (m_classColors.size() != 0 && names.size() != m_classColors.size())
        throw CException(CoreExCode::INVALID_SIZE, "Semantic segmentation error: there must be the same number of classes and colors.", __func__, __FILE__, __LINE__);

    m_classNames = names;
}

void CSemanticSegTask::setMask(const CMat &mask)
{
    auto segIOPtr = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    if (segIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid segmentation output", __func__, __FILE__, __LINE__);

    segIOPtr->setClassNames(m_classNames);
    if (m_classColors.size() > 0)
        segIOPtr->setClassColors(m_classColors);

    segIOPtr->setMask(mask);
}
