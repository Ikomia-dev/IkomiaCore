#include "CKeypointDetectionTask.h"

CKeypointDetectionTask::CKeypointDetectionTask(): C2dImageTask()
{
    initIO();
}

CKeypointDetectionTask::CKeypointDetectionTask(const std::string &name): C2dImageTask(name)
{
    initIO();
}

void CKeypointDetectionTask::initIO()
{
    addOutput(std::make_shared<CKeypointsIO>());
}

void CKeypointDetectionTask::generateRandomColors()
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

void CKeypointDetectionTask::addObject(int id, int classIndex, double confidence, double x, double y, double width, double height, const std::vector<Keypoint> keypts)
{
    auto keyptsIOPtr = std::dynamic_pointer_cast<CKeypointsIO>(getOutput(1));
    if (keyptsIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid keypoints detection output", __func__, __FILE__, __LINE__);

    if (classIndex >= m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Invalid class index, index overflows class names list", __func__, __FILE__, __LINE__);

    if (keyptsIOPtr->getObjectCount() == 0)
    {
        keyptsIOPtr->init(getName(), 0);
        keyptsIOPtr->setKeypointNames(m_keyptsNames);
        keyptsIOPtr->setKeypointLinks(m_keyptsLinks);
    }
    keyptsIOPtr->addObject(id, m_classNames[classIndex], confidence, x, y, width, height, keypts, m_classColors[classIndex]);
}

void CKeypointDetectionTask::endTaskRun()
{
    forwardInputImage(0, 0);
    C2dImageTask::endTaskRun();
}

std::vector<CKeypointLink> CKeypointDetectionTask::getKeypointLinks() const
{
    return m_keyptsLinks;
}

std::vector<std::string> CKeypointDetectionTask::getKeypointNames() const
{
    return m_keyptsNames;
}

std::vector<std::string> CKeypointDetectionTask::getObjectNames() const
{
    return m_classNames;
}

std::shared_ptr<CKeypointsIO> CKeypointDetectionTask::getResults() const
{
    auto keyptsIOPtr = std::dynamic_pointer_cast<CKeypointsIO>(getOutput(1));
    if (keyptsIOPtr == nullptr)
        throw CException(CoreExCode::NULL_POINTER, "Invalid keypoint detection output", __func__, __FILE__, __LINE__);

    return keyptsIOPtr;
}

void CKeypointDetectionTask::readClassNames(const std::string &path)
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

void CKeypointDetectionTask::setKeypointNames(const std::vector<std::string> &names)
{
    m_keyptsNames = names;
}

void CKeypointDetectionTask::setKeypointLinks(const std::vector<CKeypointLink> &links)
{
    m_keyptsLinks = links;
}

void CKeypointDetectionTask::setObjectColors(const std::vector<CColor> &colors)
{
    if (colors.size() < m_classNames.size())
        throw CException(CoreExCode::INVALID_SIZE, "Colors count must be greater or equal of class names count", __func__, __FILE__, __LINE__);

    m_classColors = colors;
}

void CKeypointDetectionTask::setObjectNames(const std::vector<std::string> &names)
{
    m_classNames = names;
    if (m_classColors.empty())
        generateRandomColors();
}
