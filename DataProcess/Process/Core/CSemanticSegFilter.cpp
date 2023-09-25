#include "CSemanticSegFilter.h"
#include "IO/CSemanticSegIO.h"
#include "UtilsTools.hpp"

//-----------------------------------//
//----- CSemanticSegFilterParam -----//
//-----------------------------------//
CSemanticSegFilterParam::CSemanticSegFilterParam()
{
}

void CSemanticSegFilterParam::setParamMap(const UMapString &paramMap)
{
    m_categories = paramMap.at("categories");
}

UMapString CSemanticSegFilterParam::getParamMap() const
{
    UMapString params;
    params.insert(std::make_pair("categories", m_categories));
    return params;
}

//------------------------------//
//----- CSemanticSegFilter -----//
//------------------------------//
CSemanticSegFilter::CSemanticSegFilter() : CSemanticSegTask()
{
    initIO();
}

CSemanticSegFilter::CSemanticSegFilter(const std::string name, const std::shared_ptr<CSemanticSegFilterParam> &pParam)
    : CSemanticSegTask(name)
{
    m_pParam = std::make_shared<CSemanticSegFilterParam>(*pParam);
    initIO();
}

void CSemanticSegFilter::initIO()
{
    // Remove graphics input
    removeInput(1);
    // Add semantic segmentation input
    addInput(std::make_shared<CSemanticSegIO>());
}

size_t CSemanticSegFilter::getProgressSteps()
{
    return 1;
}

void CSemanticSegFilter::run()
{
    auto paramPtr = std::dynamic_pointer_cast<CSemanticSegFilterParam>(m_pParam);
    if(paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameter", __func__, __FILE__, __LINE__);

    auto semanticSegIn = std::dynamic_pointer_cast<CSemanticSegIO>(getInput(1));
    if(semanticSegIn == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid semantic segmentation input", __func__, __FILE__, __LINE__);

    beginTaskRun();
    auto semanticSegOut = std::dynamic_pointer_cast<CSemanticSegIO>(getOutput(1));
    semanticSegOut->clearData();
    auto srcNames = semanticSegIn->getClassNames();
    auto srcColors = semanticSegIn->getColors();

    if(paramPtr->m_categories != "all")
    {
        m_classNames.clear();
        for (size_t i=0; i<srcNames.size(); ++i)
            m_classNames.push_back("other");

        m_classColors.clear();
        for (size_t i=0; i<srcColors.size(); ++i)
            m_classColors.push_back({0, 0, 0});

        std::set<std::string> categories;
        std::vector<std::string> categs;
        Utils::String::tokenize(paramPtr->m_categories, categs, ",");
        categories.insert(categs.begin(), categs.end());

        std::vector<int> classIndices;
        for (size_t i=0; i<srcNames.size(); ++i)
        {
            auto it = categories.find(srcNames[i]);
            if (it != categories.end())
                classIndices.push_back((int)i);
        }

        int empty_index = 0;
        CMat originalMask = semanticSegIn->getMask();
        CMat binMergeMask(originalMask.rows, originalMask.cols, CV_8UC1, cv::Scalar(0));

        for (size_t i=0; i<classIndices.size(); ++i)
        {
            int classIndex = classIndices[i];
            if (empty_index == classIndex)
                empty_index++;

            m_classNames[classIndex] = srcNames[classIndex];
            m_classColors[classIndex] = srcColors[classIndex];
            cv::Mat binMaskPerClass = (originalMask == classIndex);
            cv::bitwise_or(binMergeMask, binMaskPerClass, binMergeMask);
        }
        CMat newMask(originalMask.rows, originalMask.cols, CV_8UC1, cv::Scalar(empty_index));
        originalMask.copyTo(newMask, binMergeMask);
        setMask(newMask);
    }
    else
    {
        m_classNames = srcNames;
        m_classColors = srcColors;
        setMask(semanticSegIn->getMask());
    }

    endTaskRun();
    emit m_signalHandler->doProgress();
}

//-----------------------------------//
//----- CSemanticSegFilterFactory -----//
//-----------------------------------//
CSemanticSegFilterFactory::CSemanticSegFilterFactory()
{
    m_info.m_name = "ik_semantic_segmentation_filter";
    m_info.m_shortDescription = QObject::tr("This process filters semantic segmentation results based class category.").toStdString();
    m_info.m_path = QObject::tr("Core/Utils").toStdString();
    m_info.m_iconPath = QObject::tr(":/Images/default-process.png").toStdString();
    m_info.m_keywords = "semantic,segmentation,filtering,mask";
}

WorkflowTaskPtr CSemanticSegFilterFactory::create(const WorkflowTaskParamPtr &pParam)
{
    auto pDerivedParam = std::dynamic_pointer_cast<CSemanticSegFilterParam>(pParam);
    if(pDerivedParam != nullptr)
        return std::make_shared<CSemanticSegFilter>(m_info.m_name, pDerivedParam);
    else
        return create();
}

WorkflowTaskPtr CSemanticSegFilterFactory::create()
{
    auto pDerivedParam = std::make_shared<CSemanticSegFilterParam>();
    assert(pDerivedParam != nullptr);
    return std::make_shared<CSemanticSegFilter>(m_info.m_name, pDerivedParam);
}

//----------------------------------//
//----- CWidgetSemanticSegFilter -----//
//----------------------------------//
CWidgetSemanticSegFilter::CWidgetSemanticSegFilter(QWidget *parent): CWorkflowTaskWidget(parent)
{
    init();
}

CWidgetSemanticSegFilter::CWidgetSemanticSegFilter(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent)
    : CWorkflowTaskWidget(parent)
{
    m_pParam = std::dynamic_pointer_cast<CSemanticSegFilterParam>(pParam);
    init();
}

void CWidgetSemanticSegFilter::init()
{
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CSemanticSegFilterParam>();

    m_pEditCategories = addEdit("Categories", QString::fromStdString(m_pParam->m_categories));
}

void CWidgetSemanticSegFilter::onApply()
{
    m_pParam->m_categories = m_pEditCategories->text().toStdString();
    emit doApplyProcess(m_pParam);
}

//-----------------------------------------//
//----- CWidgetSemanticSegFilterFactory -----//
//-----------------------------------------//
CWidgetSemanticSegFilterFactory::CWidgetSemanticSegFilterFactory()
{
    m_name = "ik_semantic_segmentation_filter";
}

WorkflowTaskWidgetPtr CWidgetSemanticSegFilterFactory::create(std::shared_ptr<CWorkflowTaskParam> pParam)
{
    return std::make_shared<CWidgetSemanticSegFilter>(pParam);
}
