#include "CInstanceSegFilter.h"
#include "IO/CInstanceSegIO.h"
#include "UtilsTools.hpp"

//-----------------------------------//
//----- CInstanceSegFilterParam -----//
//-----------------------------------//
CInstanceSegFilterParam::CInstanceSegFilterParam()
{
}

void CInstanceSegFilterParam::setParamMap(const UMapString &paramMap)
{
    m_confidence = std::stof(paramMap.at("confidence"));
    m_categories = paramMap.at("categories");
}

UMapString CInstanceSegFilterParam::getParamMap() const
{
    UMapString params;
    params.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    params.insert(std::make_pair("categories", m_categories));
    return params;
}

//------------------------------//
//----- CInstanceSegFilter -----//
//------------------------------//
CInstanceSegFilter::CInstanceSegFilter() : CInstanceSegTask()
{
    initIO();
}

CInstanceSegFilter::CInstanceSegFilter(const std::string name, const std::shared_ptr<CInstanceSegFilterParam> &pParam)
    : CInstanceSegTask(name)
{
    m_pParam = std::make_shared<CInstanceSegFilterParam>(*pParam);
    initIO();
}

void CInstanceSegFilter::initIO()
{
    // Remove graphics input
    removeInput(1);
    // Add instance segmentation input
    addInput(std::make_shared<CInstanceSegIO>());
}

size_t CInstanceSegFilter::getProgressSteps()
{
    return 2;
}

void CInstanceSegFilter::run()
{
    beginTaskRun();
    m_classNames.clear();
    m_classColors.clear();

    auto paramPtr = std::dynamic_pointer_cast<CInstanceSegFilterParam>(m_pParam);
    if (paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    auto imgInPtr = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    if (imgInPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

    auto instanceSegIn = std::dynamic_pointer_cast<CInstanceSegIO>(getInput(1));
    if (instanceSegIn == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid instance segmentation input", __func__, __FILE__, __LINE__);

    auto instanceSegOut = std::dynamic_pointer_cast<CInstanceSegIO>(getOutput(1));
    if (instanceSegOut == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid instance segmentation output", __func__, __FILE__, __LINE__);

    instanceSegOut->clearData();
    CMat maskIn = instanceSegIn->getMergeMask();
    instanceSegOut->init(m_name, 0, maskIn.cols, maskIn.rows);

    std::set<std::string> categories;
    if(paramPtr->m_categories != "all")
    {
        std::vector<std::string> categs;
        Utils::String::tokenize(paramPtr->m_categories, categs, ",");
        categories.insert(categs.begin(), categs.end());
    }
    emit m_signalHandler->doProgress();

    auto instances = instanceSegIn->getInstances();
    for(size_t i=0; i<instances.size(); ++i)
    {
        if (instances[i].m_confidence >= paramPtr->m_confidence &&
                (categories.empty() || categories.find(instances[i].m_label) != categories.end()))
        {
            int classIndex = instances[i].m_classIndex;
            if (classIndex >= m_classNames.size())
            {
                for (size_t i=m_classNames.size(); i<classIndex; ++i)
                {
                    m_classNames.push_back("other");
                    m_classColors.push_back({0, 0, 0});
                }
                m_classNames.push_back(instances[i].m_label);
                m_classColors.push_back(instances[i].m_color);
            }
            else
            {
                m_classNames[classIndex] = instances[i].m_label;
                m_classColors[classIndex] = instances[i].m_color;
            }

            instanceSegOut->addInstance(instances[i].m_id, instances[i].m_type, classIndex, instances[i].m_label, instances[i].m_confidence,
                                        instances[i].m_box[0], instances[i].m_box[1], instances[i].m_box[2], instances[i].m_box[3],
                                        instances[i].m_mask, instances[i].m_color);
        }
    }
    emit m_signalHandler->doProgress();
    endTaskRun();
}

//-----------------------------------//
//----- CInstanceSegFilterFactory -----//
//-----------------------------------//
CInstanceSegFilterFactory::CInstanceSegFilterFactory()
{
    m_info.m_name = "ik_instance_segmentation_filter";
    m_info.m_description = QObject::tr("This process filters instance segmentation results based on confidence and object category."
                                       "It can also be used for panoptic segmentation task results.").toStdString();
    m_info.m_path = QObject::tr("Core/Utils").toStdString();
    m_info.m_iconPath = QObject::tr(":/Images/default-process.png").toStdString();
    m_info.m_keywords = "instance,segmentation,filtering,measures,graphics";
}

WorkflowTaskPtr CInstanceSegFilterFactory::create(const WorkflowTaskParamPtr &pParam)
{
    auto pDerivedParam = std::dynamic_pointer_cast<CInstanceSegFilterParam>(pParam);
    if(pDerivedParam != nullptr)
        return std::make_shared<CInstanceSegFilter>(m_info.m_name, pDerivedParam);
    else
        return create();
}

WorkflowTaskPtr CInstanceSegFilterFactory::create()
{
    auto pDerivedParam = std::make_shared<CInstanceSegFilterParam>();
    assert(pDerivedParam != nullptr);
    return std::make_shared<CInstanceSegFilter>(m_info.m_name, pDerivedParam);
}

//----------------------------------//
//----- CWidgetInstanceSegFilter -----//
//----------------------------------//
CWidgetInstanceSegFilter::CWidgetInstanceSegFilter(QWidget *parent): CWorkflowTaskWidget(parent)
{
    init();
}

CWidgetInstanceSegFilter::CWidgetInstanceSegFilter(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent)
    : CWorkflowTaskWidget(parent)
{
    m_pParam = std::dynamic_pointer_cast<CInstanceSegFilterParam>(pParam);
    init();
}

void CWidgetInstanceSegFilter::init()
{
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CInstanceSegFilterParam>();

    m_pSpinConfidence = addDoubleSpin("Confidence", m_pParam->m_confidence, 0.0, 1.0, 0.1, 2);
    m_pEditCategories = addEdit("Categories", QString::fromStdString(m_pParam->m_categories));
}

void CWidgetInstanceSegFilter::onApply()
{
    m_pParam->m_confidence = m_pSpinConfidence->value();
    m_pParam->m_categories = m_pEditCategories->text().toStdString();
    emit doApplyProcess(m_pParam);
}

//-----------------------------------------//
//----- CWidgetInstanceSegFilterFactory -----//
//-----------------------------------------//
CWidgetInstanceSegFilterFactory::CWidgetInstanceSegFilterFactory()
{
    m_name = "ik_instance_segmentation_filter";
}

WorkflowTaskWidgetPtr CWidgetInstanceSegFilterFactory::create(std::shared_ptr<CWorkflowTaskParam> pParam)
{
    return std::make_shared<CWidgetInstanceSegFilter>(pParam);
}
