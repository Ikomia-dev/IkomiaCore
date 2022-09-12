#include "CObjDetectFilter.h"
#include "IO/CObjectDetectionIO.h"
#include "UtilsTools.hpp"

//---------------------------------//
//----- CObjDetectFilterParam -----//
//---------------------------------//
CObjDetectFilterParam::CObjDetectFilterParam()
{
}

void CObjDetectFilterParam::setParamMap(const UMapString &paramMap)
{
    m_confidence = std::stof(paramMap.at("confidence"));
    m_categories = paramMap.at("categories");
}

UMapString CObjDetectFilterParam::getParamMap() const
{
    UMapString params;
    params.insert(std::make_pair("confidence", std::to_string(m_confidence)));
    params.insert(std::make_pair("categories", m_categories));
    return params;
}

//----------------------------//
//----- CObjDetectFilter -----//
//----------------------------//
CObjDetectFilter::CObjDetectFilter() : CWorkflowTask()
{
    addInput(std::make_shared<CObjectDetectionIO>());
    addOutput(std::make_shared<CObjectDetectionIO>());
}

CObjDetectFilter::CObjDetectFilter(const std::string name, const std::shared_ptr<CObjDetectFilterParam> &pParam)
    : CWorkflowTask(name)
{
    m_pParam = std::make_shared<CObjDetectFilterParam>(*pParam);
    addInput(std::make_shared<CObjectDetectionIO>());
    addOutput(std::make_shared<CObjectDetectionIO>());
}

size_t CObjDetectFilter::getProgressSteps()
{
    return 2;
}

void CObjDetectFilter::run()
{
    auto objDetectIn = std::dynamic_pointer_cast<CObjectDetectionIO>(getInput(0));
    auto paramPtr = std::dynamic_pointer_cast<CObjDetectFilterParam>(m_pParam);

    if(objDetectIn == nullptr || paramPtr == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid input", __func__, __FILE__, __LINE__);

    auto objDetectOut = std::dynamic_pointer_cast<CObjectDetectionIO>(getOutput(0));
    objDetectOut->clearData();

    std::set<std::string> categories;
    if(paramPtr->m_categories != "all")
    {
        std::vector<std::string> categs;
        Utils::String::tokenize(paramPtr->m_categories, categs, ",");
        categories.insert(categs.begin(), categs.end());
    }

    emit m_signalHandler->doProgress();

    auto objects = objDetectIn->getObjects();
    for(size_t i=0; i<objects.size(); ++i)
    {
        if (objects[i].m_confidence >= paramPtr->m_confidence &&
                (categories.empty() || categories.find(objects[i].m_label) != categories.end()))
        {
            objDetectOut->addObject(objects[i].m_id, objects[i].m_label, objects[i].m_confidence,
                                    objects[i].m_box[0], objects[i].m_box[1], objects[i].m_box[2], objects[i].m_box[3],
                                    objects[i].m_color);
        }
    }

    emit m_signalHandler->doProgress();
}

//-----------------------------------//
//----- CObjDetectFilterFactory -----//
//-----------------------------------//
CObjDetectFilterFactory::CObjDetectFilterFactory()
{
    m_info.m_name = "ik_obj_detection_filter";
    m_info.m_description = QObject::tr("This process filters object detection results based on confidence and object category.").toStdString();
    m_info.m_path = QObject::tr("Core/Utils").toStdString();
    m_info.m_iconPath = QObject::tr(":/Images/default-process.png").toStdString();
    m_info.m_keywords = "object,detection,filtering,measures,graphics";
}

WorkflowTaskPtr CObjDetectFilterFactory::create(const WorkflowTaskParamPtr &pParam)
{
    auto pDerivedParam = std::dynamic_pointer_cast<CObjDetectFilterParam>(pParam);
    if(pDerivedParam != nullptr)
        return std::make_shared<CObjDetectFilter>(m_info.m_name, pDerivedParam);
    else
        return create();
}

WorkflowTaskPtr CObjDetectFilterFactory::create()
{
    auto pDerivedParam = std::make_shared<CObjDetectFilterParam>();
    assert(pDerivedParam != nullptr);
    return std::make_shared<CObjDetectFilter>(m_info.m_name, pDerivedParam);
}

//----------------------------------//
//----- CWidgetObjDetectFilter -----//
//----------------------------------//
CWidgetObjDetectFilter::CWidgetObjDetectFilter(QWidget *parent): CWorkflowTaskWidget(parent)
{
    init();
}

CWidgetObjDetectFilter::CWidgetObjDetectFilter(std::shared_ptr<CWorkflowTaskParam> pParam, QWidget *parent)
    : CWorkflowTaskWidget(parent)
{
    m_pParam = std::dynamic_pointer_cast<CObjDetectFilterParam>(pParam);
    init();
}

void CWidgetObjDetectFilter::init()
{
    if(m_pParam == nullptr)
        m_pParam = std::make_shared<CObjDetectFilterParam>();

    m_pSpinConfidence = addDoubleSpin("Confidence", m_pParam->m_confidence, 0.0, 1.0, 0.1, 2);
    m_pEditCategories = addEdit("Categories", QString::fromStdString(m_pParam->m_categories));
}

void CWidgetObjDetectFilter::onApply()
{
    m_pParam->m_confidence = m_pSpinConfidence->value();
    m_pParam->m_categories = m_pEditCategories->text().toStdString();
    emit doApplyProcess(m_pParam);
}

//-----------------------------------------//
//----- CWidgetObjDetectFilterFactory -----//
//-----------------------------------------//
CWidgetObjDetectFilterFactory::CWidgetObjDetectFilterFactory()
{
    m_name = "ik_obj_detection_filter";
}

WorkflowTaskWidgetPtr CWidgetObjDetectFilterFactory::create(std::shared_ptr<CWorkflowTaskParam> pParam)
{
    return std::make_shared<CWidgetObjDetectFilter>(pParam);
}
