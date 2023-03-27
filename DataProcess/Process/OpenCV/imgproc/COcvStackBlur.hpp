#ifndef COCVSTACKBLUR_HPP
#define COCVSTACKBLUR_HPP

#include "Task/C2dImageTask.h"
#include "IO/CImageIO.h"

//------------------------------//
//----- COcvStackBlurParam -----//
//------------------------------//
class COcvStackBlurParam: public CWorkflowTaskParam
{
    public:

        COcvStackBlurParam() : CWorkflowTaskParam(){}

        void        setParamMap(const UMapString& paramMap) override
        {
            m_ksize.width = std::stoi(paramMap.at("kSizeWidth"));
            m_ksize.height = std::stoi(paramMap.at("kSizeHeight"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("kSizeWidth", std::to_string(m_ksize.width)));
            map.insert(std::make_pair("kSizeHeight", std::to_string(m_ksize.height)));
            return map;
        }

    public:

        cv::Size    m_ksize = cv::Size(5, 5);
};

//-------------------------//
//----- COcvStackBlur -----//
//-------------------------//
class COcvStackBlur : public C2dImageTask
{
    public:

        COcvStackBlur() : C2dImageTask()
        {
        }
        COcvStackBlur(const std::string name, const std::shared_ptr<COcvStackBlurParam>& pParam) : C2dImageTask(name)
        {
            m_pParam = std::make_shared<COcvStackBlurParam>(*pParam);
        }

        size_t  getProgressSteps() override
        {
            return 3;
        }

        void run() override
        {
            beginTaskRun();

            auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
            if(pInput == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

            auto pParam = std::dynamic_pointer_cast<COcvStackBlurParam>(m_pParam);
            if(pParam == nullptr)
                throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

            if(pInput->isDataAvailable() == false)
                throw CException(CoreExCode::INVALID_PARAMETER, "Empty source image", __func__, __FILE__, __LINE__);

            CMat imgDst;
            CMat imgSrc = pInput->getImage();
            auto pGraphicsInput = std::dynamic_pointer_cast<CGraphicsInput>(getInput(1));
            createGraphicsMask(imgSrc.getNbCols(), imgSrc.getNbRows(), pGraphicsInput);
            emit m_signalHandler->doProgress();

            try
            {
                cv::stackBlur(imgSrc, imgDst, pParam->m_ksize);
            }
            catch(cv::Exception& e)
            {
                throw CException(CoreExCode::INVALID_PARAMETER, e, __func__, __FILE__, __LINE__);
            }

            endTaskRun();
            applyGraphicsMask(imgSrc, imgDst, 0);
            emit m_signalHandler->doProgress();

            auto pOutput = std::dynamic_pointer_cast<CImageIO>(getOutput(0));
            if(pOutput)
                pOutput->setImage(imgDst);

            emit m_signalHandler->doProgress();
        }
};

class COcvStackBlurFactory : public CTaskFactory
{
    public:

        COcvStackBlurFactory()
        {
            m_info.m_name = "ocv_stack_blur";
            m_info.m_description = QObject::tr("This process smoothes your image with similar results as Gaussian blur.").toStdString();
            m_info.m_path = QObject::tr("OpenCV/Main modules/Image processing/Image filtering").toStdString();
            m_info.m_iconPath = QObject::tr(":/Images/opencv.png").toStdString();
            m_info.m_keywords = "Smooth,Blur,Isotropic,Filter,Gaussian";
            m_info.m_docLink = "https://docs.opencv.org/4.7.0/d4/d86/group__imgproc__filter.html#ga13a01048a8a200aab032ce86a9e7c7be";
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto pStackBlurParam = std::dynamic_pointer_cast<COcvStackBlurParam>(pParam);
            if(pStackBlurParam != nullptr)
                return std::make_shared<COcvStackBlur>(m_info.m_name, pStackBlurParam);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto pStackBlurParam = std::make_shared<COcvStackBlurParam>();
            assert(pStackBlurParam != nullptr);
            return std::make_shared<COcvStackBlur>(m_info.m_name, pStackBlurParam);
        }
};

#endif // COCVSTACKBLUR_HPP
