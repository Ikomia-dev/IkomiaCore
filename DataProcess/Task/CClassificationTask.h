#ifndef CCLASSIFICATIONTASK_H
#define CCLASSIFICATIONTASK_H

#include "C2dImageTask.h"
#include "IO/CObjectDetectionIO.h"

class DATAPROCESSSHARED_EXPORT CClassificationTask: public C2dImageTask
{
    public:

        CClassificationTask();
        CClassificationTask(const std::string& name);

        std::vector<std::string>            getNames() const;
        std::vector<ProxyGraphicsItemPtr>   getInputObjects() const;
        CMat                                getObjectSubImage(const ProxyGraphicsItemPtr& objectPtr) const;
        ObjectDetectionIOPtr                getObjectsResults() const;
        std::vector<PairString>             getWholeImageResults() const;
        CMat                                getVisualizationImage() const;

        bool                                isWholeImageClassification() const;

        void                                setNames(const std::vector<std::string>& names);
        void                                setColors(const std::vector<CColor>& colors);
        void                                setWholeImageResults(const std::vector<std::string>& sortedNames, const std::vector<std::string> &sortedConfidences);

        void                                readClassNames(const std::string& path);
        void                                addObject(const ProxyGraphicsItemPtr& objectPtr, int classIndex, double confidence);
        void                                endTaskRun() override;

    private:

        void                                initIO();
        void                                generateRandomColors();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
};

#endif // CCLASSIFICATIONTASK_H
