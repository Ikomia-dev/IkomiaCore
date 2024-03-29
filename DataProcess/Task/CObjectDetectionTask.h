#ifndef COBJECTDETECTIONTASK_H
#define COBJECTDETECTIONTASK_H

#include "C2dImageTask.h"
#include "IO/CObjectDetectionIO.h"

class DATAPROCESSSHARED_EXPORT CObjectDetectionTask: public C2dImageTask
{
    public:

        CObjectDetectionTask();
        CObjectDetectionTask(const std::string& name);

        std::string                         repr() const;

        void                                addObject(int id, int classIndex, double confidence,
                                                      double boxX, double boxY, double boxWidth, double boxHeight);
        void                                addObject(int id, int classIndex, double confidence,
                                                      double cx, double cy, double width, double height, double angle);

        void                                endTaskRun() override;

        std::vector<std::string>            getNames() const;
        ObjectDetectionIOPtr                getResults() const;
        CMat                                getImageWithGraphics() const;

        void                                readClassNames(const std::string& path);

        void                                setColors(const std::vector<CColor>& colors);
        void                                setNames(const std::vector<std::string>& names);

    private:

        void                                initIO();
        void                                generateRandomColors();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
};

#endif // COBJECTDETECTIONTASK_H
