#ifndef COBJECTDETECTIONTASK_H
#define COBJECTDETECTIONTASK_H

#include "C2dImageTask.h"


class CObjectDetectionTask: public C2dImageTask
{
    public:

        CObjectDetectionTask();
        CObjectDetectionTask(const std::string& name);

        void                                addObject(int id, int classIndex, double confidence,
                                                      double boxX, double boxY, double boxWidth, double boxHeight);
        void                                addObject(int id, int classIndex, double confidence,
                                                      double cx, double cy, double width, double height, double angle);

        void                                endTaskRun() override;

        std::vector<std::string>            getNames() const;
        std::shared_ptr<CObjectDetectionIO> getResults() const;


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
