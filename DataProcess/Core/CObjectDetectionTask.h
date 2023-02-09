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

        void                                endTaskRun() override;

        std::vector<std::string>            getNames() const;
        std::shared_ptr<CObjectDetectionIO> getResults() const;


        void                                readClassNames(const std::string& path);

        void                                setColors(const std::vector<CColor>& colors);

    private:

        void                                initIO();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
};

#endif // COBJECTDETECTIONTASK_H
