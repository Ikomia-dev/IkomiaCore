#ifndef CINSTANCESEGTASK_H
#define CINSTANCESEGTASK_H

#include "C2dImageTask.h"
#include "IO/CInstanceSegIO.h"

class CInstanceSegTask: public C2dImageTask
{
    public:

        CInstanceSegTask();
        CInstanceSegTask(const std::string& name);

        void                        addInstance(int id, int type, int classIndex, double confidence,
                                                double x, double y, double width, double height,
                                                const CMat& mask);

        void                        endTaskRun() override;

        std::vector<std::string>    getNames() const;
        InstanceSegIOPtr            getResults() const;
        CMat                        getVisualizationImage() const;

        void                        readClassNames(const std::string& path);

        void                        setColors(const std::vector<CColor> &colors);
        void                        setNames(const std::vector<std::string>& names);

    private:

        void                        init();
        void                        generateRandomColors();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
};

#endif // CINSTANCESEGTASK_H
