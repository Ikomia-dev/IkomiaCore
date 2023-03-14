#ifndef CKEYPOINTDETECTIONTASK_H
#define CKEYPOINTDETECTIONTASK_H

#include "C2dImageTask.h"
#include "IO/CKeypointsIO.h"


class DATAPROCESSSHARED_EXPORT CKeypointDetectionTask: public C2dImageTask
{
    public:

        CKeypointDetectionTask();
        CKeypointDetectionTask(const std::string& name);

        void                        addObject(int id, int classIndex, double confidence,
                                              double x, double y, double width, double height,
                                              const std::vector<Keypoint> keypts);

        void                        endTaskRun() override;

        std::vector<CKeypointLink>  getKeypointLinks() const;
        std::vector<std::string>    getKeypointNames() const;
        std::vector<std::string>    getObjectNames() const;
        KeypointsIOPtr              getResults() const;
        CMat                        getVisualizationImage() const;

        void                        readClassNames(const std::string& path);

        void                        setKeypointLinks(const std::vector<CKeypointLink>& links);
        void                        setKeypointNames(const std::vector<std::string>& names);
        void                        setObjectColors(const std::vector<CColor>& colors);
        void                        setObjectNames(const std::vector<std::string>& names);

    private:

        void                        initIO();
        void                        generateRandomColors();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
        std::vector<std::string>    m_keyptsNames;
        std::vector<CKeypointLink>  m_keyptsLinks;
};

#endif // CKEYPOINTDETECTIONTASK_H
