#ifndef CSEMANTICSEGMENTATIONTASK_H
#define CSEMANTICSEGMENTATIONTASK_H

#include "C2dImageTask.h"
#include "IO/CSemanticSegIO.h"

class CSemanticSegTask: public C2dImageTask
{
    public:

        CSemanticSegTask();
        CSemanticSegTask(const std::string& name);

        void                            endTaskRun() override;

        std::vector<std::string>        getNames() const;
        std::shared_ptr<CSemanticSegIO> getResults() const;
        CMat                            getColorMaskImage() const;

        void                            readClassNames(const std::string& path);

        void                            setColors(const std::vector<cv::Vec3b>& colors);
        void                            setNames(const std::vector<std::string>& names);
        void                            setMask(const CMat& mask);

    private:

        void                            init();
};

#endif // CSEMANTICSEGMENTATIONTASK_H
