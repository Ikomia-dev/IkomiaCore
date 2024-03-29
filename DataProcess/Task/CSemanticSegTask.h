#ifndef CSEMANTICSEGMENTATIONTASK_H
#define CSEMANTICSEGMENTATIONTASK_H

#include "C2dImageTask.h"
#include "IO/CSemanticSegIO.h"

class DATAPROCESSSHARED_EXPORT CSemanticSegTask: public C2dImageTask
{
    public:

        CSemanticSegTask();
        CSemanticSegTask(const std::string& name);

        std::string                 repr() const;

        void                        endTaskRun() override;

        std::vector<std::string>    getNames() const;
        SemanticSegIOPtr            getResults() const;
        CMat                        getImageWithMask() const;
        CMat                        getImageWithGraphics() const;
        CMat                        getImageWithMaskAndGraphics() const;

        void                        readClassNames(const std::string& path);

        void                        setColors(const std::vector<CColor> &colors);
        void                        setNames(const std::vector<std::string>& names);
        void                        setMask(const CMat& mask);

    private:

        void                        init();

    protected:

        std::vector<std::string>    m_classNames;
        std::vector<CColor>         m_classColors;
};

#endif // CSEMANTICSEGMENTATIONTASK_H
