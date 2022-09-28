#ifndef CBLOBMEASUREIOWRAP_H
#define CBLOBMEASUREIOWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CBlobMeasureIO.h"

class CBlobMeasureIOWrap : public CBlobMeasureIO, public wrapper<CBlobMeasureIO>
{
    public:

        CBlobMeasureIOWrap();
        CBlobMeasureIOWrap(const std::string& name);
        CBlobMeasureIOWrap(const CBlobMeasureIO &io);

        std::string     toJson() const;
};

#endif // CBLOBMEASUREIOWRAP_H
