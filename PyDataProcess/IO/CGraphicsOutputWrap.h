#ifndef CGRAPHICSOUTPUTWRAP_H
#define CGRAPHICSOUTPUTWRAP_H

#include "PyDataProcessGlobal.h"
#include "IO/CGraphicsOutput.h"

class CGraphicsOutputWrap: public CGraphicsOutput, public wrapper<CGraphicsOutput>
{
    public:

        CGraphicsOutputWrap();
        CGraphicsOutputWrap(const std::string& name);
        CGraphicsOutputWrap(const CGraphicsOutput &io);

        std::string     toJson() const;
};

#endif // CGRAPHICSOUTPUTWRAP_H
