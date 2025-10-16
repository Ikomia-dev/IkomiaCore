#ifndef CHARDWARECONFIG_H
#define CHARDWARECONFIG_H

#include <QJsonObject>
#include "DataProcessGlobal.hpp"


class DATAPROCESSSHARED_EXPORT CHardwareConfig
{
    public:

        CHardwareConfig();

        int     getMinCPU() const;
        int     getMinRAM() const;
        bool    isGPURequired() const;
        int     getMinVRAM() const;

        void    setMinCPU(int count);
        void    setMinRAM(int amount);
        void    setGPURequired(bool required);
        void    setMinVRAM(int amount);

        QJsonObject toJson() const;

    protected:

        virtual void to_ostream(std::ostream& os) const;

        friend DATAPROCESSSHARED_EXPORT std::ostream& operator<<(std::ostream& os, const CHardwareConfig& config);

    public:

        // Min CPU count
        int     m_minCPU = 2;
        // Min RAM in GB
        int     m_minRAM = 8;
        // GPU required
        bool    m_bRequiredGPU = false;
        // Min VRAM in GB
        int     m_minVRAM = 0;
};

#endif // CHARDWARECONFIG_H
