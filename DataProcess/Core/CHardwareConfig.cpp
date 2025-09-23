#include "CHardwareConfig.h"


CHardwareConfig::CHardwareConfig()
{
}

int CHardwareConfig::getMinCPU() const
{
    return m_minCPU;
}

int CHardwareConfig::getMinRAM() const
{
    return m_minRAM;
}

bool CHardwareConfig::isGPURequired() const
{
    return m_bRequiredGPU;
}

int CHardwareConfig::getMinVRAM() const
{
    return m_minVRAM;
}

void CHardwareConfig::setMinCPU(int count)
{
    m_minCPU = count;
}

void CHardwareConfig::setMinRAM(int amount)
{
    m_minRAM = amount;
}

void CHardwareConfig::setGPURequired(bool required)
{
    m_bRequiredGPU = required;
}

void CHardwareConfig::setMinVRAM(int amount)
{
    m_minVRAM = amount;
}

void CHardwareConfig::to_ostream(std::ostream &os) const
{
    os << "Mininum CPU count: " << m_minCPU << std::endl;
    os << "Minimum RAM (GB): " << m_minRAM << std::endl;
    os << "GPU required: " << m_bRequiredGPU << std::endl;
    os << "Minimum VRAM (GB): " << m_minVRAM << std::endl;
}

std::ostream& operator<<(std::ostream& os, const CHardwareConfig& hwConfig)
{
    hwConfig.to_ostream(os);
    return os;
}
