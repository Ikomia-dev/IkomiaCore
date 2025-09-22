#include "CTaskHardwareConfig.h"


CTaskHardwareConfig::CTaskHardwareConfig()
{
}

int CTaskHardwareConfig::getMinCPU() const
{
    return m_minCPU;
}

int CTaskHardwareConfig::getMinRAM() const
{
    return m_minRAM;
}

bool CTaskHardwareConfig::isGPURequired() const
{
    return m_bRequiredGPU;
}

int CTaskHardwareConfig::getMinVRAM() const
{
    return m_minVRAM;
}

void CTaskHardwareConfig::setMinCPU(int count)
{
    m_minCPU = count;
}

void CTaskHardwareConfig::setMinRAM(int amount)
{
    m_minRAM = amount;
}

void CTaskHardwareConfig::setGPURequired(bool required)
{
    m_bRequiredGPU = required;
}

void CTaskHardwareConfig::setMinVRAM(int amount)
{
    m_minVRAM = amount;
}

void CTaskHardwareConfig::to_ostream(std::ostream &os) const
{
    os << "Mininum CPU count: " << m_minCPU << std::endl;
    os << "Minimum RAM (GB): " << m_minRAM << std::endl;
    os << "GPU required: " << m_bRequiredGPU << std::endl;
    os << "Minimum VRAM (GB): " << m_minVRAM << std::endl;
}

std::ostream& operator<<(std::ostream& os, const CTaskHardwareConfig& hwConfig)
{
    hwConfig.to_ostream(os);
    return os;
}
