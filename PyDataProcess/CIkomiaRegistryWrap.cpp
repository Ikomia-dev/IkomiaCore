#include "CIkomiaRegistryWrap.h"

CIkomiaRegistryWrap::CIkomiaRegistryWrap(): CIkomiaRegistry()
{
}

CIkomiaRegistryWrap::CIkomiaRegistryWrap(const CIkomiaRegistry &reg): CIkomiaRegistry(reg)
{
}

CTaskInfo CIkomiaRegistryWrap::getAlgorithmInfo(const std::string &name) const
{
    CPyEnsureGIL gil;
    return CIkomiaRegistry::getAlgorithmInfo(name);
}
