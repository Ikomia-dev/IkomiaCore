#include "CSemanticVersion.h"
#include <istream>
#include <sstream>
#include <iterator>
#include <algorithm>

CSemanticVersion::CSemanticVersion(const std::string &version)
{
    m_version = version;

    // To Make processing easier in VersionDigit prepend a '.'
    std::stringstream versionStream("." + version);

    // Copy all parts of the version number into the version Info vector.
    m_versionParts.assign(std::istream_iterator<VersionDigit>(versionStream),
                          std::istream_iterator<VersionDigit>());
}

bool CSemanticVersion::operator<(const CSemanticVersion &rhs) const
{
    return std::lexicographical_compare(m_versionParts.begin(), m_versionParts.end(), rhs.m_versionParts.begin(), rhs.m_versionParts.end());
}

bool CSemanticVersion::operator>(const CSemanticVersion &rhs) const
{
    return rhs < *this;
}

bool CSemanticVersion::operator<=(const CSemanticVersion &rhs) const
{
    return !(*this < rhs);
}

bool CSemanticVersion::operator>=(const CSemanticVersion &rhs) const
{
    return !(rhs < *this);
}

bool CSemanticVersion::operator==(const CSemanticVersion &rhs) const
{
    return std::equal(m_versionParts.begin(), m_versionParts.end(), rhs.m_versionParts.begin(), rhs.m_versionParts.end());
}

bool CSemanticVersion::operator!=(const CSemanticVersion &rhs) const
{
    return !(*this == rhs);
}

std::istream& operator>>(std::istream& str, CSemanticVersion::VersionDigit& digit)
{
    str.get();
    str >> digit.m_value;
    return str;
}
