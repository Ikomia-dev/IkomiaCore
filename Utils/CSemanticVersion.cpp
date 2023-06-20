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

    // Ensure major.minor.patch format
    if (m_versionParts.size() < 3)
    {
        for (size_t i=m_versionParts.size()-1; i<3; ++i)
            m_versionParts.push_back(0);
    }
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
    return !(rhs < *this);
}

bool CSemanticVersion::operator>=(const CSemanticVersion &rhs) const
{
    return !(*this < rhs);
}

bool CSemanticVersion::operator==(const CSemanticVersion &rhs) const
{
    return std::equal(m_versionParts.begin(), m_versionParts.end(), rhs.m_versionParts.begin(), rhs.m_versionParts.end());
}

bool CSemanticVersion::operator!=(const CSemanticVersion &rhs) const
{
    return !(*this == rhs);
}

void CSemanticVersion::nextMajor()
{
    m_versionParts[0] += 1;

    for (size_t i=1; i< m_versionParts.size(); ++i)
        m_versionParts[i] = 0;

    updateString();
}

void CSemanticVersion::nextMinor()
{
    m_versionParts[1] += 1;

    for (size_t i=2; i< m_versionParts.size(); ++i)
        m_versionParts[i] = 0;

    updateString();
}

void CSemanticVersion::nextPatch()
{
    m_versionParts[2] += 1;

    for (size_t i=3; i< m_versionParts.size(); ++i)
        m_versionParts[i] = 0;

    updateString();
}

std::string CSemanticVersion::toString() const
{
    return m_version;
}

void CSemanticVersion::updateString()
{
    m_version = std::to_string(m_versionParts[0]);
    for (size_t i=1; i<m_versionParts.size(); ++i)
        m_version += "." + std::to_string(m_versionParts[i]);
}

std::istream& operator>>(std::istream& str, CSemanticVersion::VersionDigit& digit)
{
    str.get();
    str >> digit.m_value;
    return str;
}
