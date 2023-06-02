#ifndef CSEMANTICVERSION_H
#define CSEMANTICVERSION_H

#include <vector>
#include <string>

class CSemanticVersion
{
    struct VersionDigit
    {
        operator int() const
        {
            return m_value;
        }

        int m_value;
    };

    public:

        CSemanticVersion(const std::string& version);

        bool operator<(const CSemanticVersion& rhs) const;
        bool operator>(const CSemanticVersion& rhs) const;
        bool operator<=(const CSemanticVersion& rhs) const;
        bool operator>=(const CSemanticVersion& rhs) const;
        bool operator==(const CSemanticVersion& rhs) const;
        bool operator!=(const CSemanticVersion& rhs) const;

        friend std::istream& operator>>(std::istream& str, VersionDigit& digit);

    private:

        std::string         m_version;
        std::vector<int>    m_versionParts;
};

#endif // CSEMANTICVERSION_H
