#ifndef UTILSDEFINE_HPP
#define UTILSDEFINE_HPP

namespace Ikomia
{
    enum class PluginState : size_t
    {
        VALID,
        DEPRECATED,
        INVALID
    };

    /**
     * @enum OS
     * @brief The OS enum defines the possible operating systems.
     */
    enum OSType
    {
        ALL,        /**< Cross-platform */
        LINUX,      /**< Linux */
        WIN,        /**< Windows 10 */
        OSX,         /**< Mac OS X 10.13 or higher */
        UNDEFINED  /**< Undefined */
    };

    /**
     * @enum Language
     * @brief The Language enum defines the possible programming languages.
     */
    enum ApiLanguage
    {
        CPP,    /**< C++ */
        PYTHON  /**< Python */
    };

    enum CpuArch
    {
        X86_64,
        ARM_64,
        ARM_32,
        NOT_SUPPORTED
    };

    enum License
    {
        CUSTOM,
        AGPL_30,
        APACHE_20,
        BSD_2_CLAUSE,
        BSD_3_CLAUSE,
        CC0_10,
        CC_BY_NC_40,
        GPL_30,
        LGPL_30,
        MIT
    };
}

#endif // UTILSDEFINE_HPP
