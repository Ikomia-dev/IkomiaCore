// Copyright (C) 2021 Ikomia SAS
// Contact: https://www.ikomia.com
//
// This file is part of the Ikomia API libraries.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include "CTaskInfo.h"
#include "UtilsTools.hpp"
#include "Main/CoreTools.hpp"

CTaskInfo::CTaskInfo()
{
    m_minIkomiaVersion = Utils::IkomiaApp::getCurrentVersionNumber();
}

std::string CTaskInfo::getName() const
{
    return m_name;
}

std::string CTaskInfo::getPath() const
{
    return m_path;
}

std::string CTaskInfo::getShortDescription() const
{
    return m_shortDescription;
}

std::string CTaskInfo::getDescription() const
{
    return m_description;
}

std::string CTaskInfo::getDocumentationLink() const
{
    return m_docLink;
}

std::string CTaskInfo::getIconPath() const
{
    return m_iconPath;
}

std::string CTaskInfo::getKeywords() const
{
    return m_keywords;
}

std::string CTaskInfo::getAuthors() const
{
    return m_authors;
}

std::string CTaskInfo::getArticle() const
{
    return m_article;
}

std::string CTaskInfo::getArticleUrl() const
{
    return m_articleUrl;
}

std::string CTaskInfo::getJournal() const
{
    return m_journal;
}

std::string CTaskInfo::getVersion() const
{
    return m_version;
}

std::string CTaskInfo::getMinIkomiaVersion() const
{
    return m_minIkomiaVersion;
}

std::string CTaskInfo::getMaxIkomiaVersion() const
{
    return m_maxIkomiaVersion;
}

std::string CTaskInfo::getMinPythonVersion() const
{
    return m_minPythonVersion;
}

std::string CTaskInfo::getMaxPythonVersion() const
{
    return m_maxPythonVersion;
}

std::string CTaskInfo::getLicense() const
{
    return m_license;
}

std::string CTaskInfo::getRepository() const
{
    return m_repo;
}

std::string CTaskInfo::getOriginalRepository() const
{
    return m_originalRepo;
}

int CTaskInfo::getYear() const
{
    return m_year;
}

ApiLanguage CTaskInfo::getLanguage() const
{
    return m_language;
}

OSType CTaskInfo::getOS() const
{
    return m_os;
}

AlgoType CTaskInfo::getAlgoType() const
{
    return m_algoType;
}

std::string CTaskInfo::getAlgoTasks() const
{
    return m_algoTasks;
}

CHardwareConfig &CTaskInfo::getHardwareConfig()
{
    return m_minHardwareConfig;
}

bool CTaskInfo::isInternal() const
{
    return m_bInternal;
}

void CTaskInfo::setName(const std::string &name)
{
    m_name = name;
}

void CTaskInfo::setPath(const std::string &path)
{
    m_path = path;
}

void CTaskInfo::setShortDescription(const std::string &description)
{
    m_shortDescription = description;
}

void CTaskInfo::setDescription(const std::string &description)
{
    Utils::deprecationWarning(m_name + ": description field is deprecated", "", QtDebugMsg);
    m_description = description;
}

void CTaskInfo::setDocumentationLink(const std::string &link)
{
    m_docLink = link;
}

void CTaskInfo::setIconPath(const std::string &path)
{
    m_iconPath = path;
}

void CTaskInfo::setKeywords(const std::string &keywords)
{
    m_keywords = keywords;
}

void CTaskInfo::setAuthors(const std::string &authors)
{
    m_authors = authors;
}

void CTaskInfo::setArticle(const std::string &article)
{
    m_article = article;
}

void CTaskInfo::setArticleUrl(const std::string &url)
{
    m_articleUrl = url;
}

void CTaskInfo::setJournal(const std::string &journal)
{
    m_journal = journal;
}

void CTaskInfo::setYear(const int year)
{
    m_year = year;
}

void CTaskInfo::setVersion(const std::string &version)
{
    m_version = version;
}

void CTaskInfo::setMinIkomiaVersion(const std::string &version)
{
    m_minIkomiaVersion = version;
}

void CTaskInfo::setMaxIkomiaVersion(const std::string &version)
{
    m_maxIkomiaVersion = version;
}

void CTaskInfo::setMinPythonVersion(const std::string &version)
{
    m_minPythonVersion = version;
}

void CTaskInfo::setMaxPythonVersion(const std::string &version)
{
    m_maxPythonVersion = version;
}

void CTaskInfo::setLicense(const std::string& license)
{
    m_license = license;
}

void CTaskInfo::setRepository(const std::string& repository)
{
    m_repo = repository;
}

void CTaskInfo::setOriginalRepository(const std::string &repository)
{
    m_originalRepo = repository;
}

void CTaskInfo::setLanguage(const ApiLanguage &language)
{
    m_language = language;
}

void CTaskInfo::setOS(const OSType &os)
{
    m_os = os;
}

void CTaskInfo::setAlgoType(const AlgoType &type)
{
    m_algoType = type;
}

void CTaskInfo::setAlgoTasks(const std::string &tasks)
{
    m_algoTasks = tasks;
}

void CTaskInfo::setHardwareConfig(const CHardwareConfig &config)
{
    m_minHardwareConfig = config;
}

std::ostream& operator<<(std::ostream& os, const CTaskInfo& info)
{
    info.to_ostream(os);
    return os;
}

void CTaskInfo::to_ostream(std::ostream &os) const
{
    os << "Name: " << m_name << std::endl;
    os << "Path: " << m_path << std::endl;
    os << "Short description: " << m_shortDescription << std::endl;
    os << "Description: " << m_description << std::endl;
    os << "Documentation link: " << m_docLink << std::endl;
    os << "Icon path: " << m_iconPath << std::endl;
    os << "Keywords: " << m_keywords << std::endl;
    os << "Authors: " << m_authors << std::endl;
    os << "Article: " << m_article << std::endl;
    os << "Article link: " << m_articleUrl << std::endl;
    os << "Journal/conference: " << m_journal << std::endl;
    os << "Year: " << std::to_string(m_year) << std::endl;
    os << "Type: " << Utils::Plugin::getAlgoTypeString(m_algoType) << std::endl;
    os << "Tasks: " << m_algoTasks << std::endl;
    os << "Version: " << m_version << std::endl;
    os << "Ikomia minimum version: " << m_minIkomiaVersion << std::endl;
    os << "Python minimum version: " << m_minPythonVersion << std::endl;
    os << "License: " << m_license << std::endl;
    os << "Repository: " << m_repo << std::endl;
    std::string language = m_language == ApiLanguage::CPP ? "C++" : "Python";
    os << "Language: " << language << std::endl;
    os << "OS: " << Utils::OS::getName(m_os) << std::endl;
    os << "Hardware configuration: " << m_minHardwareConfig << std::endl;
}
