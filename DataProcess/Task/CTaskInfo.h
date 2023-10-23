/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the Ikomia API libraries.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#ifndef CTASKINFO_H
#define CTASKINFO_H

#include <string>
#include "DataProcessGlobal.hpp"
#include "UtilsDefine.hpp"
#include "Main/CoreDefine.hpp"

using namespace Ikomia;

/**
 * @ingroup groupDataProcess
 * @brief The CTaskInfo class manages metadata associated with a process task.
 * Information are then available for consulting in the software.
 * These metadata are also used by the system search engine (process library and store).
 */
class DATAPROCESSSHARED_EXPORT CTaskInfo
{
    public:

        /**
         * @brief Default constructor.
         */
        CTaskInfo();

        std::string getName() const;
        std::string getPath() const;
        std::string getShortDescription() const;
        std::string getDescription() const;
        std::string getDocumentationLink() const;
        std::string getIconPath() const;
        std::string getKeywords() const;
        std::string getAuthors() const;
        std::string getArticle() const;
        std::string getArticleUrl() const;
        std::string getJournal() const;
        std::string getVersion() const;
        std::string getMinIkomiaVersion() const;
        std::string getMaxIkomiaVersion() const;
        std::string getMinPythonVersion() const;
        std::string getMaxPythonVersion() const;
        std::string getLicense() const;
        std::string getRepository() const;
        std::string getOriginalRepository() const;
        int         getYear() const;
        ApiLanguage getLanguage() const;
        OSType      getOS() const;
        AlgoType    getAlgoType() const;
        std::string getAlgoTasks() const;

        /** @cond INTERNAL */
        bool        isInternal() const;
        /** @endcond */

        void        setName(const std::string& name);
        void        setPath(const std::string& path);
        void        setShortDescription(const std::string& description);
        void        setDescription(const std::string& description);
        void        setDocumentationLink(const std::string& link);
        void        setIconPath(const std::string& path);
        void        setKeywords(const std::string& keywords);
        void        setAuthors(const std::string& authors);
        void        setArticle(const std::string& article);
        void        setArticleUrl(const std::string& url);
        void        setJournal(const std::string& journal);
        void        setYear(const int year);
        void        setVersion(const std::string& version);
        void        setMinIkomiaVersion(const std::string& version);
        void        setMaxIkomiaVersion(const std::string& version);
        void        setMinPythonVersion(const std::string& version);
        void        setMaxPythonVersion(const std::string& version);
        void        setLicense(const std::string& license);
        void        setRepository(const std::string& repository);
        void        setOriginalRepository(const std::string& repository);
        void        setLanguage(const ApiLanguage& language);
        void        setOS(const OSType& os);
        void        setAlgoType(const AlgoType& type);
        void        setAlgoTasks(const std::string& tasks);

        friend DATAPROCESSSHARED_EXPORT std::ostream& operator<<(std::ostream& os, const CTaskInfo& info);

    protected:

        virtual void    to_ostream(std::ostream& os) const;

    public:

        /** @cond INTERNAL */
        int         m_id;
        bool        m_bInternal = true;
        /** @endcond */

        std::string m_name;                         /**< Process task name. Must be unique */
        std::string m_path;                         /**< Path in the system tree structure of the process library */
        std::string m_shortDescription;             /**< Short description of the process */
        std::string m_description;                  /**< Full description of the process - deprecated */
        std::string m_docLink;                      /**< Internet link to an associated documentation page */
        std::string m_iconPath;                     /**< File path to a custom icon */
        std::string m_keywords;                     /**< Keywords associated with the process: useful for search engine */
        std::string m_authors;                      /**< Authors of the process */
        std::string m_article;                      /**< Associated research article */
        std::string m_articleUrl;                   /**< Url of artivle */
        std::string m_journal;                      /**< Journal of the article */
        int         m_year = -1;                    /**< Year of the article or the algorithme */
        std::string m_version = "1.0.0";            /**< Version of the implementation */
        std::string m_minIkomiaVersion = "0.10.0";   /**< Minimum version of the Ikomia Core & API */
        std::string m_maxIkomiaVersion;             /**< Maximum version of the Ikomia Core & API */
        std::string m_minPythonVersion = "3.7";     /**< Minimum compatible Python version */
        std::string m_maxPythonVersion = "3.11";    /**< Maximum compatible Python version */
        std::string m_license;                      /**< Algorithm licence */
        std::string m_repo;                         /**< Implementation repository */
        std::string m_originalRepo;                 /**< Original repository */
        std::string m_createdDate;
        std::string m_modifiedDate;
        ApiLanguage m_language = ApiLanguage::CPP;  /**< Programming language */
        OSType      m_os = OSType::LINUX;           /**< Compatible operating system */
        AlgoType    m_algoType = AlgoType::OTHER;   /**< Type of algorithm */
        std::string m_algoTasks;                    /**< Type of tasks adressed: CLASSIFICATION, OBJECT_DETECTION... */
};

#endif // CTASKINFO_H
