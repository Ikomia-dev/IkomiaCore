/*
 * Copyright (C) 2021 Ikomia SAS
 * Contact: https://www.ikomia.com
 *
 * This file is part of the IkomiaStudio software.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CGRAPHICSREGISTRATION_H
#define CGRAPHICSREGISTRATION_H

#include "CGraphicsItem.hpp"

class CORESHARED_EXPORT CGraphicsRegistration
{
    public:

        CGraphicsRegistration();

        const CGraphicsAbstractFactory&         getQtBasedFactory() const;
        const CProxyGraphicsAbstractFactory&    getProxyFactory() const;

        void                                    registerItem(const std::shared_ptr<CGraphicsFactory>& pFactory);
        void                                    registerItem(const std::shared_ptr<CProxyGraphicsFactory>& pFactory);

    private:

        CGraphicsAbstractFactory        m_factory;
        CProxyGraphicsAbstractFactory   m_proxyFactory;
};

#endif // CGRAPHICSREGISTRATION_H
