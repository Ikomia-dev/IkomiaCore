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

#ifndef CCOLORPUSHBUTTON_H
#define CCOLORPUSHBUTTON_H

#include <QPushButton>
#include "UtilsGlobal.hpp"

class UTILSSHARED_EXPORT CColorPushButton : public QPushButton
{
    Q_OBJECT

    public:

        CColorPushButton(const QString &title, const QColor& color, QWidget* parent = nullptr);

        QColor  getColor() const;

        void    setColor(const QColor& color);

    signals:

        void    colorChanged(const QColor& color);

    private:

        void    init();
        void    initConnections();

        void    updateColor();

    private:

        QColor  m_color = Qt::red;
        QString m_title;
        const QString m_styleSheet = "QPushButton { background: %1; border: none; } "
                                     "QPushButton:hover {border: 2px solid %2; } "
                                     "QPushButton:!enabled { background: transparent; border: 2px solid %2; }";
};

#endif // CCOLORPUSHBUTTON_H
