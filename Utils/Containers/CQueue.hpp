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

#ifndef CQUEUE_HPP
#define CQUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include "CException.h"

template <typename T>
class CQueue
{
    public:

        T       pop()
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            while (m_queue.empty())
            {
                if (m_cancelled)
                    throw CException(CoreExCode::PROCESS_CANCELLED, "Threaded queue cancelled.");

                if(m_cond.wait_for(mlock, std::chrono::milliseconds(m_timeout)) == std::cv_status::timeout)
                    throw CException(CoreExCode::TIMEOUT_REACHED, "Threaded queue timeout.");

                if (m_cancelled)
                    throw CException(CoreExCode::PROCESS_CANCELLED, "Threaded queue cancelled");
            }
            auto item = m_queue.front();
            m_queue.pop();
            return item;
        }
        void    pop(T& item)
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            while (m_queue.empty())
            {
                if (m_cancelled)
                    throw CException(CoreExCode::PROCESS_CANCELLED, "Threaded queue cancelled.");

                if(m_cond.wait_for(mlock, std::chrono::milliseconds(m_timeout)) == std::cv_status::timeout)
                    throw CException(CoreExCode::TIMEOUT_REACHED, "Threaded queue timeout.");

                if (m_cancelled)
                    throw CException(CoreExCode::PROCESS_CANCELLED, "Threaded queue cancelled.");
            }
            item = m_queue.front();
            m_queue.pop();
        }
        void    push(T const& item)
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            m_queue.push(item);
            mlock.unlock();
            m_cond.notify_one();
        }
        void    push(T&& item)
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            m_queue.push(std::move(item));
            mlock.unlock();
            m_cond.notify_one();
        }

        size_t  size() const
        {
            return m_queue.size();;
        }
        void    clear()
        {
            std::queue<T> empty;
            std::unique_lock<std::mutex> mlock(m_mutex);
            std::swap( m_queue, empty );
            mlock.unlock();
            m_cond.notify_one();
        }
        void    cancel()
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            m_cancelled = true;
            m_cond.notify_all();
        }
        void    activate()
        {
            std::unique_lock<std::mutex> mlock(m_mutex);
            m_cancelled = false;
            m_cond.notify_all();
        }
        void    setTimeout(int ms)
        {
            m_timeout = ms;
        }

    private:

        std::queue<T>           m_queue;
        std::mutex              m_mutex;
        std::condition_variable m_cond;
        bool                    m_cancelled = false;
        int                     m_timeout = 5000; // In milliseconds
};

#endif // CQUEUE_HPP
