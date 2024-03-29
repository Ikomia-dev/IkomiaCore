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

#ifndef CTIMER_HPP
#define CTIMER_HPP

#include <string>
#include <iostream>


#if defined(__MACH__)

    #include <mach/clock.h>
    #include <mach/mach.h>

#elif (defined(linux) || defined(__linux__) || defined(__linux)) || (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__))

    #include <time.h>
    #include <sys/time.h>

#elif defined(_WIN32)

    #if defined(_MSC_VER) && !defined(NOMINMAX)
        #define NOMINMAX // Otherwise MS compilers act like idiots when using std::numeric_limits<>::max() and including windows.h
    #endif

    #include <windows.h>

#endif



// ~Nanosecond-precision cross-platform (linux/bsd/mac/windows, C++03/C++11) simple timer class:

namespace Ikomia
{
    namespace Utils
    {

// Mac OSX implementation:
#if defined(__MACH__)

    /**
     * @brief
     * This class implements a nanosecond-precision cross-platform(Linux, BSD, Mac OS X, Windows) timer class.
     */
    class CTimer
    {
    private:
        clock_serv_t system_clock;
        mach_timespec_t timeInit, timeStart, timeCurrent;
    public:
        CTimer()
        {
            host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &system_clock);
        }

        ~CTimer()
        {
            mach_port_deallocate(mach_task_self(), system_clock);
        }

        inline void start()
        {
            clock_get_time(system_clock, &timeInit);
            timeStart = timeInit;
        }
        inline void printElapsedTime_ms(std::string name)
        {
            std::cout << name << " : " << get_elapsed_ms() << " ms" << std::endl;
        }

        inline void printTotalElapsedTime_ms(std::string name)
        {
            std::cout << name << " : " << get_total_elapsed_ms() << " ms" << std::endl;
        }
        inline double get_elapsed_ms()
        {
            return get_elapsed_ns() / 1000000.0;
        }

        inline double get_total_elapsed_ms()
        {
            return get_total_elapsed_ns() / 1000000.0;
        }

        inline double get_elapsed_us()
        {
            return get_elapsed_ns() / 1000.0;
        }

        inline double get_total_elapsed_us()
        {
            return get_total_elapsed_ns() / 1000.0;
        }

        double get_elapsed_ns()
        {
            clock_get_time(system_clock, &timeCurrent);
            double val = ((1000000000.0 * static_cast<double>(timeCurrent.tv_sec - timeStart.tv_sec)) + static_cast<double>(timeCurrent.tv_nsec - timeStart.tv_nsec));
            timeStart = timeCurrent;
            return val;
        }
        double get_total_elapsed_ns()
        {
            clock_get_time(system_clock, &timeCurrent);
            return ((1000000000.0 * static_cast<double>(timeCurrent.tv_sec - timeInit.tv_sec)) + static_cast<double>(timeCurrent.tv_nsec - timeInit.tv_nsec));
        }
    };

// Linux/BSD implementation:
#elif (defined(linux) || defined(__linux__) || defined(__linux)) || (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__))

    /**
     * @brief
     * This class implements a nanosecond-precision cross-platform(Linux, BSD, Mac OS X, Windows) timer class.
     */
    class CTimer
    {
        private:
            struct timespec timeStart, timeCurrent, timeInit;
        public:
            CTimer() {}

            inline void start()
            {
                clock_gettime(CLOCK_MONOTONIC, &timeInit);
                timeStart = timeInit;
            }

            inline void printElapsedTime_ms(std::string name)
            {
                std::cout << name << " : " << get_elapsed_ms() << " ms" << std::endl;
            }

            inline void printTotalElapsedTime_ms(std::string name)
            {
                std::cout << name << " : " << get_total_elapsed_ms() << " ms" << std::endl;
            }

            inline double get_elapsed_ms()
            {
                return get_elapsed_ns() / 1000000.0;
            }

            inline double get_total_elapsed_ms()
            {
                return get_total_elapsed_ns() / 1000000.0;
            }

            inline double get_elapsed_us()
            {
                return get_elapsed_ns() / 1000.0;
            }

            inline double get_total_elapsed_us()
            {
                return get_total_elapsed_ns() / 1000.0;
            }

            double get_elapsed_ns()
            {
                clock_gettime(CLOCK_MONOTONIC, &timeCurrent);
                double t = ((1000000000.0 * static_cast<double>(timeCurrent.tv_sec - timeStart.tv_sec)) + static_cast<double>(timeCurrent.tv_nsec - timeStart.tv_nsec));
                timeStart = timeCurrent;
                return t;
            }

            double get_total_elapsed_ns()
            {
                clock_gettime(CLOCK_MONOTONIC, &timeCurrent);
                return ((1000000000.0 * static_cast<double>(timeCurrent.tv_sec - timeInit.tv_sec)) + static_cast<double>(timeCurrent.tv_nsec - timeInit.tv_nsec));
            }
    };

// Windows implementation:
#elif defined(_WIN32)

    /**
     * @brief
     * This class implements a nanosecond-precision cross-platform(Linux, BSD, Mac OS X, Windows) timer class.
     */
    class CTimer
    {
    private:
        LARGE_INTEGER ticksInit, ticksStart, ticksCurrent;
        double frequency;
    public:
        CTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            frequency = static_cast<double>(freq.QuadPart);
        }

        inline void start()
        {
            QueryPerformanceCounter(&ticksInit);
            ticksStart = ticksInit;
        }

        double get_elapsed_ms()
        {
            QueryPerformanceCounter(&ticksCurrent);
            double val = (static_cast<double>(ticksCurrent.QuadPart - ticksStart.QuadPart) * 1000.0) / frequency;
            ticksStart = ticksCurrent;
            return val;
        }

        double get_total_elapsed_ms()
        {
            QueryPerformanceCounter(&ticksCurrent);
            return (static_cast<double>(ticksCurrent.QuadPart - ticksInit.QuadPart) * 1000.0) / frequency;
        }

        inline double get_elapsed_us()
        {
            return get_elapsed_ms() * 1000.0;
        }

        inline double get_total_elapsed_us()
        {
            return get_total_elapsed_ms() * 1000.0;
        }

        inline double get_elapsed_ns()
        {
            return get_elapsed_ms() * 1000000.0;
        }

        inline double get_total_elapsed_ns()
        {
            return get_total_elapsed_ms() * 1000000.0;
        }

        inline void printElapsedTime_ms(std::string name)
        {
            std::cout << name << " : " << get_elapsed_ms() << " ms" << std::endl;
        }

        inline void printTotalElapsedTime_ms(std::string name)
        {
            std::cout << name << " : " << get_total_elapsed_ms() << " ms" << std::endl;
        }
    };
#endif
// Else: failure warning - your OS is not supported



/*#if defined(__MACH__) || (defined(linux) || defined(__linux__) || defined(__linux)) || (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) || defined(_WIN32)
inline void nanosecond_delay(double delay_ns)
{
    CTimer timer;
    timer.start();

    while(timer.get_elapsed_ns() < delay_ns)
    {};
}

inline void microsecond_delay(double delay_us)
{
    nanosecond_delay(delay_us * 1000.0);
}

inline void millisecond_delay(double delay_ms)
{
    nanosecond_delay(delay_ms * 1000000.0);
}
#endif*/

    } // namespace Utils
} // namespace Ikomia

#endif // CTIMER_HPP
