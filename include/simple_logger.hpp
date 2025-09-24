#ifndef SIMPLE_LOGGER_HPP
#define SIMPLE_LOGGER_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

// Logger muy simple, sin dependencias externas
class SimpleLogger
{
public:
    enum Level
    {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };

private:
    static std::string get_timestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    static std::string level_to_string(Level level)
    {
        switch (level)
        {
        case DEBUG:
            return "[DEBUG]";
        case INFO:
            return "[INFO " + std::string(__FILE__) + "]";
        case WARNING:
            return "[WARN] ";
        case ERROR:
            return "[ERROR]";
        }
        return "[UNKNOWN]";
    }

public:
    static Level &get_current_level()
    {
        static Level current_level = INFO;
        return current_level;
    }

    static void set_level(Level level)
    {
        get_current_level() = level;
    }

    template <typename... Args>
    static void log(Level level, Args &&...args)
    {
        if (level < get_current_level())
            return;

        std::ostream &stream = (level >= ERROR) ? std::cerr : std::cout;
        stream << get_timestamp() << " " << level_to_string(level) << " ";
        (stream << ... << args);
        stream << '\n';
    }

    template <typename... Args>
    static void debug(Args &&...args)
    {
        log(DEBUG, args...);
    }

    template <typename... Args>
    static void info(Args &&...args)
    {
        log(INFO, args...);
    }

    template <typename... Args>
    static void warning(Args &&...args)
    {
        log(WARNING, args...);
    }

    template <typename... Args>
    static void error(Args &&...args)
    {
        log(ERROR, args...);
    }
};

// Macros para uso más conveniente
#define LOG_DEBUG(...) SimpleLogger::debug(__VA_ARGS__)
#define LOG_INFO(...) SimpleLogger::info(__VA_ARGS__)
#define LOG_WARNING(...) SimpleLogger::warning(__VA_ARGS__)
#define LOG_ERROR(...) SimpleLogger::error(__VA_ARGS__)

#endif
