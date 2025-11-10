#include <fmt/format.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#include "algorithms/bootstrap.cuh"
#include "algorithms/paths.cuh"
#include "core/tensor.cuh"
#include "headers.cuh"
#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/file_io.cuh"
#include "utils/logging.cuh"
#include "utils.cuh"

#include <sys/resource.h>
#include <sys/sysinfo.h>

#ifndef FECUDA_SOURCE_DIR
#define FECUDA_SOURCE_DIR "."
#endif

namespace
{
    struct BenchmarkMetrics
    {
        double elapsed_ms = 0.0;
        double cpu_total_percent = 0.0;
        double cpu_per_core_percent = 0.0;
        double ram_after_mb = 0.0;
        double ram_delta_mb = 0.0;
        double ram_percent = 0.0;
        double gpu_util_percent = 0.0;
        double gpu_mem_percent = 0.0;
        double gpu_mem_peak_mb = 0.0;
        bool gpu_metrics_available = false;
    };

    struct BenchmarkRun
    {
        std::string label;
        BenchmarkMetrics metrics;
    };

    struct SystemMemoryInfo
    {
        double total_bytes = 0.0;
        double total_mb = 0.0;
        bool valid = false;
    };

    struct CliConfig
    {
        std::string user_tag = "manual";
        bool user_tag_provided = false;
        bool profile_child = false;
        std::vector<std::string> only_labels;
        std::vector<std::string> forwarded_args;
    };

    double timeval_to_seconds(const timeval &tv)
    {
        return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) / 1'000'000.0;
    }

    CliConfig parse_cli(int argc, char **argv)
    {
        CliConfig config;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            if (arg == "--profile-child")
            {
                config.profile_child = true;
                continue;
            }

            if (arg == "--only")
            {
                if (i + 1 < argc)
                {
                    config.only_labels.emplace_back(argv[++i]);
                }
                else
                {
                    std::cerr << "Advertencia: --only requiere un argumento" << std::endl;
                }
                continue;
            }
            if (arg.rfind("--only=", 0) == 0)
            {
                config.only_labels.emplace_back(arg.substr(7));
                continue;
            }

            if (!config.user_tag_provided && !arg.empty() && arg[0] != '-')
            {
                config.user_tag = arg;
                config.user_tag_provided = true;
            }

            config.forwarded_args.push_back(arg);
        }

        if (const char *forced_only = std::getenv("FECUDA_BENCH_ONLY_OVERRIDE"))
        {
            if (forced_only[0] != '\0')
            {
                config.only_labels.clear();
                config.only_labels.emplace_back(forced_only);
            }
        }

        return config;
    }

    std::string sanitize_token(const std::string &input)
    {
        if (input.empty())
        {
            return "run";
        }
        std::string sanitized;
        sanitized.reserve(input.size());
        for (char c : input)
        {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
            {
                sanitized.push_back(c);
            }
            else
            {
                sanitized.push_back('_');
            }
        }
        if (sanitized.empty())
        {
            sanitized = "run";
        }
        return sanitized;
    }

    std::string format_double(double value, int precision = 6)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision) << value;
        return oss.str();
    }

    std::string current_date_string()
    {
        std::time_t now = std::time(nullptr);
        std::tm local_tm{};
#if defined(_GNU_SOURCE) || defined(__unix__)
        localtime_r(&now, &local_tm);
#else
        local_tm = *std::localtime(&now);
#endif
        char buffer[16];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d", &local_tm);
        return std::string(buffer);
    }

    std::filesystem::path resolve_report_directory(const CliConfig &config)
    {
        if (const char *forced_dir = std::getenv("FECUDA_BENCH_REPORT_DIR"))
        {
            return std::filesystem::weakly_canonical(std::filesystem::path(forced_dir));
        }

        const std::string folder_name = fmt::format("{}_{}", current_date_string(), sanitize_token(config.user_tag));
        auto base = std::filesystem::path(FECUDA_SOURCE_DIR);
        if (!base.is_absolute())
        {
            base = std::filesystem::current_path() / base;
        }
        return std::filesystem::weakly_canonical(base) / "results" / "benchmarks" / folder_name;
    }

    std::string shell_quote(const std::string &value)
    {
        std::string quoted = "'";
        for (char c : value)
        {
            if (c == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted.push_back(c);
            }
        }
        quoted += "'";
        return quoted;
    }

    bool should_run_label(const CliConfig &config, const std::string &label)
    {
        if (config.only_labels.empty())
        {
            return true;
        }
        return std::find(config.only_labels.begin(), config.only_labels.end(), label) != config.only_labels.end();
    }

    std::string resolve_executable_path()
    {
        std::vector<char> buffer(4096);
        ssize_t len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
        if (len > 0)
        {
            buffer[len] = '\0';
            return std::string(buffer.data());
        }
        return {};
    }

    SystemMemoryInfo query_system_memory()
    {
        struct sysinfo info{};
        if (sysinfo(&info) != 0)
        {
            std::cerr << "No se pudo consultar sysinfo para memoria total" << std::endl;
            return {};
        }
        const double total_bytes = static_cast<double>(info.totalram) * info.mem_unit;
        const double total_mb = total_bytes / (1024.0 * 1024.0);
        return {total_bytes, total_mb, total_bytes > 0.0};
    }

    double read_process_rss_mb()
    {
        std::ifstream status("/proc/self/status");
        if (!status.is_open())
        {
            return 0.0;
        }

        std::string key;
        while (status >> key)
        {
            if (key == "VmRSS:")
            {
                double value_kb = 0.0;
                status >> value_kb;
                return value_kb / 1024.0;
            }
            std::string rest_of_line;
            std::getline(status, rest_of_line);
        }
        return 0.0;
    }

    bool run_ncu_report(const std::filesystem::path &report_dir,
                        const CliConfig &config,
                        const std::string &exe_path,
                        const std::vector<std::string> &extra_args = {},
                        const std::string *forced_only_label = nullptr)
    {
        if (exe_path.empty())
        {
            std::cerr << "No se pudo resolver la ruta del ejecutable para generar el perfil NCU" << std::endl;
            return false;
        }

        const std::string base_name = forced_only_label
                                          ? fmt::format("ncu_report_{}", sanitize_token(*forced_only_label))
                                          : "ncu_report";
        const auto export_base = (report_dir / base_name).string();
        const auto log_file = (report_dir / (base_name + ".txt")).string();

        std::ostringstream command;
        command << "FECUDA_BENCH_REPORT_DIR=" << shell_quote(report_dir.string()) << " ";
        if (forced_only_label)
        {
            command << "FECUDA_BENCH_ONLY_OVERRIDE=" << shell_quote(*forced_only_label) << " ";
        }
        command << "ncu -f --target-processes all "
                << "--set full "
                << "--section LaunchStats "
                << "--section SpeedOfLight "
                << "--section MemoryWorkloadAnalysis "
                << "--section SchedulerStats "
                << "--export " << shell_quote(export_base) << " "
                << "--log-file " << shell_quote(log_file) << " "
                << "--csv "
                << shell_quote(exe_path);

        for (const auto &arg : config.forwarded_args)
        {
            command << " " << shell_quote(arg);
        }
        for (const auto &arg : extra_args)
        {
            command << " " << shell_quote(arg);
        }
        command << " --profile-child";

        std::cout << "Generando reporte NCU en " << report_dir << std::endl;
        int ret = std::system(command.str().c_str());
        if (ret != 0)
        {
            std::cerr << "El comando NCU falló con código " << ret << std::endl;
            return false;
        }
        return true;
    }

    class NvmlContext
    {
    public:
        NvmlContext()
        {
            const nvmlReturn_t status = nvmlInit_v2();
            if (status == NVML_SUCCESS)
            {
                available_ = true;
            }
            else
            {
                std::cerr << "NVML no disponible: " << nvmlErrorString(status) << std::endl;
            }
        }

        ~NvmlContext()
        {
            if (available_)
            {
                nvmlShutdown();
            }
        }

        bool available() const { return available_; }

    private:
        bool available_ = false;
    };

    struct GpuMonitorStats
    {
        double avg_util_percent = 0.0;
        double avg_mem_percent = 0.0;
        double peak_mem_mb = 0.0;
        double mem_total_mb = 0.0;
        bool valid = false;
    };

    class GpuMonitor
    {
    public:
        explicit GpuMonitor(nvmlDevice_t device, bool available)
            : device_(device), available_(available) {}

        bool start()
        {
            if (!available_)
            {
                return false;
            }

            running_.store(true);
            worker_ = std::thread([this]()
                                  { monitor_loop(); });
            return true;
        }

        void stop()
        {
            if (!available_)
            {
                return;
            }
            running_.store(false);
            if (worker_.joinable())
            {
                worker_.join();
            }
        }

        void record_manual_sample()
        {
            if (!available_)
            {
                return;
            }

            nvmlUtilization_t util{};
            nvmlMemory_t mem_info{};
            if (nvmlDeviceGetUtilizationRates(device_, &util) != NVML_SUCCESS)
            {
                return;
            }
            if (nvmlDeviceGetMemoryInfo(device_, &mem_info) != NVML_SUCCESS)
            {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            add_sample_locked(util, mem_info);
        }

        GpuMonitorStats stats() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (sample_count_ == 0)
            {
                return {};
            }
            GpuMonitorStats stats;
            stats.avg_util_percent = util_sum_ / static_cast<double>(sample_count_);
            stats.avg_mem_percent = mem_util_sum_ / static_cast<double>(sample_count_);
            stats.peak_mem_mb = peak_mem_bytes_ / (1024.0 * 1024.0);
            stats.mem_total_mb = mem_total_bytes_ / (1024.0 * 1024.0);
            stats.valid = true;
            return stats;
        }

    private:
        void monitor_loop()
        {
            while (running_.load())
            {
                nvmlUtilization_t util{};
                nvmlReturn_t util_status = nvmlDeviceGetUtilizationRates(device_, &util);

                nvmlMemory_t mem_info{};
                nvmlReturn_t mem_status = nvmlDeviceGetMemoryInfo(device_, &mem_info);

                if (util_status == NVML_SUCCESS && mem_status == NVML_SUCCESS)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    add_sample_locked(util, mem_info);
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }

        nvmlDevice_t device_{};
        bool available_ = false;
        std::atomic<bool> running_{false};
        std::thread worker_;

        mutable std::mutex mutex_;
        double util_sum_ = 0.0;
        double mem_util_sum_ = 0.0;
        double peak_mem_bytes_ = 0.0;
        double mem_total_bytes_ = 0.0;
        size_t sample_count_ = 0;

        void add_sample_locked(const nvmlUtilization_t &util, const nvmlMemory_t &mem_info)
        {
            util_sum_ += util.gpu;
            const double mem_percent = (mem_info.total > 0)
                                           ? (static_cast<double>(mem_info.used) / static_cast<double>(mem_info.total)) * 100.0
                                           : 0.0;
            mem_util_sum_ += mem_percent;
            peak_mem_bytes_ = std::max(peak_mem_bytes_, static_cast<double>(mem_info.used));
            mem_total_bytes_ = std::max(mem_total_bytes_, static_cast<double>(mem_info.total));
            ++sample_count_;
        }
    };

    void cleanup_tensor_vector(std::vector<TensorResult> &tensors)
    {
        for (auto &tensor : tensors)
        {
            safe_tensor_cleanup(tensor);
        }
        tensors.clear();
    }

    BenchmarkMetrics measure_run(const std::function<void()> &workload,
                                 const SystemMemoryInfo &memory_info,
                                 nvmlDevice_t device,
                                 bool nvml_available,
                                 unsigned int logical_cores)
    {
        BenchmarkMetrics metrics;
        struct rusage usage_start{};
        struct rusage usage_end{};

        if (getrusage(RUSAGE_SELF, &usage_start) != 0)
        {
            std::cerr << "No se pudo obtener uso de recursos inicial" << std::endl;
        }

        const double rss_before_mb = read_process_rss_mb();

        GpuMonitor monitor(device, nvml_available);
        if (nvml_available)
        {
            monitor.start();
        }

        auto start = std::chrono::high_resolution_clock::now();
        try
        {
            workload();
            CHECK_CUDA(cudaDeviceSynchronize());
            if (nvml_available)
            {
                monitor.record_manual_sample();
            }
        }
        catch (...)
        {
            if (nvml_available)
            {
                monitor.stop();
            }
            throw;
        }
        auto end = std::chrono::high_resolution_clock::now();

        if (nvml_available)
        {
            monitor.stop();
        }

        if (getrusage(RUSAGE_SELF, &usage_end) != 0)
        {
            std::cerr << "No se pudo obtener uso de recursos final" << std::endl;
        }

        const double elapsed_seconds = std::chrono::duration<double>(end - start).count();
        metrics.elapsed_ms = elapsed_seconds * 1000.0;

        const double cpu_time =
            (timeval_to_seconds(usage_end.ru_utime) - timeval_to_seconds(usage_start.ru_utime)) +
            (timeval_to_seconds(usage_end.ru_stime) - timeval_to_seconds(usage_start.ru_stime));
        const double total_cpu_percent = (elapsed_seconds > 0.0) ? (cpu_time / elapsed_seconds) * 100.0 : 0.0;
        metrics.cpu_total_percent = total_cpu_percent;
        const double cores = static_cast<double>(std::max(1u, logical_cores));
        metrics.cpu_per_core_percent = total_cpu_percent / cores;

        const double rss_after_mb = read_process_rss_mb();
        metrics.ram_after_mb = rss_after_mb;
        metrics.ram_delta_mb = std::max(0.0, rss_after_mb - rss_before_mb);
        if (memory_info.valid && memory_info.total_mb > 0.0)
        {
            metrics.ram_percent = (rss_after_mb / memory_info.total_mb) * 100.0;
        }

        if (nvml_available)
        {
            const auto gpu_stats = monitor.stats();
            if (gpu_stats.valid)
            {
                metrics.gpu_metrics_available = true;
                metrics.gpu_util_percent = gpu_stats.avg_util_percent;
                metrics.gpu_mem_percent = gpu_stats.avg_mem_percent;
                metrics.gpu_mem_peak_mb = gpu_stats.peak_mem_mb;
            }
        }

        return metrics;
    }

    void ensure_tensor_ownership(TensorResult &tensor)
    {
        if (tensor.data != nullptr && !tensor.owns_memory)
        {
            tensor.owns_memory = true;
        }
    }

    BenchmarkRun run_tensor_case(const std::string &label,
                                 const TensorResult &tensor,
                                 float threshold,
                                 int order,
                                 const SystemMemoryInfo &memory_info,
                                 nvmlDevice_t device,
                                 bool nvml_available,
                                 unsigned int logical_cores)
    {
        BenchmarkRun run{label, {}};
        auto workload = [&]()
        {
            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;

            iterative_maxmin_cuadrado(tensor, threshold, order, paths, values, pure_paths, pure_values, true);

            cleanup_tensor_vector(paths);
            cleanup_tensor_vector(values);
            cleanup_tensor_vector(pure_paths);
            cleanup_tensor_vector(pure_values);
        };

        run.metrics = measure_run(workload, memory_info, device, nvml_available, logical_cores);
        return run;
    }

    BenchmarkRun run_bootstrap_case(const std::string &label,
                                    const TensorResult &source_tensor,
                                    int replicas,
                                    float threshold,
                                    int order,
                                    const SystemMemoryInfo &memory_info,
                                    nvmlDevice_t device,
                                    bool nvml_available,
                                    unsigned int logical_cores)
    {
        BenchmarkRun run{label, {}};
        auto workload = [&]()
        {
            const size_t total_floats = static_cast<size_t>(source_tensor.M) * source_tensor.N * replicas;
            float *bootstrap_res = static_cast<float *>(std::malloc(total_floats * sizeof(float)));
            if (!bootstrap_res)
            {
                throw std::runtime_error("No se pudo asignar memoria para bootstrap");
            }
            std::unique_ptr<float, decltype(&std::free)> host_guard(bootstrap_res, &std::free);

            float *d_bootstrap = bootstrap_wrapper(source_tensor.data, source_tensor.M, source_tensor.N,
                                                   source_tensor.batch, replicas);
            struct DeviceGuard
            {
                float *ptr;
                ~DeviceGuard()
                {
                    if (ptr)
                    {
                        cudaFree(ptr);
                    }
                }
            } device_guard{d_bootstrap};

            CHECK_CUDA(cudaMemcpy(bootstrap_res, d_bootstrap,
                                  total_floats * sizeof(float),
                                  cudaMemcpyDeviceToHost));

            TensorResult bootstrap_tensor(bootstrap_res, false, replicas, source_tensor.M, source_tensor.N, 1, true);
            host_guard.release();

            std::vector<TensorResult> paths;
            std::vector<TensorResult> values;
            std::vector<TensorResult> pure_paths;
            std::vector<TensorResult> pure_values;

            iterative_maxmin_cuadrado(bootstrap_tensor, threshold, order, paths, values, pure_paths, pure_values, true);

            cleanup_tensor_vector(paths);
            cleanup_tensor_vector(values);
            cleanup_tensor_vector(pure_paths);
            cleanup_tensor_vector(pure_values);
            safe_tensor_cleanup(bootstrap_tensor);
        };

        run.metrics = measure_run(workload, memory_info, device, nvml_available, logical_cores);
        return run;
    }

    void print_run(const BenchmarkRun &run, std::ostream *summary)
    {
        auto log_line = [&](const std::string &line)
        {
            std::cout << line << '\n';
            if (summary)
            {
                (*summary) << line << '\n';
            }
        };

        log_line(fmt::format("[{}]", run.label));
        log_line(fmt::format("  Tiempo total          : {:>10.3f} ms", run.metrics.elapsed_ms));
        log_line(fmt::format("  CPU proceso (total)   : {:>10.2f} %", run.metrics.cpu_total_percent));
        log_line(fmt::format("  CPU prom/core         : {:>10.2f} %", run.metrics.cpu_per_core_percent));
        log_line(fmt::format("  RAM usada en corrida  : {:>10.2f} MB", run.metrics.ram_delta_mb));
        log_line(fmt::format("  RAM proceso total     : {:>10.2f} MB ({:>6.2f} % del sistema)",
                             run.metrics.ram_after_mb, run.metrics.ram_percent));
        if (run.metrics.gpu_metrics_available)
        {
            log_line(fmt::format("  GPU util (promedio)   : {:>10.2f} %", run.metrics.gpu_util_percent));
            log_line(fmt::format("  GPU memoria (avg/pico): {:>10.2f} % / {:>10.2f} MB",
                                 run.metrics.gpu_mem_percent, run.metrics.gpu_mem_peak_mb));
        }
        else
        {
            log_line("  Métricas GPU          : No disponibles");
        }
        log_line(std::string(60, '-'));
    }

    void write_metrics_csv(const std::filesystem::path &report_dir,
                           const std::vector<BenchmarkRun> &runs)
    {
        const auto csv_path = report_dir / "metrics.csv";
        std::ofstream csv(csv_path);
        if (!csv.is_open())
        {
            std::cerr << "No se pudo escribir el CSV de métricas en " << csv_path << std::endl;
            return;
        }

        csv << "label,elapsed_ms,cpu_total_percent,cpu_per_core_percent,"
            << "ram_delta_mb,ram_after_mb,ram_percent,gpu_util_percent,"
            << "gpu_mem_percent,gpu_mem_peak_mb,gpu_metrics_available\n";

        for (const auto &run : runs)
        {
            const auto &m = run.metrics;
            csv << '"' << run.label << '"' << ","
                << format_double(m.elapsed_ms) << ","
                << format_double(m.cpu_total_percent) << ","
                << format_double(m.cpu_per_core_percent) << ","
                << format_double(m.ram_delta_mb) << ","
                << format_double(m.ram_after_mb) << ","
                << format_double(m.ram_percent) << ","
                << (m.gpu_metrics_available ? format_double(m.gpu_util_percent) : "") << ","
                << (m.gpu_metrics_available ? format_double(m.gpu_mem_percent) : "") << ","
                << (m.gpu_metrics_available ? format_double(m.gpu_mem_peak_mb) : "") << ","
                << (m.gpu_metrics_available ? "1" : "0")
                << "\n";
        }

        std::cout << "Métricas tabulares guardadas en " << csv_path << std::endl;
    }
}

int main(int argc, char **argv)
{
    try
    {
        const CliConfig cli = parse_cli(argc, argv);
        const auto report_dir = resolve_report_directory(cli);
        std::filesystem::create_directories(report_dir);

        std::ostringstream summary_stream;
        std::ostream *summary_ptr = cli.profile_child ? nullptr : &summary_stream;
        auto log_line = [&](const std::string &line)
        {
            std::cout << line << '\n';
            if (summary_ptr)
            {
                (*summary_ptr) << line << '\n';
            }
        };

        log_line(fmt::format("Directorio de reporte: {}", report_dir.string()));
        log_line("=== Benchmark iterative_maxmin ===");

        const SystemMemoryInfo memory_info = query_system_memory();
        const unsigned int logical_cores = std::max(1u, std::thread::hardware_concurrency());

        NvmlContext nvml_context;
        nvmlDevice_t device{};
        bool nvml_available = false;
        if (nvml_context.available())
        {
            nvmlReturn_t handle_status = nvmlDeviceGetHandleByIndex(0, &device);
            if (handle_status == NVML_SUCCESS)
            {
                nvml_available = true;
            }
            else
            {
                std::cerr << "No se pudo obtener dispositivo NVML: "
                          << nvmlErrorString(handle_status) << std::endl;
            }
        }

        TensorResult cc;
        TensorResult ee;

        const std::filesystem::path dataset_root = std::filesystem::path(FECUDA_SOURCE_DIR) / "datasets_txt";
        const std::filesystem::path cc_path = dataset_root / "CC.txt";
        const std::filesystem::path ee_path = dataset_root / "EE.txt";

        if (!leer_matriz_3d_desde_archivo(cc_path.string().c_str(), cc, 10, 16, 16, 1))
        {
            std::cerr << "No se pudo cargar CC.txt desde " << cc_path << std::endl;
            return 1;
        }
        ensure_tensor_ownership(cc);

        if (!leer_matriz_3d_desde_archivo(ee_path.string().c_str(), ee, 10, 4, 4, 1))
        {
            std::cerr << "No se pudo cargar EE.txt desde " << ee_path << std::endl;
            safe_tensor_cleanup(cc);
            return 1;
        }
        ensure_tensor_ownership(ee);

        const std::vector<float> thresholds = {0.1f, 0.3f, 0.5f, 0.7f};
        const int order = 5;
        std::vector<BenchmarkRun> runs;
        runs.reserve(thresholds.size() * 2 + 4);

        for (float thr : thresholds)
        {
            const std::string label_cc = fmt::format("tensor_cc_thr_{:.2f}", thr);
            if (should_run_label(cli, label_cc))
            {
                runs.push_back(run_tensor_case(label_cc, cc, thr, order, memory_info, device, nvml_available, logical_cores));
            }

            const std::string label_ee = fmt::format("tensor_ee_thr_{:.2f}", thr);
            if (should_run_label(cli, label_ee))
            {
                runs.push_back(run_tensor_case(label_ee, ee, thr, order, memory_info, device, nvml_available, logical_cores));
            }
        }

        const std::vector<int> bootstrap_reps = {10, 100, 1000, 10000};
        for (int replicas : bootstrap_reps)
        {
            const std::string label = fmt::format("bootstrap_cc_reps_{}", replicas);
            if (should_run_label(cli, label))
            {
                runs.push_back(run_bootstrap_case(label, cc, replicas, 0.2f, order, memory_info, device, nvml_available, logical_cores));
            }
        }

        for (const auto &run : runs)
        {
            print_run(run, summary_ptr);
        }

        safe_tensor_cleanup(cc);
        safe_tensor_cleanup(ee);

        if (!cli.profile_child)
        {
            write_metrics_csv(report_dir, runs);

            const auto summary_path = report_dir / "summary.txt";
            std::ofstream summary_file(summary_path);
            if (summary_file.is_open())
            {
                summary_file << summary_stream.str();
                std::cout << "Resumen guardado en " << summary_path << std::endl;
            }
            else
            {
                std::cerr << "No se pudo escribir el resumen en " << summary_path << std::endl;
            }

            const std::string exe_path = resolve_executable_path();
            if (exe_path.empty())
            {
                std::cerr << "No se encontró el ejecutable para ejecutar NCU." << std::endl;
            }
            else
            {
                const std::vector<std::string> profile_targets = {
                    "bootstrap_cc_reps_10",
                    "bootstrap_cc_reps_100",
                    "bootstrap_cc_reps_1000"};

                for (const auto &target : profile_targets)
                {
                    const bool executed = std::any_of(runs.begin(), runs.end(),
                                                      [&](const BenchmarkRun &run)
                                                      { return run.label == target; });
                    if (!executed)
                    {
                        std::cout << "Saltando NCU para " << target << " (no se ejecutó en esta corrida)." << std::endl;
                        continue;
                    }

                    std::cout << "=== NCU: " << target << " ===" << std::endl;
                    if (!run_ncu_report(report_dir, cli, exe_path, {}, &target))
                    {
                        std::cerr << "Fallo el reporte NCU para " << target << std::endl;
                    }
                }
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error en bench.cu: " << ex.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Error desconocido en bench.cu" << std::endl;
        return 1;
    }

    return 0;
}
