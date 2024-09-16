#include <algorithm>

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <sycl/sycl.hpp>
#include <dpc_common.hpp>

#include "f.h"
#include "trapezoid.h"
#include "timestamps.h"

// {{UnoAPI:main-print-function-values:begin}}
// function template to avoid code repetition (DRY)
// works as long as the first argument has a size()
// method and an indexing operator[]
template <class Indexable> void print_function_values(const Indexable & values, const double x_min, const double dx, const uint x_precision, const uint y_precision) {
    for (auto i{0UL}; i < values.size(); i++) {
        fmt::print("{}: f({:.{}f}) = {:.{}f}\n", i, x_min + i * dx, x_precision, values[i], y_precision);
    }
}
// {{UnoAPI:main-print-function-values:end}}

template <typename T>
class SyclAllocator {
public:
    // Defining type alias `value_type` for T.
    using value_type = T;

    // Pointer to SYCL queue. This allows the allocator can interact with the queue
    // without copying the queue itself.
    sycl::queue* q;

    // Constructor That accepts a reference to a SYCL queue and stores it as a pointer.
    SyclAllocator(sycl::queue &queue) : q(&queue) {}

    T* allocate(std::size_t n) {
        // Allocating using sycl::malloc_shared, allowing for host and device memory accessing.
        // Dereferencing the queue pointer to specify which queue's context the memory should be allocated in.
        return static_cast<T*>(sycl::malloc_shared(n * sizeof(T), *q));
    }

    void deallocate(T* p, std::size_t n) {
        // Deallocating memory with sycl::free.
        sycl::free(p, *q);
    }
};

int main(const int argc, const char * const argv[]) {
    // {{UnoAPI:main-declarations:begin}}
    size_t total_workload{1000};
    uint grain_size{100};
    double x_min{0.0};
    double x_max{1.0};
    bool with_usm_allocation{false};
    bool show_function_values{false};
    bool run_sequentially{false};
    bool run_cpuonly{false};
    uint x_precision{1};
    uint y_precision{1};
    std::string perf_output;
    ts_vector timestamps;
    std::string device_name;
    // {{UnoAPI:main-declarations:end}}

    // {{UnoAPI:main-cli-setup-and-parse:begin}}
    CLI::App app{"Trapezoidal integration"};
    app.option_defaults()->always_capture_default(true);
    app.add_option("-l,--lower,--xmin", x_min, "x min value");
    app.add_option("-u,--upper,--xmax", x_max, "x max value");
    app.add_option("-n,--total-workload", total_workload, "total workload (number of trapezoids)")->check(CLI::PositiveNumber.description(" >= 1"));
    app.add_option("-g,--grain-size", grain_size, "number of inner (sequential) trapezoids (for each outer trapezoid)")->check(CLI::PositiveNumber.description(" >= 1"));
    app.add_flag("-w, --with-usm-allocation", with_usm_allocation);
    app.add_flag("-s,--sequential", run_sequentially);
    app.add_flag("-c,--cpu-only", run_cpuonly);
    app.add_flag("-v,--show-function-values", show_function_values);
    app.add_option("-x,--x-format-precision", x_precision, "decimal precision for x values")->check(CLI::PositiveNumber.description(" >= 1"));
    app.add_option("-y,--y-format-precision", y_precision, "decimal precision for y (function) values")->check(CLI::PositiveNumber.description(" >= 1"));
    app.add_option("-p,--perfdata-output-file", perf_output, "output file for performance data");

    CLI11_PARSE(app, argc, argv);
    // {{UnoAPI:main-cli-setup-and-parse:end}}

    if (x_min > x_max) {
        spdlog::error("invalid range: [{}, {}]", x_min, x_max);
        return 1;
    }

    if (total_workload % grain_size != 0) {
        spdlog::warn("total workload {} is not a multiple of grain size {}", total_workload, grain_size);
    }
    
    // {{UnoAPI:main-domain-setup:begin}}
    size_t number_of_trapezoids{total_workload / grain_size};
    const auto size{number_of_trapezoids + 1};
    const auto dx{(x_max - x_min) / number_of_trapezoids};
    const auto half_dx{0.5 * dx}; // precomputed for area calculation
    const auto dx_inner{dx / grain_size};
    const auto half_dx_inner{0.5 * dx_inner};
    // {{UnoAPI:main-domain-setup:end}}

    spdlog::info("integrating function from {} to {} using {} trapezoid(s) with grain size {}, dx = {}", x_min, x_max, total_workload, grain_size, dx);
    mark_time(timestamps, "Start");

    // {{UnoAPI:main-sequential-option:begin}}
    if (run_sequentially) {
        device_name = "sequential";
        std::vector values(size, 0.0);
        auto result{0.0};

        mark_time(timestamps,"Memory allocation");
        spdlog::info("starting sequential integration");

        // populate vector with function values and add trapezoid area to result
        // the inner loop performs a finer-grained calculation
        values[0] += f(x_min);
        for (auto i{0UL}; i < number_of_trapezoids; i++) {
            const auto x_pos{x_min + i * dx};
            result += outer_trapezoid(grain_size, x_pos, dx_inner, half_dx_inner);
            values[i + 1] = f(x_pos);
        }

        mark_time(timestamps, "Integration");
        spdlog::info("result should be available now");
        fmt::print("result = {}\n", result);

        if (show_function_values) {
            spdlog::info("showing function values");
            print_function_values(values, x_min, dx, x_precision, y_precision);
            mark_time(timestamps, "Output");
        }
    }
    // {{UnoAPI:main-sequential-option:end}}
    
    else {
        // {{UnoAPI:main-parallel-devices:begin}}
        sycl::device device { run_cpuonly ? sycl::cpu_selector_v : sycl::default_selector_v };
        // {{UnoAPI:main-parallel-devices:end}}
        // we allow the queue to figure out the correct ordering of the three tasks
        // {{UnoAPI:main-parallel-queue:begin}}
        sycl::queue q{device, dpc_common::exception_handler};
        mark_time(timestamps,"Queue creation");
        device_name = q.get_device().get_info<sycl::info::device::name>();
        spdlog::info("Device: {}", device_name);
        // {{UnoAPI:main-parallel-queue:end}}

        if (with_usm_allocation) {
            // USM backed std::vector allocation::begin
            // Creating an instance of SyclAllocator for type double. The allocator is associated with SYCL queue `q`, memory allocations
            // will be maanged by this custom alocator for the device and host via Unified Shared Memory (USM).
            SyclAllocator<double> allocator(q);
            // Creating a std::vector `v_vec` to hold type double. The custom allocator (SyclAllocator) manages the memory in shared USM,
            // making the vector's data accessible from both the host and device.
            std::vector<double, SyclAllocator<double>> v_vec(size, allocator);
            // Creating a std::vector `t_vec` to hold type double. The custom allocator (SyclAllocator) manages the memory in shared USM,
            // making the vector's data accessible from both the host and device.
            std::vector<double, SyclAllocator<double>> t_vec(number_of_trapezoids, allocator);
            // Keeping r_buf as sycl::buffer, sycl::buffer is used for sycl::reduction. Have not found a way to do this
            // with USM yet.
            sycl::buffer<double> r_buf{sycl::range<1>{1}};
            // USM backed std::vector allocation::end
            mark_time(timestamps,"Memory allocation");

            // populate std::vector with function values::begin
            q.submit([&](auto & h) {
                // Capturing `v_vec: std::vector` by reference.
                h.parallel_for(size, [=,v = &v_vec[0]](const auto & index) {
                    // Using pointer `v` to modify the data in std::vector `v_vec`
                    v[index] = f(x_min + index * dx);
                });
            }); // end of command group
            // No need for q.wait() here, there are no dependencies between the first and second task.
            // populate std::vector with function values::end

            // populate buffer with trapezoid values
            // the inner, sequential loop performs a finer-grained calculation
            // {{UnoAPI:main-parallel-submit-parallel-for-trapezoids:begin}}
            q.submit([&](auto & h) {
                h.parallel_for(size, [=,t = &t_vec[0]](const auto & index) {
                    t[index] = outer_trapezoid(grain_size, x_min + index * dx, dx_inner, half_dx_inner);
                });
            }); // end of command group
            // Need q.wait() here, there are dependencies between the second and third task.
            q.wait();
            // {{UnoAPI:main-parallel-submit-parallel-for-trapezoids:end}}

            // perform reduction into result
            // {{UnoAPI:main-parallel-submit-reduce:begin}}
            q.submit([&](auto & h) {
                const auto sum_reduction{sycl::reduction(r_buf, h, sycl::plus<>())};
                h.parallel_for(sycl::range<1>{number_of_trapezoids}, sum_reduction, [=,t = &t_vec[0]](const auto & index, auto & sum) {
                    sum.combine(t[index]);
                });
            }); // end of command group
            q.wait();
            // {{UnoAPI:main-parallel-submit-reduce:end}}
            spdlog::info("done submitting to queue...waiting for results");
            const sycl::host_accessor result{r_buf};
            mark_time(timestamps,"Integration");
            spdlog::info("result should be available now");
            fmt::print("result = {}\n", result[0]);
        }
        else {
            // important: buffer NOT explicitly backed by host-allocated vector
            // this allows the data to live on the device until accessed on the host (if desired)
            // {{UnoAPI:main-parallel-buffers:begin}}
            sycl::buffer<double> v_buf{sycl::range<1>{size}};
            sycl::buffer<double> t_buf{sycl::range<1>{number_of_trapezoids}};
            sycl::buffer<double> r_buf{sycl::range<1>{1}};
            // {{UnoAPI:main-parallel-buffers:end}}

            mark_time(timestamps,"Memory allocation");
            spdlog::info("preparing for vectorized integration");
            // populate buffer with function values
            // {{UnoAPI:main-parallel-submit-parallel-for-values:begin}}
            q.submit([&](auto & h) {
                const sycl::accessor v{v_buf, h};
                h.parallel_for(size, [=](const auto & index) {
                    v[index] = f(x_min + index * dx);
                });
            }); // end of command group
            // {{UnoAPI:main-parallel-submit-parallel-for-values:end}}

            // populate buffer with trapezoid values
            // the inner, sequential loop performs a finer-grained calculation
            // {{UnoAPI:main-parallel-submit-parallel-for-trapezoids:begin}}
            q.submit([&](auto & h) {
                const sycl::accessor t{t_buf, h};
                h.parallel_for(size, [=](const auto & index) {
                    t[index] = outer_trapezoid(grain_size, x_min + index * dx, dx_inner, half_dx_inner);
                });
            }); // end of command group
            // {{UnoAPI:main-parallel-submit-parallel-for-trapezoids:end}}

            // perform reduction into result
            // {{UnoAPI:main-parallel-submit-reduce:begin}}
            q.submit([&](auto & h) {
                const sycl::accessor t{t_buf, h};
                const auto sum_reduction{sycl::reduction(r_buf, h, sycl::plus<>())};
                h.parallel_for(sycl::range<1>{number_of_trapezoids}, sum_reduction, [=](const auto & index, auto & sum) {
                    sum.combine(t[index]);
                });
            }); // end of command group
            // {{UnoAPI:main-parallel-submit-reduce:end}}
            spdlog::info("done submitting to queue...waiting for results");

            // {{UnoAPI:main-parallel-gather-on-host:begin}}
            const sycl::host_accessor result{r_buf};
            mark_time(timestamps,"Integration");
            spdlog::info("result should be available now");
            fmt::print("result = {}\n", result[0]);
            // {{UnoAPI:main-parallel-gather-on-host:end}}
        }
    }
    // end of scope waits for the queued work to complete

    mark_time(timestamps,"DONE");
    spdlog::info("all done for now");
    print_timestamps(timestamps, perf_output, device_name);

    return 0;
}
