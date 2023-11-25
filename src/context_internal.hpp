#pragma once

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "bipp/bipp.h"
#include "bipp/config.h"
#include "bipp/context.hpp"
#include "bipp/exceptions.hpp"
#include "logger.hpp"
#include "memory/allocator.hpp"
#include "memory/allocator_factory.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/util/blas_api.hpp"
#include "gpu/util/queue.hpp"
#include "gpu/util/runtime_api.hpp"
#endif

#if !defined(BIPP_MAGMA) && defined(BIPP_CUDA)
#include <cusolverDn.h>
#endif

#ifdef BIPP_MPI
#include <mpi.h>

#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_init_guard.hpp"
#endif

namespace bipp {

class ContextInternal {
public:
  explicit ContextInternal(BippProcessingUnit pu)
      : hostAlloc_(AllocatorFactory::host()), log_(BIPP_LOG_LEVEL_OFF) {
#ifdef BIPP_MPI
    initialize_mpi_init_guard();
    int myRank = 0;
    mpi_check_status(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
#else
    int myRank = 0;
#endif

    if (pu == BIPP_PU_AUTO) {
      // select GPU if available
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
      try {
        int deviceId = 0;
        gpu::api::get_device(&deviceId);
        pu_ = BIPP_PU_GPU;
      } catch (const GPUError& e) {
        pu_ = BIPP_PU_CPU;
      }
#else
      pu_ = BIPP_PU_CPU;
#endif
    } else {
      pu_ = pu;
    }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    if (pu_ == BIPP_PU_GPU) queue_.emplace();
#endif

    const char* logOut = "stdout";
    if(const char* logOutEnv = std::getenv("BIPP_LOG_OUT")) {
      logOut = logOutEnv;
    }

    std::string logRankString = "0";
    if(const char* logRankEnv = std::getenv("BIPP_LOG_RANK")) {
      logRankString = logRankEnv;
    }
    const int logRank = std::stoi(logRankString);

    auto logLevel = BIPP_LOG_LEVEL_OFF;

    // Set initial log level if environment variable is set
    if(const char* envLog = std::getenv("BIPP_LOG_LEVEL")) {
      if (!std::strcmp(envLog, "off") || !std::strcmp(envLog, "OFF"))
        logLevel = BIPP_LOG_LEVEL_OFF;
      else if (!std::strcmp(envLog, "debug") || !std::strcmp(envLog, "DEBUG"))
        logLevel = BIPP_LOG_LEVEL_DEBUG;
      else if (!std::strcmp(envLog, "info") || !std::strcmp(envLog, "INFO"))
        logLevel = BIPP_LOG_LEVEL_INFO;
      else if (!std::strcmp(envLog, "warn") || !std::strcmp(envLog, "WARN"))
        logLevel = BIPP_LOG_LEVEL_WARN;
      else if (!std::strcmp(envLog, "error") || !std::strcmp(envLog, "ERROR"))
        logLevel = BIPP_LOG_LEVEL_ERROR;
    }

    if(logRank == myRank) {
        log_ = Logger(logLevel, logOut);
    }

    if(pu_== BIPP_PU_CPU)
      log_.log(BIPP_LOG_LEVEL_INFO, "{} CPU context created", static_cast<const void*>(this));
    else
      log_.log(BIPP_LOG_LEVEL_INFO, "{} GPU context created", static_cast<const void*>(this));
  }

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  auto gpu_queue() -> gpu::Queue& { return queue_.value(); }

  auto gpu_queue() const -> const gpu::Queue& { return queue_.value(); }
#endif

  auto processing_unit() const -> BippProcessingUnit { return pu_; }

  auto host_alloc() const -> const std::shared_ptr<Allocator>& { return hostAlloc_; }

  auto set_log(BippLogLevel level, const char* out = "stdout") { log_ = Logger(level, out); }

  auto logger() -> Logger& { return log_; }

  ~ContextInternal() {
    try {
      log_.log_timings(BIPP_LOG_LEVEL_INFO);
      log_.log(BIPP_LOG_LEVEL_INFO, "{} Context destroyed", static_cast<const void*>(this));
    } catch (...) {
    }
  }

private:
  BippProcessingUnit pu_;
  Logger log_;
  std::shared_ptr<Allocator> hostAlloc_;

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
  std::optional<gpu::Queue> queue_;
#endif
};

struct InternalContextAccessor {
  static auto get(const Context& ctx) -> const std::shared_ptr<ContextInternal>& {
    return ctx.ctx_;
  }
};

}  // namespace bipp
