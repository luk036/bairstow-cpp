
#include <doctest/doctest.h> // for ResultBuilder, CHECK, TEST_CASE

#include <atomic>
#include <memory>
#include <vector>

#include <pthreadpool.h>

typedef std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)>
    auto_pthreadpool_t;

const size_t kParallelize1DRange = 1303;

TEST_CASE("test pthreadpool") {
  std::vector<std::atomic_bool> indicators(kParallelize1DRange);

  auto_pthreadpool_t threadpool(pthreadpool_create(1), pthreadpool_destroy);
  REQUIRE(threadpool.get());

  pthreadpool_parallelize_1d(
      threadpool.get(),
      [&indicators](size_t i) {
        indicators[i].store(true, std::memory_order_relaxed);
      },
      kParallelize1DRange);

  for (size_t i = 0; i < kParallelize1DRange; i++) {
    CHECK(indicators[i].load(std::memory_order_relaxed));
  }
}
