**Build.** CMake is most common: `cmake -S . -B build [-DCMAKE_BUILD_TYPE=Debug]`, `cmake --build build -j`, tests `ctest --test-dir build --output-on-failure`. Also Make, Meson, Bazel. Compile directly only for throwaway: `g++ -std=c++20 -Wall -Wextra file.cpp -o prog`. Match the project's standard and warning flags; build with warnings on.

**Resource management (RAII).** Every resource (memory, file, lock, socket) is owned by an object whose destructor releases it. Use `std::unique_ptr` for sole ownership, `std::shared_ptr` only for genuinely shared ownership, `std::weak_ptr` to break cycles. Avoid raw `new`/`delete`; never `malloc`/`free` in C++. Containers (`std::vector`, `std::string`, `std::array`, `std::map`) over raw arrays and C strings.

**Idioms.** `const` correctness everywhere; pass large objects by `const&`, cheap ones by value, use `&&`/`std::move` for transfers. `auto` where it aids clarity, range-based `for`, `nullptr`, `constexpr`/`consteval`, `enum class`, structured bindings, `std::optional`/`std::variant`/`std::string_view` (C++17), `<algorithm>`/ranges (C++20) over hand loops. Rule of 0 (let the compiler generate special members); if you declare one of destructor/copy/move, handle all (rule of 3/5). Mark overrides `override`, non-inherited classes `final`.

**Errors.** Exceptions for exceptional failures (throw by value, catch by `const&`); or `std::optional`/`std::expected`/error codes where the project prefers. RAII guarantees cleanup on throw. Don't throw across module/ABI boundaries you don't control.

**Headers.** Declarations in `.hpp`/`.h`, definitions in `.cpp`; include guards or `#pragma once`; include what you use; forward-declare to cut dependencies.

**Testing.** GoogleTest (`TEST(Suite, Name)`, `EXPECT_EQ`/`ASSERT_*`) or Catch2 (`TEST_CASE`/`REQUIRE`), wired into CTest. Run `ctest`. Sanitizers help: `-fsanitize=address,undefined`.

**Pitfalls.** Dangling references/iterators; use-after-free and double-free (smart pointers prevent most); undefined behavior (signed overflow, OOB, uninitialized reads); object slicing when storing derived by value; forgetting virtual destructors on base classes; data races (use `std::mutex`/`std::atomic`); `std::endl` flushing in hot loops.
