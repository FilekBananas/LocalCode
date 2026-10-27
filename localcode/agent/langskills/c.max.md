**Build.** `Makefile` (`make`, `make test`) or CMake (`cmake -B build && cmake --build build`). Direct: `cc -std=c11 -Wall -Wextra -Wpedantic -g file.c -o prog`. Build with warnings enabled and clean; add `-fsanitize=address,undefined` while developing. Match the project's C standard (c99/c11/c17).

**Memory.** Fully manual. Each `malloc`/`calloc`/`realloc` is paired with exactly one `free`. Check the result of every allocation (NULL on failure). After `free`, null the pointer to avoid use-after-free/double-free. Own your lifetimes: document who frees what. `realloc` can move — never keep the old pointer. Use `valgrind` or ASan to find leaks and invalid access.

**Safety.** C will not protect you: bounds-check all indexing and buffer writes. Prefer sized functions: `snprintf` over `sprintf`, `memcpy`/`memmove` with explicit sizes, `strncpy`/`strlcpy` (and always null-terminate). Validate every external length/size. Avoid VLAs and `alloca` on attacker-controlled sizes. Beware integer overflow in size computations (`a*b` before `malloc`). Initialize variables before use.

**Errors.** Convention: return `int` (0 ok, non-zero/`-1` error) or a pointer (NULL on error), and set `errno`; the caller checks and `perror`/`strerror`. Propagate failures up; free partial allocations on the error path (goto-cleanup pattern is idiomatic).

**Structure.** Declarations and types in headers with `#ifndef X_H`/`#pragma once` guards; definitions in `.c`. `static` for file-local functions/vars. `const` for read-only params. Keep functions small; one responsibility.

**Style.** Consistent brace/indent style with the file; snake_case; descriptive names; comment the non-obvious invariants. Avoid macros where an `enum`/`static inline` works.

**Testing.** No single standard — projects use a `make test`, a custom harness, Unity, CMocka, or Check. Run what's there.

**Pitfalls.** Buffer overflows, off-by-one, use-after-free/double-free, uninitialized memory, forgetting to null-terminate strings, integer overflow, signed/unsigned comparison, returning pointers to stack locals, format-string bugs (`printf(user)`), ignoring return values.
