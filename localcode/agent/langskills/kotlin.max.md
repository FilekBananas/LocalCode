**Build.** Gradle is standard, usually Kotlin DSL (`build.gradle.kts`). Use the wrapper: `./gradlew build`, `./gradlew test`, `./gradlew run`. Dependencies in `dependencies {}`. Don't invoke `kotlinc` directly in a Gradle project. Android projects add the Android Gradle Plugin.

**Null safety.** The headline feature: types are non-null by default; nullable is `T?`. Use `?.` (safe call), `?:` (Elvis/default), `let`/`run` for scoped non-null work, and smart casts after `if (x != null)`. Avoid `!!` (throws) except where null is a real invariant violation. Platform types from Java are `T!` — annotate or check.

**Idioms.** `val` (immutable) by default, `var` only when reassigned. `data class` for value holders (free `equals`/`hashCode`/`copy`/destructuring). `sealed class`/`sealed interface` + exhaustive `when` for state machines. Extension functions instead of utility classes. Named and default arguments instead of overloads/builders. `object` for singletons, `companion object` for statics. Scope functions `let/run/with/apply/also` — pick by intent. Prefer read-only collections (`listOf`, `mapOf`) and transform with `map`/`filter`/`fold`.

**Concurrency.** Coroutines: `suspend fun`, `launch`/`async` in a `CoroutineScope`, `withContext(Dispatchers.IO)` for blocking work, structured concurrency (child jobs cancel with the parent). `Flow` for streams. Don't block a coroutine thread.

**Errors.** Exceptions as in Java but Kotlin has no checked exceptions. Use `runCatching {}` → `Result` where it reads well; `require`/`check`/`error` for preconditions.

**Interop.** Seamless with Java; mind nullability at the boundary, `@JvmStatic`/`@JvmOverloads`/`@JvmField` when Java must call Kotlin.

**Testing.** JUnit 5 or `kotlin.test` (`@Test`, `assertEquals`, `assertFailsWith`); MockK for mocking; Kotest as an alternative style. Run `./gradlew test`.

**Pitfalls.** Overusing `!!`; leaking `lateinit var` before init (`isInitialized`); mutating a `var` captured in a lambda; forgetting `when` must be exhaustive as an expression; equality of `data class` ignoring array contents; blocking inside coroutines.
