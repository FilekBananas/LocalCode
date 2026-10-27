**Tooling.** .NET CLI: `dotnet restore`, `dotnet build`, `dotnet test`, `dotnet run`, `dotnet format`. Projects are `*.csproj` (SDK-style); solutions `*.sln`. Packages come from NuGet: `dotnet add package X` (version-pinned in the csproj `<PackageReference>`). Target framework is in `<TargetFramework>` (e.g. `net8.0`).

**Idioms.** `var` when the type is obvious. Auto-properties (`public int X { get; set; }`) and `init`-only setters, `record`/`record struct` for immutable data. `async`/`await` end to end — async methods return `Task`/`Task<T>`/`ValueTask`, are named `...Async`, and accept a `CancellationToken`; never `.Result`/`.Wait()` (deadlocks). LINQ (`Where`/`Select`/`FirstOrDefault`) for queries. Pattern matching (`switch` expressions, `is`, property patterns). `using` declarations/statements for `IDisposable`. Expression-bodied members where concise.

**Nullability.** With `<Nullable>enable</Nullable>`, reference types are non-null by default; use `?` for nullable and handle warnings (`?.`, `??`, null checks). Don't suppress with `!` casually.

**Errors.** Throw specific exception types (`ArgumentNullException`, `InvalidOperationException`, custom). Catch only what you can handle; preserve stack traces (`throw;` not `throw ex;`). Validate arguments (`ArgumentNullException.ThrowIfNull`). Avoid exceptions for control flow.

**Style.** Microsoft conventions: `PascalCase` for types/methods/properties, `camelCase` for locals/params, `_camelCase` or `camelCase` for private fields (match the repo), `I`-prefixed interfaces. One public type per file ideally. Program to interfaces (`IEnumerable<T>`, `IReadOnlyList<T>`) in signatures.

**Testing.** xUnit (`[Fact]`, `[Theory]`+`[InlineData]`, `Assert.Equal`), NUnit (`[Test]`), or MSTest (`[TestMethod]`). Moq/NSubstitute for mocks, FluentAssertions for readable asserts if present. Run `dotnet test`.

**Pitfalls.** Async deadlocks from `.Result`/`.Wait()`; not disposing `IDisposable`; `NullReferenceException`; LINQ deferred execution/multiple enumeration; `async void` (only for event handlers); capturing `struct` in closures; `==` vs `.Equals` for value equality; boxing in hot paths.
