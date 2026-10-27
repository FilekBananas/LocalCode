**Build.** Maven (`pom.xml`): `mvn compile`, `mvn test`, `mvn package`, `mvn verify`. Gradle (`build.gradle`/`.kts`): `./gradlew build test` (use the wrapper `./gradlew`, not a global gradle). Dependencies go in `pom.xml` `<dependencies>` or Gradle `dependencies {}`; don't add jars by hand.

**Layout.** Maven/Gradle standard: `src/main/java/<pkg>/`, `src/test/java/<pkg>/`, resources under `src/main/resources`. Package names are reverse-domain lowercase.

**Idioms.** Program to interfaces (`List` not `ArrayList` in signatures). `final` for fields/params that don't change; immutability where practical (or `record` for data carriers, Java 16+). Return `Optional<T>` rather than null for "maybe absent"; never return null collections (use `List.of()`/`Collections.emptyList()`). Try-with-resources for anything `AutoCloseable`. Generics everywhere (no raw types). Streams/lambdas for transformations, but keep them readable. `var` for obvious local types (Java 10+).

**Errors.** Checked exceptions for recoverable conditions, unchecked (`RuntimeException` subclasses) for programming errors. Catch the specific type; never swallow (`catch (Exception e) {}`). Wrap-and-rethrow with a cause. Validate arguments early (`Objects.requireNonNull`).

**Equality/collections.** Override `equals` and `hashCode` together and consistently; implement `Comparable`/`Comparator` for ordering. Use the right collection (`HashMap`, `ArrayList`, `LinkedHashMap`…). `StringBuilder` for concatenation in loops.

**Style.** Oracle/Google conventions: `camelCase` methods/fields, `PascalCase` types, `UPPER_SNAKE` constants. One top-level public class per file. Javadoc on public API. Format/lint with Spotless/Checkstyle if configured.

**Testing.** JUnit 5 (JUnit Jupiter): `@Test`, `@BeforeEach`, `@ParameterizedTest`, `Assertions.assertEquals/assertThrows`. Mockito for mocks, AssertJ for fluent assertions if present. Run `mvn test` / `./gradlew test`.

**Pitfalls.** `NullPointerException` from unchecked returns; `==` on objects instead of `.equals()`; autoboxing in hot loops / `Integer` identity; mutable static state; resource leaks without try-with-resources; `equals` without `hashCode`; catching and ignoring `InterruptedException` (restore the flag).
