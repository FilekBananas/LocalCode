**Composer.** Dependency manager: `composer install` (from `composer.lock`), `composer update`, `composer require vendor/pkg`, `composer require --dev ...`. Autoloading is PSR-4 via `vendor/autoload.php` (configured in `composer.json` `autoload.psr-4`). Scripts run with `composer run <name>`.

**Modern style.** Start files with `declare(strict_types=1);`. Type-hint every parameter, property, and return (including `void`, nullable `?T`, union `A|B`, `mixed`). Use namespaces matching PSR-4, explicit visibility (`public`/`protected`/`private`) on all properties and methods, `readonly` properties and constructor promotion (8.1+), enums (8.1+), `match` expressions, named arguments, null-safe `?->`, `??`/`??=`. Prefer `===`/`!==` (strict comparison) over `==`.

**Layout.** PSR-4: `src/` mapped to a namespace, `tests/` for tests, one class per file named after the class. Configuration and front controller (`public/index.php`) per framework (Laravel, Symfony) if used — follow the framework's conventions.

**Errors.** Throw `Exception`/`Error` subclasses implementing `Throwable`; catch specifically; never silence with the `@` operator. Set `error_reporting(E_ALL)` in dev. Validate and sanitize all external input.

**Databases.** Use PDO (or the framework's query builder/ORM) with prepared statements and bound parameters — never interpolate user data into SQL. Escape output for the context (htmlspecialchars for HTML) to avoid XSS.

**Style.** PSR-12 coding style, PSR-1 basics. Format/lint with PHP-CS-Fixer or PHP_CodeSniffer (`phpcs`), static analysis with PHPStan or Psalm if configured.

**Testing.** PHPUnit: classes `extends TestCase`, methods `test*()` or attribute `#[Test]`, `$this->assertEquals(...)`, data providers, `expectException`. Run `./vendor/bin/phpunit`. Pest is an alternative in some repos.

**Pitfalls.** SQL injection / XSS from unescaped input; loose `==` comparisons (`0 == "a"` surprises pre-8.0); mixing types without strict_types; `null` method calls; global state/superglobals; mis-set array vs list; silent `@` errors; leaking stack traces in production.
