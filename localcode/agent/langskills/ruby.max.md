**Bundler/gems.** Dependencies in `Gemfile`, locked in `Gemfile.lock`; `bundle install`, `bundle update`, and run project tools through `bundle exec <tool>` so versions match. Gems are built with `gem build *.gemspec`. Rails apps use `bin/rails`, `bin/rake`.

**Idioms.** Everything is an object and expressions return values. Favor blocks and Enumerable (`each`, `map`, `select`/`reject`, `reduce`, `find`, `each_with_object`) over manual loops. Symbols (`:key`) for identifiers/hash keys; keyword arguments for clarity; string interpolation `"#{}"`; `||=` memoization; safe navigation `&.`; guard clauses (`return x if cond`) over nested `if`. Use `attr_reader/accessor`, modules/mixins (`include`/`extend`) for shared behavior, `Struct`/`Data` (3.2+) for value objects. `.freeze` constants and literals you don't mutate. Follow the principle of least surprise; prefer readable one-liners but not clever ones.

**Style.** Ruby Style Guide / RuboCop: 2-space indentation, `snake_case` methods and variables, `CamelCase` classes/modules, `SCREAMING_SNAKE` constants, predicate methods end in `?`, dangerous/mutating ones in `!`. Keep methods short. Run `bundle exec rubocop` if configured (`-a` to autocorrect safe offenses).

**Errors.** `raise` specific `StandardError` subclasses (never raise strings for real errors); `rescue => e` catches `StandardError` — rescue the narrowest class, don't rescue `Exception` (it grabs signals). `ensure` for cleanup; `retry` sparingly.

**Testing.** RSpec: `describe`/`context`/`it`, `expect(x).to eq(y)`, `let`, `before`, doubles/`allow`/`receive`. Minitest: `assert_equal`, `Test::Unit`-style or spec-style. Run `bundle exec rspec` or `bundle exec rake test`. Rails: fixtures/factories (FactoryBot).

**Pitfalls.** Mutating shared/frozen objects; `nil` errors (`NoMethodError on nil`) — use `&.`/`dig`/defaults; monkey-patching core classes; truthiness (only `nil`/`false` are falsy, `0` and `""` are truthy); heavy metaprogramming that hides behavior; N+1 queries in Rails; blocking the main thread in concurrent code (GVL).
