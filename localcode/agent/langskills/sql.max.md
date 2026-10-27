**Dialect.** SQL is not fully portable — confirm the engine (PostgreSQL, MySQL/MariaDB, SQLite, SQL Server, Oracle) and use its types, functions, `LIMIT`/`TOP`/`FETCH`, identifier quoting, and upsert syntax (`ON CONFLICT` vs `ON DUPLICATE KEY` vs `MERGE`). Match what the schema/migrations already use.

**Safety.** Always parameterize/bind values from application code (`?`/`$1`/`:name`) — never concatenate user input into SQL (injection). Scope every `UPDATE`/`DELETE` with a `WHERE`; test the matching `SELECT` first. Run destructive or multi-statement changes inside a transaction (`BEGIN … COMMIT`/`ROLLBACK`) so a failure leaves the data consistent.

**Query quality.** List explicit columns, not `SELECT *` (stability, less I/O). Use explicit `JOIN ... ON`; qualify columns with table aliases in multi-table queries. Prefer set-based operations over row-by-row loops/cursors. Use `EXISTS`/`IN` appropriately; understand `LEFT JOIN` + `NULL` filtering vs `INNER JOIN`. `GROUP BY` all non-aggregated selected columns; filter groups with `HAVING`, rows with `WHERE`. Window functions (`ROW_NUMBER() OVER (...)`) for ranking/running totals. CTEs (`WITH`) for readability.

**Performance.** Index columns used in `WHERE`, `JOIN`, and `ORDER BY`; composite indexes follow left-prefix order. Keep predicates sargable — avoid wrapping an indexed column in a function or leading `%LIKE`. Read the plan (`EXPLAIN`/`EXPLAIN ANALYZE`) when a query is slow. Beware implicit type casts defeating indexes.

**Schema & migrations.** Change schema through the project's migration framework (Alembic, Flyway, Liquibase, Rails/Django migrations, etc.) — versioned, reviewed, reversible — not hand-run DDL on a live DB. Name constraints; use appropriate types (numeric for money, `TIMESTAMP WITH TIME ZONE` for instants), `NOT NULL` + defaults, foreign keys, and `UNIQUE`/`CHECK` constraints to enforce invariants in the DB.

**Style.** Uppercase keywords, lowercase identifiers (or match the repo), one major clause per line, consistent aliasing. Comment non-obvious logic.

**Pitfalls.** Injection; missing `WHERE` on update/delete; `NULL` logic (`= NULL` is never true — use `IS NULL`, and `NOT IN` with NULLs); timezone/`DATE` handling; floating-point money; N+1 from the app; locking/deadlocks from long transactions; relying on row order without `ORDER BY`.
