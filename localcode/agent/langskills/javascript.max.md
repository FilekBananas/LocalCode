**Package manager.** Pick by lockfile: `package-lock.json`→`npm`, `pnpm-lock.yaml`→`pnpm`, `yarn.lock`→`yarn`, `bun.lockb`→`bun`. Install with `npm ci`/`pnpm i --frozen-lockfile` in CI, `npm install` locally. Never mix managers. Run tasks through `package.json` `scripts` (`npm run build`, `npm test`).

**Modules.** `"type": "module"` in package.json → ESM (`import`/`export`, `.mjs` is always ESM); otherwise CommonJS (`require`/`module.exports`, `.cjs` always CJS). Match the surrounding files.

**Style.** `const` by default, `let` when reassigned, never `var`. Arrow functions, `async/await` over raw `.then()`, template literals, destructuring, default params, spread/rest, optional chaining `?.` and nullish coalescing `??`. Use strict equality `===`. Keep functions small and pure where possible. Format with Prettier, lint with ESLint if configured (`npm run lint`).

**Errors & async.** Wrap `await` in try/catch; always handle or propagate promise rejections (an unhandled rejection crashes Node). Throw `Error` instances, not strings. Validate external input. For concurrent work use `Promise.all` / `Promise.allSettled`.

**Dependencies.** Keep `dependencies` vs `devDependencies` correct. Prefer the platform/stdlib (`fetch`, `URL`, `structuredClone`) before adding a package. Don't add a dependency that isn't already there without asking.

**Testing.** Use the configured runner: Jest, Vitest, Mocha, or the built-in `node --test`. Files usually `*.test.js`/`*.spec.js` or under `__tests__/`. Run `npm test`; add `--watch` locally. Mock I/O, assert behavior not implementation.

**Pitfalls.** `this` binding (lost in detached callbacks — use arrow or `.bind`); floating-point money math; `==` coercion; mutating shared objects; forgetting `await` (returns a pending promise); `for...in` over arrays (use `for...of`/`forEach`); timezone bugs with `Date`.
