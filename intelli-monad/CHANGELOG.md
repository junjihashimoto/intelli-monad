# Revision history for intelli-monad

## 0.1.3.0 -- 2026-04-27

### Added
* New REPL command `:model` for switching the active model at runtime.
* Model name can now be set via the config file and overridden by an environment variable.
* `OPENAI_HTTP_DEBUG` environment variable and additional HTTP debugging settings.
* Timeout passthrough to the louter backend.
* Helper functions used by the examples are now exported from `IntelliMonad.Prompt`.

### Changed
* Default model switched to an open-weight option; the default model name is now empty (must be supplied via config or environment).
* REPL command prefix changed from `/` to `:` (e.g. `:model`, `:help`).
* `cabal.project` no longer requires cabal files in parent directories.

### Removed
* Removed `cabal.project.freeze` so the package builds on Debian.

### Internal
* Hid unused functions and tidied up `IntelliMonad.Prompt` / `IntelliMonad.Types`.

### Acknowledgments
* Many thanks to [@julialongtin](https://github.com/julialongtin) (Julia Longtin) for contributing all of the changes in this release.

## 0.1.0.2 -- 2024-03-30

* Add 'generate' function with typed input and output.
* Add system hook functionality to customize system flow
* Support for key value store


## 0.1.0.1 -- 2024-03-21

* Add CLI options
* Support for arxiv

## 0.1.0.0 -- 2024-03-15

* First version. Released on an unsuspecting world.
