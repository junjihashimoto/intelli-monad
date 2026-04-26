# intelli-monad

A type-level prompt library for Haskell that provides high-level APIs for interacting with Large Language Models (LLMs).

## Features

- **Type-safe prompts** - Use Haskell's type system to define and validate prompts
- **Multi-backend support** - Works with OpenAI, Anthropic, and Gemini APIs
- **Function calling** - Define tools and functions that LLMs can call
- **Session management** - Persistent conversation history with SQLite
- **Streaming support** - Real-time response streaming
- **Custom instructions** - Add system-level instructions to guide LLM behavior

## Quick Start

### 1. Install Dependencies

```bash
cd intelli-monad
cabal build all
```

### 2. Configure Your LLM Backend

Create `intellimonad-config.yaml`:

```yaml
# Backend type: openai, anthropic, or gemini
backend: openai

# API endpoint
endpoint: "https://api.openai.com"

# API key
apiKey: "sk-..."

# Model name
model: "gpt-4"
```

### 3. Run Examples

```bash
# Calculator example
cabal run calc

# Auto-talk example (Haruhi & Kyon conversation)
cabal run auto-talk

# REPL
cabal run intelli-monad -- repl default
```

## Configuration

### intellimonad-config.yaml

The configuration file specifies which LLM backend to use. The `backend` field determines the API protocol.

**OpenAI:**
```yaml
backend: openai
endpoint: "https://api.openai.com"
apiKey: "sk-..."
model: "gpt-4"
```

**Anthropic (Claude):**
```yaml
backend: anthropic
endpoint: "https://api.anthropic.com"
apiKey: "sk-ant-..."
model: "claude-3-5-sonnet-20241022"
```

**Google Gemini:**
```yaml
backend: gemini
endpoint: "https://generativelanguage.googleapis.com"
apiKey: "..."
model: "gemini-pro"
```

**Local LLM (via OpenAI-compatible server):**
```yaml
backend: openai
endpoint: "http://localhost:9000"
apiKey: ""  # Empty for local setups without auth
model: "gpt-4"
```

**Notes:**
- The `backend` field is optional and defaults to `openai` if not specified
- The `apiKey` can be empty for local setups that don't require authentication
- Each backend uses its own API format, which is automatically handled by the library

## Usage Examples

### Simple Chat

```haskell
import IntelliMonad.Prompt
import IntelliMonad.Types
import IntelliMonad.Persist

main :: IO ()
main = do
  let defaultReq = LouterRequest (defaultRequest @OpenAI)

  runInputT defaultSettings $
    runPrompt @SqliteConf [] [] "my-session" defaultReq $ do
      response <- call [user "What is 2+2?"]
      liftIO $ print response
```

### Function Calling

```haskell
import IntelliMonad.Types
import Data.Aeson (FromJSON, ToJSON)
import GHC.Generics

-- Define a tool
data Calculator = Calculator
  { operation :: Text
  , x :: Double
  , y :: Double
  } deriving (Generic, JSONSchema, FromJSON, ToJSON)

instance HasFunctionObject Calculator where
  getFunctionName = "calculator"
  getFunctionDescription = "Perform arithmetic operations"
  getFieldDescription "operation" = "Operation: add, subtract, multiply, divide"
  getFieldDescription "x" = "First number"
  getFieldDescription "y" = "Second number"

instance Tool Calculator where
  data Output Calculator = CalcResult Double deriving (Show)

  toolExec (Calculator op x y) = do
    let result = case op of
          "add" -> x + y
          "subtract" -> x - y
          "multiply" -> x * y
          "divide" -> x / y
          _ -> 0
    return $ CalcResult result
```

## REPL Commands

```bash
# Start REPL
cd intelli-monad
cabal run intelli-monad -- repl [session-name]

# Commands:
:help              Show help
:clear             Clear conversation
:show-contents     Show all messages
:show-usage        Show token usage
:show-request      Show current request config
:edit              Edit conversation
:list-sessions     List all sessions
:switch-session    Switch to different session
:delete-session    Delete a session
:read-image        Read and analyze an image
```

## Project Structure

```
intelli-monad/
├── intelli-monad/          # Main library
│   ├── src/
│   │   └── IntelliMonad/
│   │       ├── Types.hs    # Core types and LLM API
│   │       ├── Prompt.hs   # Prompt management
│   │       ├── Tools.hs    # Function calling
│   │       ├── Persist.hs  # Session persistence
│   │       ├── Repl.hs     # REPL implementation
│   │       └── Config.hs   # Configuration
│   ├── app/
│   │   ├── Main.hs         # Main REPL executable
│   │   ├── calc.hs         # Calculator example
│   │   └── auto-talk.hs    # Auto-talk example
│   └── intellimonad-config.yaml  # Configuration file
```

## Development

```bash
# Build
cabal build all

# Run tests
cabal test

# Build with profiling
cabal build --enable-profiling
```

## Dependencies

- **louter** - Multi-protocol LLM client library
- **persistent** - Database persistence
- **sqlite** - SQLite backend for sessions
- **aeson** - JSON parsing
- **yaml** - YAML configuration
- **haskeline** - REPL line editing

## License

MIT
