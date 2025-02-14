# intelli-monad

intelli-monad provides high level APIs for prompt engineering using openai.

Featuring:
* Type level function calling with JSON-Schema
* Validation of return value
* Repl interface
* Persistent of prompt logs with SQLite
* Session management with repl

intelli-monad is based on openai-servant-gen.
openai-servant-gen is automatically generated from OpenAPI interface.

# Install

```bash
git clone git@github.com:junjihashimoto/intelli-monad.git
cd intelli-monad
export PATH=~/.local/bin:$PATH
cabal install intelli-monad
```

# Usage of repl

After install intelli-monad, set OPENAI_API_KEY, then run intelli-monad command.
The system commands begin with prefix ":". Anything else will be the user's prompt.

```bash
$ export OPENAI_API_KEY=xxx
$ export OPENAI_MODEL=xxx
$ intelli-monad
% :help
:quit
:clear
:show contents
:show usage
:show request
:show context
:show session
:list sessions
:copy session <from> <to>
:delete session <session name>
:switch session <session name>
:help
% hello
assistant: Hello! How can I assist you today?
```

# Usage of function calling with validation

Here is an example of function calling and validation.
In this example, validation is performed using the input of function calling.

Define the function calling as ValidateNumber, and define the context as Math.

JSONSchema type-class can output it as JSON Schema.
Defining HasFunctionObject class adds descriptin to each field. 
This allows Openai's interface to understand the meaning of each field.
Tool type-class defines the input and output types of function calling, and defines the contents of the function.

CustomInstruction type-class defines the context with headers and footers.

runPromptWithValidation function calls LLM.
The results will be validated and a number will be returned.


```haskell
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

import Data.Aeson
import Data.Proxy
import GHC.Generics
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import OpenAI.Types

data ValidateNumber = ValidateNumber
  { number :: Double
  }
  deriving (Eq, Show, Generic, JSONSchema, FromJSON, ToJSON)

instance HasFunctionObject ValidateNumber where
  getFunctionName = "output_number"
  getFunctionDescription = "validate input number"
  getFieldDescription "number" = "A number that system outputs."

instance Tool ValidateNumber where
  data Output ValidateNumber = ValidateNumberOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic, FromJSON, ToJSON)
  toolExec _ = return $ ValidateNumberOutput 0 "" ""

data Math = Math

instance CustomInstruction Math where
  customHeader = [(Content System (Message "Calcurate user input, then output just the number. Then call 'output_number' function.") "" defaultUTCTime)]
  customFooter = []

main :: IO ()
main = do
  v <- runPromptWithValidation @ValidateNumber @StatelessConf [] [CustomInstructionProxy (Proxy @Math)] "default" (fromModel "gpt-4") "2+3+3+sin(3)"
  print (v :: Maybe ValidateNumber)
```

# Configuration

To specify OpenAI endpoints and models as part of data instead of using environment variables, you can use a configuration file. Create a `intellimonad-config.yaml` file in the root directory of your project with the following content:

```yaml
apiKey: "your_openai_api_key"
endpoint: "https://api.openai.com/v1/"
model: "gpt-4"
```

The `apiKey` is your OpenAI API key, `endpoint` is the OpenAI endpoint, and `model` is the model you want to use.

Make sure to update the `initializePrompt` and `runPrompt` functions to read from the configuration file as shown in the examples in the `intelli-monad/app/auto-talk.hs` and `intelli-monad/app/calc.hs` files.
