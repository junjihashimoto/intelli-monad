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
  toolHeader = [(Content System (Message "Calcurate user input, then output just the number. Then call 'output_number' function.") "" defaultUTCTime)]

main :: IO ()
main = do
  v <- runPromptWithValidation @ValidateNumber @StatelessConf [] [] "default" (fromModel "gpt-4") "2+3+3+sin(3)"
--   runPrompt (Proxy :: Proxy (Int -> Int)) "Output uesr input multiplied by 3, then call 'output_number' function." (20)
  print (v :: Maybe ValidateNumber)
