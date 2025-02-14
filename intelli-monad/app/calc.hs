{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Main where

import Data.Aeson
import Data.Text (Text)
import Data.Proxy
import GHC.Generics
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import IntelliMonad.Config
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


data Number = Number
  { number :: Double
  }
  deriving (Eq, Show, Generic, JSONSchema, FromJSON, ToJSON)

instance HasFunctionObject Number where
  getFunctionName = "number"
  getFunctionDescription = "validate input"
  getFieldDescription "number" = "A number"

instance Tool Number where
  data Output Number = NumberOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic, FromJSON, ToJSON)
  toolExec _ = return $ NumberOutput 0 "" ""

data Input = Input
  { formula :: Text
  }
  deriving (Eq, Show, Generic, JSONSchema, FromJSON, ToJSON)

instance HasFunctionObject Input where
  getFunctionName = "formula"
  getFunctionDescription = "Describe a formula"
  getFieldDescription "formula" = "A formula"

main :: IO ()
main = do
  config <- readConfig
  v <- runPromptWithValidation @ValidateNumber @StatelessConf [] [] "default" (fromModel config.model) "2+3+3+sin(3)"
  print (v :: Maybe ValidateNumber)

  v <- generate [user "Calcurate a formula"] (Input "1+3")
  print (v :: Maybe Number)
