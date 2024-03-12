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
  deriving (Eq, Show, Generic, FromJSON, ToJSON)

instance Tool ValidateNumber where
  data Output ValidateNumber = ValidateNumberOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic, FromJSON, ToJSON)

  toolFunctionName = "output_number"
  toolSchema =
    ChatCompletionTool
      { chatCompletionToolType = "function",
        chatCompletionToolFunction =
          FunctionObject
            { functionObjectDescription = Just "",
              functionObjectName = toolFunctionName @ValidateNumber,
              functionObjectParameters =
                Just $
                  [ ("type", "object"),
                    ( "properties",
                      Object
                        [ ( "number",
                            Object
                              [ ("type", "number"),
                                ("description", "A number that system outputs.")
                              ]
                          )
                        ]
                    ),
                    ("required", Array ["output"])
                  ]
            }
      }
  toolExec _ = return $ ValidateNumberOutput 0 "" ""

data Math = Math

instance CustomInstruction Math where
  customHeader = [(Content System (Message "Calcurate user input, then output just the number. Then call 'output_number' function.") "" defaultUTCTime)]
  customFooter = []

main :: IO ()
main = do
  v <- runPromptWithValidation @ValidateNumber @StatelessConf [] [CustomInstructionProxy (Proxy @Math)] "default" (fromModel "gpt-4") "2+3+3+sin(3)"
  print (v :: Maybe ValidateNumber)
