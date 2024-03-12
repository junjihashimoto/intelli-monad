{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module IntelliMonad.CustomInstructions where

import qualified Data.Aeson as A
import Data.Proxy
import GHC.Generics
import IntelliMonad.Types
import qualified OpenAI.API as API
import qualified OpenAI.Types as API

defaultCustomInstructions :: [CustomInstructionProxy]
defaultCustomInstructions = []

data ValidateNumber = ValidateNumber
  { number :: Double
  }
  deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

instance Tool ValidateNumber where
  data Output ValidateNumber = ValidateNumberOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)

  toolFunctionName = "output_number"
  toolSchema =
    API.ChatCompletionTool
      { chatCompletionToolType = "function",
        chatCompletionToolFunction =
          API.FunctionObject
            { functionObjectDescription = Just "",
              functionObjectName = toolFunctionName @ValidateNumber,
              functionObjectParameters =
                Just $
                  [ ("type", "object"),
                    ( "properties",
                      A.Object
                        [ ( "number",
                            A.Object
                              [ ("type", "number"),
                                ("description", "A number that system outputs.")
                              ]
                          )
                        ]
                    ),
                    ("required", A.Array ["number"])
                  ]
            }
      }
  toolExec args = do
    return $ ValidateNumberOutput 0 "" ""

data Math = Math

instance CustomInstruction Math where
  customHeader = [(Content System (Message "Calcurate user input, then output just the number. Then call 'output_number' function.") "" defaultUTCTime)]
  customFooter = []

headers :: [CustomInstructionProxy] -> Contents
headers [] = []
headers (tool : tools') =
  case tool of
    (CustomInstructionProxy (_ :: Proxy a)) -> customHeader @a <> headers tools'

footers :: [CustomInstructionProxy] -> Contents
footers [] = []
footers (tool : tools') =
  case tool of
    (CustomInstructionProxy (_ :: Proxy a)) -> customFooter @a <> footers tools'
