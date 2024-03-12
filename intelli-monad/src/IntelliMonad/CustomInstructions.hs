{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
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
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

module IntelliMonad.CustomInstructions where

import IntelliMonad.Types
import Data.Proxy
import GHC.Generics
import qualified Data.Aeson as A
import qualified OpenAI.API as API
import qualified OpenAI.Types as API

defaultCustomInstructions :: [CustomInstructionProxy]
defaultCustomInstructions = []


data ValidateNumber = ValidateNumber
  { number :: Double
  }
  deriving (Eq, Show, Generic)

instance A.FromJSON ValidateNumber

instance A.ToJSON ValidateNumber

instance A.FromJSON (Output ValidateNumber)

instance A.ToJSON (Output ValidateNumber)

instance Tool ValidateNumber where
  data Output ValidateNumber = ValidateNumberOutput
     { code :: Int,
       stdout :: String,
       stderr :: String
     } deriving (Eq, Show, Generic)

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
                    ("required", A.Array ["output"])
                  ]
            }
      }
  toolExec args = do
     return $ ValidateNumberOutput 0 "" ""

data Math = Math

instance CustomInstruction Math where
  customHeader = [ (Content System (Message "Calcurate user input, then output just the number. Then call 'output_number' function.") "" defaultUTCTime) ]
  customFooter = []

headers :: [CustomInstructionProxy] -> Contents
headers [] = []
headers (tool:tools') =
  case tool of
    (CustomInstructionProxy (_ :: Proxy a)) -> customHeader @a <> headers tools'

footers :: [CustomInstructionProxy] -> Contents
footers [] = []
footers (tool:tools') =
  case tool of
    (CustomInstructionProxy (_ :: Proxy a)) -> customFooter @a <> footers tools'
