{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes  #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE FlexibleInstances  #-}


module Prompt.Tools where

import qualified OpenAI.Types as API

import qualified Data.Aeson as A
import           System.Process
import           GHC.Generics
import           GHC.IO.Exception

import           Prompt.Types

data BashInput = BashInput
  { script :: String
  } deriving (Eq, Show, Generic)

data BashOutput = BashOutput
  { code :: Int
  , stdout :: String
  , stderr :: String
  } deriving (Eq, Show, Generic)

instance A.FromJSON BashInput
instance A.ToJSON BashInput
instance A.FromJSON BashOutput
instance A.ToJSON BashOutput

instance Tool BashInput BashOutput where
  toolFunctionName = "call_bash_script"
  toolSchema = API.ChatCompletionTool
    { chatCompletionToolType = "function"
    , chatCompletionToolFunction = API.FunctionObject
      { functionObjectDescription = Just "Call a bash script in a local environment"
      , functionObjectName = toolFunctionName @BashInput
      , functionObjectParameters = Just $
        [ ("type", "object")
        , ("properties", A.Object [
              ("script", A.Object [
                  ("type", "string"),
                  ("description", "A script executing in a local environment ")
                  ])
              ])
        , ("required", A.Array ["script"])
        ]
      }
    }
  toolExec args = do
    (code, stdout, stderr) <- readCreateProcessWithExitCode (shell args.script) ""
    let code' = case code of
          ExitSuccess -> 0
          ExitFailure v -> v
    return $ BashOutput code' stdout stderr

