{-# LANGUAGE AllowAmbiguousTypes #-}
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

module IntelliMonad.Tools.Bash where

import qualified Data.Aeson as A
import GHC.Generics
import GHC.IO.Exception
import IntelliMonad.Types
import qualified OpenAI.Types as API
import System.Process

data Bash = Bash
  { script :: String
  }
  deriving (Eq, Show, Generic)

instance A.FromJSON Bash

instance A.ToJSON Bash

instance A.FromJSON (Output Bash)

instance A.ToJSON (Output Bash)

instance Tool Bash where
  data Output Bash = BashOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic)

  toolFunctionName = "call_bash_script"
  toolSchema =
    API.ChatCompletionTool
      { chatCompletionToolType = "function",
        chatCompletionToolFunction =
          API.FunctionObject
            { functionObjectDescription = Just "Call a bash script in a local environment",
              functionObjectName = toolFunctionName @Bash,
              functionObjectParameters =
                Just $
                  [ ("type", "object"),
                    ( "properties",
                      A.Object
                        [ ( "script",
                            A.Object
                              [ ("type", "string"),
                                ("description", "A script executing in a local environment ")
                              ]
                          )
                        ]
                    ),
                    ("required", A.Array ["script"])
                  ]
            }
      }
  toolExec args = do
    (code, stdout, stderr) <- readCreateProcessWithExitCode (shell args.script) ""
    let code' = case code of
          ExitSuccess -> 0
          ExitFailure v -> v
    return $ BashOutput code' stdout stderr
