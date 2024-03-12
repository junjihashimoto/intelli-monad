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

module IntelliMonad.Tools.Bash where

import Data.Maybe (catMaybes, fromMaybe)
import Control.Monad (forM, forM_)
import Data.Aeson (encode)
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import qualified Data.Text as T
import Data.Text (Text)
import qualified Data.Text.Encoding as T
import Control.Monad.IO.Class
import GHC.Generics
import GHC.IO.Exception
import IntelliMonad.Types
import Network.HTTP.Client (newManager)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Servant.API
import Servant.Client
import System.Environment (getEnv)
import System.Process
import Data.Proxy
import Data.Time

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
     } deriving (Eq, Show, Generic)

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
