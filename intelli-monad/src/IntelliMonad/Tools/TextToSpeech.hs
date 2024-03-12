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

module IntelliMonad.Tools.TextToSpeech where

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

data TextToSpeech = TextToSpeech
  { script :: T.Text
  }
  deriving (Eq, Show, Generic)


instance A.FromJSON TextToSpeech

instance A.ToJSON TextToSpeech

instance A.FromJSON (Output TextToSpeech)

instance A.ToJSON (Output TextToSpeech)

instance Tool TextToSpeech where
  data Output TextToSpeech = TextToSpeechOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic)
  toolFunctionName = "text_to_speech"
  toolSchema =
    API.ChatCompletionTool
      { chatCompletionToolType = "function",
        chatCompletionToolFunction =
          API.FunctionObject
            { functionObjectDescription = Just "Speak text",
              functionObjectName = toolFunctionName @TextToSpeech,
              functionObjectParameters =
                Just $
                  [ ("type", "object"),
                    ( "properties",
                      A.Object
                        [ ( "script",
                            A.Object
                              [ ("type", "string"),
                                ("description", "A script for speech")
                              ]
                          )
                        ]
                    ),
                    ("required", A.Array ["script"])
                  ]
            }
      }
  toolExec args = do
    api_key <- (API.clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"
    url <- parseBaseUrl "https://api.openai.com/v1/"
    manager <- newManager tlsManagerSettings
    let API.OpenAIBackend {..} = API.createOpenAIClient
    let request =
          API.CreateSpeechRequest
            { API.createSpeechRequestModel = API.CreateSpeechRequestModel "tts-1",
              API.createSpeechRequestInput = args.script,
              API.createSpeechRequestVoice = "alloy",
              API.createSpeechRequestResponseUnderscoreformat = Just "mp3",
              API.createSpeechRequestSpeed = Nothing
            }
    res <- API.callOpenAI (mkClientEnv manager url) $ createSpeech api_key request
    BS.writeFile "out.mp3" res
    (code, stdout, stderr) <- readCreateProcessWithExitCode (shell $ "afplay " <> "out.mp3") ""
    let code' = case code of
          ExitSuccess -> 0
          ExitFailure v -> v
    return $ TextToSpeechOutput code' stdout stderr

