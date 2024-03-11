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
import qualified OpenAI.API as API

import qualified Data.Aeson as A
import qualified Data.Text as T
import qualified Data.ByteString as BS
import           System.Process
import           GHC.Generics
import           GHC.IO.Exception
import           System.Environment      (getEnv)

import           Network.HTTP.Client     (newManager)
import           Network.HTTP.Client.TLS (tlsManagerSettings)
import           Servant.Client
import           Servant.API


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


data TextToSpeechInput = TextToSpeechInput
  { script :: T.Text
  } deriving (Eq, Show, Generic)

data TextToSpeechOutput = TextToSpeechOutput
  { code :: Int
  , stdout :: String
  , stderr :: String
  } deriving (Eq, Show, Generic)

instance A.FromJSON TextToSpeechInput
instance A.ToJSON TextToSpeechInput
instance A.FromJSON TextToSpeechOutput
instance A.ToJSON TextToSpeechOutput

instance Tool TextToSpeechInput TextToSpeechOutput where
  toolFunctionName = "text_to_speech"
  toolSchema = API.ChatCompletionTool
    { chatCompletionToolType = "function"
    , chatCompletionToolFunction = API.FunctionObject
      { functionObjectDescription = Just "Speak text"
      , functionObjectName = toolFunctionName @TextToSpeechInput
      , functionObjectParameters = Just $
        [ ("type", "object")
        , ("properties", A.Object
            [ ("script", A.Object [
                  ("type", "string"),
                  ("description", "A script for speech")
                  ])
            ])
        , ("required", A.Array ["script"])
        ]
      }
    }
  toolExec args = do
    api_key <- (API.clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"
    url <- parseBaseUrl "https://api.openai.com/v1/"
    manager <- newManager tlsManagerSettings
    let API.OpenAIBackend{..} = API.createOpenAIClient
    let request = API.CreateSpeechRequest
                    { API.createSpeechRequestModel = API.CreateSpeechRequestModel "tts-1"
                    , API.createSpeechRequestInput = args.script
                    , API.createSpeechRequestVoice = "alloy"
                    , API.createSpeechRequestResponseUnderscoreformat = Just "mp3"
                    , API.createSpeechRequestSpeed = Nothing
                    }
    res <- API.callOpenAI (mkClientEnv manager url) $ createSpeech api_key request
    BS.writeFile "out.mp3" res
    (code, stdout, stderr) <- readCreateProcessWithExitCode (shell $ "afplay " <> "out.mp3") ""
    let code' = case code of
          ExitSuccess -> 0
          ExitFailure v -> v
    return $ TextToSpeechOutput code' stdout stderr

