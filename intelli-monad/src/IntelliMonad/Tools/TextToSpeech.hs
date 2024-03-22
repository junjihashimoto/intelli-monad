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

module IntelliMonad.Tools.TextToSpeech where

import Control.Monad.IO.Class
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import qualified Data.Text as T
import GHC.Generics
import GHC.IO.Exception
import IntelliMonad.Types
import Network.HTTP.Client (newManager)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Servant.Client
import System.Environment (getEnv)
import System.Process

data TextToSpeech = TextToSpeech
  { script :: T.Text
  }
  deriving (Eq, Show, Generic, JSONSchema, A.FromJSON, A.ToJSON)

instance HasFunctionObject TextToSpeech where
  getFunctionName = "text_to_speech"
  getFunctionDescription = "Speak text"
  getFieldDescription "script" = "A script for speech"

instance Tool TextToSpeech where
  data Output TextToSpeech = TextToSpeechOutput
    { code :: Int,
      stdout :: String,
      stderr :: String
    }
    deriving (Eq, Show, Generic, A.FromJSON, A.ToJSON)
  toolExec args = liftIO $ do
    api_key <- (API.clientAuth . T.pack) <$> (getEnv "OPENAI_API_KEY")
    url <- parseBaseUrl "https://api.openai.com/v1/"
    manager <- newManager tlsManagerSettings
    let API.OpenAIBackend {..} = API.createOpenAIClient
    let request =
          API.CreateSpeechRequest
            { API.createSpeechRequestModel = API.CreateSpeechRequestModel "tts-1",
              API.createSpeechRequestInput = (script args :: T.Text),
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
