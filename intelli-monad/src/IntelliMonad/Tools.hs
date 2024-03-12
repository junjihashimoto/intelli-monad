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

module IntelliMonad.Tools where

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

defaultTools :: [ToolProxy]
defaultTools =
  [ ToolProxy (Proxy :: Proxy Bash)
  , ToolProxy (Proxy :: Proxy TextToSpeech)
  ]

addTools :: [ToolProxy] -> API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest
addTools [] v = v
addTools (tool:tools') v =
  case tool of
    (ToolProxy (_ :: Proxy a)) -> addTools tools' (toolAdd @a v)

toolExec'
  :: forall t to m. (MonadIO m, MonadFail m, Tool t, A.FromJSON t, A.ToJSON (Output t))
  => Text -> Text -> Text -> Text -> Prompt m (Maybe Content)
toolExec' sessionName id' name' args' = do
  if name' == toolFunctionName @t
    then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String t) of
           Left err -> return Nothing
           Right input -> do
             output <- liftIO $ toolExec input
             time <- liftIO getCurrentTime
             return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) sessionName time)
    else return Nothing

(<||>)
  :: forall m. (MonadIO m, MonadFail m)
  => (Text -> Text -> Text -> Text -> Prompt m (Maybe Content))
  -> (Text -> Text -> Text -> Text -> Prompt m (Maybe Content))
  -> Text -> Text -> Text -> Text -> Prompt m (Maybe Content)
(<||>) tool0 tool1 sessionName id' name' args' = do
  a <- tool0 sessionName id' name' args'
  case a of
    Just v -> return (Just v)
    Nothing -> tool1 sessionName id' name' args'

mergeToolCall :: (MonadIO m, MonadFail m) => [ToolProxy] -> Text -> Text -> Text -> Text -> Prompt m (Maybe Content)
mergeToolCall [] _ _ _ _ = return Nothing
mergeToolCall (tool:tools') sessionName id' name' args' = do
  case tool of
    (ToolProxy (_ :: Proxy a)) -> (toolExec' @a <||> mergeToolCall tools') sessionName id' name' args'
   
hasToolCall :: Contents -> Bool
hasToolCall cs =
  let loop [] = False
      loop ((Content _ (ToolCall _ _ _) _ _) : _) = True
      loop (_ : cs') = loop cs'
   in loop cs

filterToolCall :: Contents -> Contents
filterToolCall cs =
  let loop [] = []
      loop (m@(Content _ (ToolCall _ _ _) _ _) : cs') = m : loop cs'
      loop (_ : cs') = loop cs'
   in loop cs


tryToolExec :: (MonadIO m, MonadFail m) => [ToolProxy] -> Text -> Contents -> Prompt m Contents
tryToolExec tools sessionName contents = do
  cs <- forM (filterToolCall contents) $ \c@(Content user (ToolCall id' name' args') _ _) -> do
    mergeToolCall tools sessionName id' name' args'
  return $ catMaybes cs
