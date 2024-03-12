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

module IntelliMonad.Prompt where

import Codec.Picture.Png (encodePng)
import Control.Monad (forM, forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (MonadTrans, lift)
import Control.Monad.Trans.State (get, put, runStateT)
import Data.Aeson (encode)
import qualified Data.Aeson as A
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base64 as Base64
import Data.Coerce
import qualified Data.Map as M
import Data.Maybe (catMaybes, fromMaybe)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Data.Time
import Data.Time.Calendar
import Database.Persist.Sqlite (SqliteConf)
import IntelliMonad.Persist
import IntelliMonad.Tools
import IntelliMonad.Types
import Network.HTTP.Client (managerResponseTimeout, newManager, responseTimeoutMicro)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Servant.Client (mkClientEnv, parseBaseUrl)
import System.Environment (getEnv, lookupEnv)

defaultUTCTime :: UTCTime
defaultUTCTime = UTCTime (coerce (0 :: Integer)) 0

getContext :: (MonadIO m, MonadFail m) => Prompt m Context
getContext = get

setContext :: (MonadIO m, MonadFail m) => Context -> Prompt m ()
setContext context = do
  put context
  withDB $ \conn -> save @SqliteConf conn context
  return ()

push :: (MonadIO m, MonadFail m) => Contents -> Prompt m ()
push contents = do
  prev <- getContext
  let nextContents = prev.contextBody <> contents
      next =
        prev
          { contextBody = nextContents,
            contextRequest = toRequest prev.contextRequest (prev.contextHeader <> nextContents <> prev.contextFooter)
          }
  setContext next

  withDB $ \conn -> saveContents @SqliteConf conn contents
  return ()

pushToolReturn :: (MonadIO m, MonadFail m) => Contents -> Prompt m ()
pushToolReturn contents = do
  prev <- getContext
  let toolMap = M.fromList (map (\v@(Content _ (ToolReturn id' _ _) _ _) -> (id', v)) contents)
      nextContents =
        concat $
          map
            ( \v -> case v of
                Content _ (Message _) _ _ -> [v]
                Content _ (Image _ _) _ _ -> [v]
                Content _ (ToolCall id' _ _) _ _ ->
                  case M.lookup id' toolMap of
                    Just v' -> [v, v']
                    Nothing -> [v]
                Content _ (ToolReturn _ _ _) _ _ -> [v]
            )
            prev.contextBody
      next =
        prev
          { contextBody = nextContents,
            contextRequest = toRequest prev.contextRequest (prev.contextHeader <> nextContents <> prev.contextFooter)
          }
  setContext next

  withDB $ \conn -> saveContents @SqliteConf conn contents
  return ()

call :: (MonadIO m, MonadFail m) => Prompt m Contents
call = do
  prev <- getContext
  (contents, res) <- liftIO $ runRequest prev.contextSessionName prev.contextRequest prev.contextBody
  let current_total_tokens = fromMaybe 0 $ API.completionUsageTotalUnderscoretokens <$> API.createChatCompletionResponseUsage res
      next =
        prev
          { contextResponse = Just res,
            contextTotalTokens = current_total_tokens
          }
  setContext next
  push contents
  if hasToolCall contents
    then do
      showContents contents
      retTool <- tryToolExec next.contextSessionName contents
      showContents retTool
      pushToolReturn retTool
      call
    else return contents

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

tryToolExec :: (MonadIO m, MonadFail m) => Text -> Contents -> Prompt m Contents
tryToolExec sessionName contents = do
  cs <- forM (filterToolCall contents) $ \c@(Content user (ToolCall id' name' args') _ _) -> do
    if name' == toolFunctionName @BashInput
      then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String BashInput) of
        Left err -> return Nothing
        Right input -> do
          output <- liftIO $ toolExec input
          time <- liftIO getCurrentTime
          return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) sessionName time)
      else
        if name' == toolFunctionName @TextToSpeechInput
          then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String TextToSpeechInput) of
            Left err -> return Nothing
            Right input -> do
              output <- liftIO $ toolExec input
              time <- liftIO getCurrentTime
              return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) sessionName time)
          else return Nothing
  return $ catMaybes cs

runPrompt :: (MonadIO m, MonadFail m) => Text -> API.CreateChatCompletionRequest -> Prompt m a -> m a
runPrompt sessionName req func = do
  context <- withDB $ \conn -> do
    load @SqliteConf conn sessionName >>= \case
      Just v -> return v
      Nothing -> do
        time <- liftIO getCurrentTime
        let init =
              Context
                { contextRequest = req,
                  contextResponse = Nothing,
                  contextHeader = [],
                  contextBody = [],
                  contextFooter = [],
                  contextTotalTokens = 0,
                  contextSessionName = sessionName,
                  contextCreated = time
                }
        initialize @SqliteConf conn init
        return init
  fst <$> runStateT func context

instance ChatCompletion Contents where
  toRequest orgRequest contents =
    let messages = flip map contents $ \case
          Content user (Message message) _ _ ->
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Just $ API.ChatCompletionRequestMessageContentText message,
                API.chatCompletionRequestMessageRole = userToText user,
                API.chatCompletionRequestMessageName = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content user (Image type' img) _ _ ->
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent =
                  Just $
                    API.ChatCompletionRequestMessageContentParts
                      [ API.ChatCompletionRequestMessageContentPart
                          { API.chatCompletionRequestMessageContentPartType = "image_url",
                            API.chatCompletionRequestMessageContentPartImageUnderscoreurl =
                              Just $
                                API.ChatCompletionRequestMessageContentPartImageImageUrl
                                  { API.chatCompletionRequestMessageContentPartImageImageUrlUrl =
                                      "data:image/png;base64," <> img,
                                    API.chatCompletionRequestMessageContentPartImageImageUrlDetail = Nothing
                                  }
                          }
                      ],
                API.chatCompletionRequestMessageRole = userToText user,
                API.chatCompletionRequestMessageName = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content user (ToolCall id' name' args') _ _ ->
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Nothing,
                API.chatCompletionRequestMessageRole = userToText user,
                API.chatCompletionRequestMessageName = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecalls =
                  Just
                    [ API.ChatCompletionMessageToolCall
                        { API.chatCompletionMessageToolCallId = id',
                          API.chatCompletionMessageToolCallType = "function",
                          API.chatCompletionMessageToolCallFunction =
                            API.ChatCompletionMessageToolCallFunction
                              { API.chatCompletionMessageToolCallFunctionName = name',
                                API.chatCompletionMessageToolCallFunctionArguments = args'
                              }
                        }
                    ],
                API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
          Content user (ToolReturn id' name' ret') _ _ ->
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Just $ API.ChatCompletionRequestMessageContentText ret',
                API.chatCompletionRequestMessageRole = userToText user,
                API.chatCompletionRequestMessageName = Just name',
                API.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
     in orgRequest {API.createChatCompletionRequestMessages = messages}
  fromResponse sessionName response =
    concat $ flip map (API.createChatCompletionResponseChoices response) $ \res ->
      let message = API.createChatCompletionResponseChoicesInnerMessage res
          role = textToUser $ API.chatCompletionResponseMessageRole message
          content = API.chatCompletionResponseMessageContent message
       in case API.chatCompletionResponseMessageToolUnderscorecalls message of
            Just toolcalls -> map (\(API.ChatCompletionMessageToolCall id' _ (API.ChatCompletionMessageToolCallFunction name' args')) -> Content role (ToolCall id' name' args') sessionName defaultUTCTime) toolcalls
            Nothing -> [Content role (Message (fromMaybe "" content)) sessionName defaultUTCTime]

runRequest :: (ChatCompletion a) => Text -> API.CreateChatCompletionRequest -> a -> IO (a, API.CreateChatCompletionResponse)
runRequest sessionName defaultReq request = do
  api_key <- (API.clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"
  url <- do
    lookupEnv "OPENAI_ENDPOINT" >>= \case
      Just url -> parseBaseUrl url
      Nothing -> parseBaseUrl "https://api.openai.com/v1/"
  manager <-
    newManager
      ( tlsManagerSettings
          { managerResponseTimeout = responseTimeoutMicro (120 * 1000 * 1000)
          }
      )
  let API.OpenAIBackend {..} = API.createOpenAIClient
      req = (toRequest defaultReq request)
  res <- API.callOpenAI (mkClientEnv manager url) $ createChatCompletion api_key req
  return (fromResponse sessionName res, res)

showContents :: (MonadIO m) => Contents -> m ()
showContents res = do
  forM_ res $ \(Content user message _ _) ->
    liftIO $
      T.putStrLn $
        userToText user
          <> ": "
          <> case message of
            Message t -> t
            Image _ _ -> "Image: ..."
            c@(ToolCall _ _ _) -> T.pack $ show c
            c@(ToolReturn _ _ _) -> T.pack $ show c