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

module Prompt where

import Codec.Picture.Png (encodePng)
import Control.Monad (forM, forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (MonadTrans, lift)
import Control.Monad.Trans.State (get, put, runStateT)
import Data.Aeson (encode)
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base64 as Base64
import Data.Coerce
import qualified Data.Map as M
import Data.Maybe (catMaybes, fromMaybe)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Data.Time
import Data.Time.Calendar
import Database.Persist.Sqlite (SqliteConf)
import Network.HTTP.Client (managerResponseTimeout, newManager, responseTimeoutMicro)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Prompt.Tools
import Prompt.Types
import Servant.Client (mkClientEnv, parseBaseUrl)
import System.Console.Haskeline
import System.Environment (getEnv, lookupEnv)

type Contents = [Content]

defaultUTCTime :: UTCTime
defaultUTCTime = UTCTime (coerce (0 :: Integer)) 0

getContext :: (MonadIO m, MonadFail m) => Prompt m Context
getContext = get

withDB :: (MonadIO m, MonadFail m) => (Conn SqliteConf -> m a) -> m a
withDB func =
  setup (config @SqliteConf) >>= \case
    Nothing -> fail "Can not open a database."
    Just (conn :: Conn SqliteConf) -> func conn

setContext :: (MonadIO m, MonadFail m) => Context -> Prompt m ()
setContext context = do
  put context
  withDB $ \conn -> save @SqliteConf conn context
  return ()

push :: (MonadIO m, MonadFail m) => Contents -> Prompt m ()
push contents = do
  prev <- getContext
  let nextContents = prev.contextContents <> contents
      next =
        prev
          { contextContents = nextContents,
            contextRequest = toRequest prev.contextRequest nextContents
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
            prev.contextContents
      next =
        prev
          { contextContents = nextContents,
            contextRequest = toRequest prev.contextRequest nextContents
          }
  setContext next

  withDB $ \conn -> saveContents @SqliteConf conn contents
  return ()

call :: (MonadIO m, MonadFail m) => Prompt m Contents
call = do
  prev <- getContext
  (contents, res) <- liftIO $ runRequest prev.contextRequest prev.contextContents
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
      retTool <- tryToolExec contents
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

tryToolExec :: (MonadIO m, MonadFail m) => Contents -> Prompt m Contents
tryToolExec contents = do
  cs <- forM (filterToolCall contents) $ \c@(Content user (ToolCall id' name' args') _ _) -> do
    if name' == toolFunctionName @BashInput
      then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String BashInput) of
        Left err -> return Nothing
        Right input -> do
          output <- liftIO $ toolExec input
          time <- liftIO getCurrentTime
          return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) "default" time)
      else
        if name' == toolFunctionName @TextToSpeechInput
          then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String TextToSpeechInput) of
            Left err -> return Nothing
            Right input -> do
              output <- liftIO $ toolExec input
              time <- liftIO getCurrentTime
              return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) "default" time)
          else return Nothing
  return $ catMaybes cs

runPrompt :: (MonadIO m, MonadFail m) => API.CreateChatCompletionRequest -> Prompt m a -> m a
runPrompt req func = do
  context <- withDB $ \conn -> do
    load @SqliteConf conn "default" >>= \case
      Just v -> return v
      Nothing -> do
        time <- liftIO getCurrentTime
        let init =
              Context
                { contextRequest = req,
                  contextResponse = Nothing,
                  contextContents = [],
                  contextTotalTokens = 0,
                  contextSessionName = "default",
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
  fromResponse response =
    concat $ flip map (API.createChatCompletionResponseChoices response) $ \res ->
      let message = API.createChatCompletionResponseChoicesInnerMessage res
          role = textToUser $ API.chatCompletionResponseMessageRole message
          content = API.chatCompletionResponseMessageContent message
       in case API.chatCompletionResponseMessageToolUnderscorecalls message of
            Just toolcalls -> map (\(API.ChatCompletionMessageToolCall id' _ (API.ChatCompletionMessageToolCallFunction name' args')) -> Content role (ToolCall id' name' args') "default" defaultUTCTime) toolcalls
            Nothing -> [Content role (Message (fromMaybe "" content)) "default" defaultUTCTime]

runRequest :: (ChatCompletion a) => API.CreateChatCompletionRequest -> a -> IO (a, API.CreateChatCompletionResponse)
runRequest defaultReq request = do
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
  return (fromResponse res, res)

getTextInputLine :: (MonadTrans t) => String -> t (InputT IO) (Maybe T.Text)
getTextInputLine prompt = fmap (fmap T.pack) (lift $ getInputLine prompt)

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

runRepl :: API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl defaultReq contents = do
  let settings =
        toolAdd @BashInput $
          toolAdd @TextToSpeechInput $
            defaultReq
  runInputT defaultSettings (runPrompt settings (push contents >> loop))
  where
    loop :: Prompt (InputT IO) ()
    loop = do
      minput <- getTextInputLine "% "
      case minput of
        Nothing -> return ()
        Just "quit" -> return ()
        Just "clear" -> do
          loop
        Just "show-contents" -> do
          context <- getContext
          showContents context.contextContents
          loop
        Just "show-usage" -> do
          context <- getContext
          liftIO $ do
            print context.contextTotalTokens
          loop
        Just "show-context" -> do
          context <- getContext
          liftIO $ do
            print context
          loop
        Just "show-context-as-json" -> do
          prev <- getContext
          let req = toRequest prev.contextRequest prev.contextContents
          liftIO $ do
            BS.putStr $ BS.toStrict $ encode req
          loop
        Just input -> do
          let contents = [Content User (Message input) "default" defaultUTCTime]
          push contents
          ret <- call
          showContents ret
          loop
