{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
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
import qualified Data.Map as M
import Data.Maybe (catMaybes, fromMaybe)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Data.Time
import Data.Time.Calendar
import Database.Persist.Sqlite (SqliteConf)
import IntelliMonad.CustomInstructions
import IntelliMonad.Persist
import IntelliMonad.Tools
import IntelliMonad.Types
import Network.HTTP.Client (managerResponseTimeout, newManager, responseTimeoutMicro)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import Servant.Client (mkClientEnv, parseBaseUrl)
import System.Environment (getEnv, lookupEnv)

getContext :: (MonadIO m, MonadFail m) => Prompt m Context
getContext = context <$> get

setContext :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Context -> Prompt m ()
setContext context = do
  env <- get
  put $ env {context = context}
  withDB @p $ \conn -> save @p (conn :: Conn p) context
  return ()

switchContext :: (MonadIO m, MonadFail m) => Context -> Prompt m ()
switchContext context = do
  env <- get
  put $ env {context = context}
  return ()

push :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Contents -> Prompt m ()
push contents = do
  prev <- getContext
  let nextContents = prev.contextBody <> contents
      next =
        prev
          { contextBody = nextContents,
            contextRequest = toRequest prev.contextRequest (prev.contextHeader <> nextContents <> prev.contextFooter)
          }
  setContext @p next

  withDB @p $ \conn -> saveContents @p conn contents
  return ()

pushToolReturn :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Contents -> Prompt m ()
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
  setContext @p next

  withDB @p $ \conn -> saveContents @p conn contents
  return ()

call :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Prompt m Contents
call = do
  prev <- getContext
  ((contents, finishReason), res) <- liftIO $ runRequest prev.contextSessionName prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
  let current_total_tokens = fromMaybe 0 $ API.completionUsageTotalUnderscoretokens <$> API.createChatCompletionResponseUsage res
      next =
        prev
          { contextResponse = Just res,
            contextTotalTokens = current_total_tokens
          }
  setContext @p next
  push @p contents

  case finishReason of
    Stop -> return contents
    ToolCalls -> callTool next contents
    FunctionCall -> callTool next contents
    Length -> call @p
    _ -> return contents
  where
    callTool next contents = do
      showContents contents
      env <- get
      retTool <- tryToolExec env.tools next.contextSessionName contents
      showContents retTool
      pushToolReturn @p retTool
      call @p

callPrompt :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Text -> Prompt m Contents
callPrompt input = do
  time <- liftIO getCurrentTime
  context <- getContext
  let contents = [Content User (Message input) context.contextSessionName time]
  push @p contents
  call @p

runPromptWithValidation ::
  forall validation p m a.
  ( MonadIO m,
    MonadFail m,
    PersistentBackend p,
    Tool validation,
    A.FromJSON validation,
    A.FromJSON (Output validation),
    A.ToJSON validation,
    A.ToJSON (Output validation)
  ) =>
  [ToolProxy] ->
  [CustomInstructionProxy] ->
  Text ->
  API.CreateChatCompletionRequest ->
  Text ->
  m (Maybe validation)
runPromptWithValidation tools customs sessionName req input = do
  let valid = ToolProxy (Proxy :: Proxy validation)
  context <- runPrompt @p (valid : tools) customs sessionName req (callPrompt @p input >> getContext)
  case findToolCall valid context.contextBody of
    Just (Content _ (ToolCall id' name' args') _ _) -> do
      let v = (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String validation)
      case v of
        Left err -> do
          liftIO $ putStrLn err
          return Nothing
        Right v' -> return $ Just v'
    _ -> return Nothing

runPrompt :: forall p m a. (MonadIO m, MonadFail m, PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> API.CreateChatCompletionRequest -> Prompt m a -> m a
runPrompt tools customs sessionName req func = do
  let settings = addTools tools (req {API.createChatCompletionRequestTools = Nothing})
  context <- withDB @p $ \conn -> do
    load @p conn sessionName >>= \case
      Just v ->
        return $
          PromptEnv
            { context = v,
              tools = tools,
              customInstructions = customs
            }
      Nothing -> do
        time <- liftIO getCurrentTime
        let init =
              PromptEnv
                { context =
                    Context
                      { contextRequest = settings,
                        contextResponse = Nothing,
                        contextHeader = headers customs,
                        contextBody = [],
                        contextFooter = footers customs,
                        contextTotalTokens = 0,
                        contextSessionName = sessionName,
                        contextCreated = time
                      },
                  tools = tools,
                  customInstructions = customs
                }
        initialize @p conn (init.context)
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
                            API.chatCompletionRequestMessageContentPartText = Nothing,
                            API.chatCompletionRequestMessageContentPartImageUnderscoreurl =
                              Just $
                                API.ChatCompletionRequestMessageContentPartImageImageUrl
                                  { API.chatCompletionRequestMessageContentPartImageImageUrlUrl =
                                      "data:image/" <> type' <> ";base64," <> img,
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
    let res = head (API.createChatCompletionResponseChoices response)
        message = API.createChatCompletionResponseChoicesInnerMessage res
        role = textToUser $ API.chatCompletionResponseMessageRole message
        content = API.chatCompletionResponseMessageContent message
        finishReason = textToFinishReason $ API.createChatCompletionResponseChoicesInnerFinishUnderscorereason res
        v = case API.chatCompletionResponseMessageToolUnderscorecalls message of
          Just toolcalls -> map (\(API.ChatCompletionMessageToolCall id' _ (API.ChatCompletionMessageToolCallFunction name' args')) -> Content role (ToolCall id' name' args') sessionName defaultUTCTime) toolcalls
          Nothing -> [Content role (Message (fromMaybe "" content)) sessionName defaultUTCTime]
     in (v, finishReason)

runRequest :: (ChatCompletion a) => Text -> API.CreateChatCompletionRequest -> a -> IO ((a, FinishReason), API.CreateChatCompletionResponse)
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

  lookupEnv "OPENAI_DEBUG" >>= \case
    Just "1" -> do
      liftIO $ do
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""
    _ -> return ()
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

fromModel :: Text -> API.CreateChatCompletionRequest
fromModel model =
  defaultRequest
    { API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel model
    }
