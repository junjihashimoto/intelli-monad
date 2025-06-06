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

import Control.Monad (forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.State (get, put, runStateT)
import qualified Data.Aeson as A
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Base64 as Base64
import qualified Data.Map as M
import Data.Maybe (fromMaybe)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Data.Time
import IntelliMonad.Config
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
  _ <- withDB @p $ \conn -> save @p (conn :: Conn p) context
  return ()

getSessionName :: (MonadIO m, MonadFail m) => Prompt m Text
getSessionName = contextSessionName <$> getContext

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
            contextRequest = updateRequest prev.contextRequest (prev.contextHeader <> nextContents <> prev.contextFooter)
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
            contextRequest = updateRequest prev.contextRequest (prev.contextHeader <> nextContents <> prev.contextFooter)
          }
  setContext @p next

  withDB @p $ \conn -> saveContents @p conn contents
  return ()

callPreHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Prompt m ()
callPreHook = do
  env <- get
  forM_ env.hooks $ \(HookProxy (h :: h)) -> do
    preHook @h @p h
  

callPostHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Prompt m ()
callPostHook = do
  env <- get
  forM_ env.hooks $ \(HookProxy (h :: h)) -> do
    postHook @h @p h

call :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Prompt m Contents
call = loop []
  where
    loop ret = do
      callPreHook @p
      prev <- getContext
      ((contents, finishReason), res) <- liftIO $ runRequest prev.contextSessionName prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
      let current_total_tokens = 0 -- fromMaybe 0 $ API.completionUsageTotalUnderscoretokens <$> API.createChatCompletionResponseUsage res
          next =
            prev
              { contextResponse = Just res,
                contextTotalTokens = current_total_tokens
              }
      setContext @p next
      push @p contents
      callPostHook @p

      let ret' = ret <> contents

      case finishReason of
        Stop -> return ret'
        ToolCalls -> callTool next contents ret'
        FunctionCall -> callTool next contents ret'
        Length -> loop ret'
        _ -> return ret'

    callTool next contents ret = do
      showContents contents
      env <- get
      retTool <- tryToolExec @p env.tools next.contextSessionName contents
      showContents retTool
      pushToolReturn @p retTool
      v <- call @p
      return $ ret <> retTool <> v

callWithText :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Text -> Prompt m Contents
callWithText input = do
  time <- liftIO getCurrentTime
  context <- getContext
  let contents = [Content User (Message input) context.contextSessionName time]
  push @p contents
  call @p

callWithContents :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Contents -> Prompt m Contents
callWithContents input = do
  push @p input
  call @p

callWithValidation ::
  forall validation p m.
  ( MonadIO m,
    MonadFail m,
    PersistentBackend p,
    Tool validation,
    A.FromJSON validation,
    A.FromJSON (Output validation),
    A.ToJSON validation,
    A.ToJSON (Output validation),
    HasFunctionObject validation,
    JSONSchema validation
  ) =>
  Contents ->
  Prompt m (Maybe validation)
callWithValidation contents = do
  let valid = ToolProxy (Proxy :: Proxy validation)
  case findToolCall valid contents of
    Just (Content _ (ToolCall _ _ args') _ _) -> do
      let v = (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String validation)
      case v of
        Left err -> do
          liftIO $ putStrLn err
          return Nothing
        Right v' -> return $ Just v'
    _ -> return Nothing

runPromptWithValidation ::
  forall validation p m.
  ( MonadIO m,
    MonadFail m,
    PersistentBackend p,
    Tool validation,
    A.FromJSON validation,
    A.FromJSON (Output validation),
    A.ToJSON validation,
    A.ToJSON (Output validation),
    HasFunctionObject validation,
    JSONSchema validation
  ) =>
  [ToolProxy] ->
  [CustomInstructionProxy] ->
  Text ->
  LLMRequest ->
  Text ->
  m (Maybe validation)
runPromptWithValidation tools customs sessionName req input = do
  let valid = ToolProxy (Proxy :: Proxy validation)
  runPrompt @p (valid : tools) customs sessionName req (callWithText @p input >>= callWithValidation @validation @p)

user :: Text -> Content
user input = Content User (Message input) "default" defaultUTCTime

system :: Text -> Content
system input = Content System (Message input) "default" defaultUTCTime

assistant :: Text -> Content
assistant input = Content Assistant (Message input) "default" defaultUTCTime

generate ::
  forall input output m p.
  ( MonadIO m,
    MonadFail m,
    p ~ StatelessConf,
    A.ToJSON input,
    A.FromJSON input,
    JSONSchema input,
    Tool output,
    A.FromJSON output,
    A.FromJSON (Output output),
    A.ToJSON output,
    A.ToJSON (Output output),
    HasFunctionObject output,
    JSONSchema output
  ) => Contents -> input -> m (Maybe output)
generate userContext input = do
  let valid = ToolProxy (Proxy :: Proxy output)
      req = (fromModel "gpt-4")
  runPrompt @p [valid] [] "default" req $ do
    time <- liftIO getCurrentTime
    context <- getContext
    let schemaText :: Text
        schemaText = T.decodeUtf8Lenient $ BS.toStrict $ A.encode $ toAeson (schema @input)
        inputText :: Text
        inputText = T.decodeUtf8Lenient $ BS.toStrict $ A.encode input
        contents = userContext ++
          [ user ("#User-input format is as follows:\n" <> schemaText)
          , user ("#User-input is as follows:\n" <> inputText)
          , user ("#Save the processing results using " <> toolFunctionName @output <> " function")
          ]
    
    push @p contents
    call @p >>= callWithValidation @output @StatelessConf

initializePrompt :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> LLMRequest -> m PromptEnv
initializePrompt tools customs sessionName req = do
--  config <- readConfig
  let settings = addTools tools req
  withDB @p $ \conn -> do
    load @p conn sessionName >>= \case
      Just v ->
        return $
          PromptEnv
            { context = v,
              tools = tools,
              customInstructions = customs,
              backend = (PersistProxy (config @p)),
              hooks = []
            }
      Nothing -> do
        time <- liftIO getCurrentTime
        let init' =
              PromptEnv
                { context =
                    Context
                      { contextRequest = settings,
                        contextResponse = Nothing,
                        contextHeader = headers customs ++ toolHeaders tools,
                        contextBody = [],
                        contextFooter = toolFooters tools ++ footers customs,
                        contextTotalTokens = 0,
                        contextSessionName = sessionName,
                        contextCreated = time
                      },
                  tools = tools,
                  customInstructions = customs,
                  backend = (PersistProxy (config @p)),
                  hooks = []
                }
        initialize @p conn (init'.context)
        return init'

runPrompt :: forall p m a. (MonadIO m, MonadFail m, PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> LLMRequest -> Prompt m a -> m a
runPrompt tools customs sessionName req func = do
  context <- initializePrompt @p tools customs sessionName req
  fst <$> runStateT func context

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

clear :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Prompt m ()
clear = do
  prev <- getContext
  setContext @p $ prev {contextBody = []}

callWithImage :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => Text -> Prompt m Contents
callWithImage imagePath = do
  let tryReadFile = T.decodeUtf8Lenient . Base64.encode <$> BS.readFile (T.unpack imagePath)
      imageType =
        if T.isSuffixOf ".png" imagePath
          then "png"
          else
            if T.isSuffixOf ".jpg" imagePath || T.isSuffixOf ".jpeg" imagePath
              then "jpeg"
              else "jpeg"
  file <- liftIO $ tryReadFile
  time <- liftIO getCurrentTime
  context <- getContext
  let contents' = [Content User (Image imageType file) context.contextSessionName time]
  callWithContents @p contents'
