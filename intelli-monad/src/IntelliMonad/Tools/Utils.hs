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

module IntelliMonad.Tools.Utils where

import Control.Monad (forM)
import Control.Monad.IO.Class
import Data.Aeson (encode)
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import Data.Maybe (catMaybes)
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text.Encoding as T
import Data.Time
import IntelliMonad.Types
import qualified OpenAI.Types as API

addTools :: [ToolProxy] -> API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest
addTools [] v = v
addTools (tool : tools') v =
  case tool of
    (ToolProxy (_ :: Proxy a)) -> addTools tools' (toolAdd @a v)

toolExec' ::
  forall t p m.
  (PersistentBackend p, MonadIO m, MonadFail m, Tool t, A.FromJSON t, A.ToJSON (Output t)) =>
  Text ->
  Text ->
  Text ->
  Text ->
  Prompt m (Maybe Content)
toolExec' sessionName id' name' args' = do
  if name' == toolFunctionName @t
    then case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String t) of
      Left _ -> return Nothing
      Right input -> do
        output <- toolExec @t @p @m input
        time <- liftIO getCurrentTime
        return $ Just $ (Content Tool (ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))) sessionName time)
    else return Nothing

(<||>) ::
  forall m.
  (MonadIO m, MonadFail m) =>
  (Text -> Text -> Text -> Text -> Prompt m (Maybe Content)) ->
  (Text -> Text -> Text -> Text -> Prompt m (Maybe Content)) ->
  Text ->
  Text ->
  Text ->
  Text ->
  Prompt m (Maybe Content)
(<||>) tool0 tool1 sessionName id' name' args' = do
  a <- tool0 sessionName id' name' args'
  case a of
    Just v -> return (Just v)
    Nothing -> tool1 sessionName id' name' args'

mergeToolCall :: forall p m. (PersistentBackend p, MonadIO m, MonadFail m) => [ToolProxy] -> Text -> Text -> Text -> Text -> Prompt m (Maybe Content)
mergeToolCall [] _ _ _ _ = return Nothing
mergeToolCall (tool : tools') sessionName id' name' args' = do
  case tool of
    (ToolProxy (_ :: Proxy a)) -> (toolExec' @a @p <||> mergeToolCall @p tools') sessionName id' name' args'

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

tryToolExec :: forall p m. (PersistentBackend p, MonadIO m, MonadFail m) => [ToolProxy] -> Text -> Contents -> Prompt m Contents
tryToolExec tools sessionName contents = do
  cs <- forM (filterToolCall contents) $ \(Content _ (ToolCall id' name' args') _ _) -> do
    mergeToolCall @p tools sessionName id' name' args'
  return $ catMaybes cs

findToolCall :: ToolProxy -> Contents -> Maybe Content
findToolCall _ [] = Nothing
findToolCall t@(ToolProxy (Proxy :: Proxy a)) (c : cs) =
  case c of
    Content _ (Message _) _ _ -> findToolCall t cs
    Content _ (Image _ _) _ _ -> findToolCall t cs
    Content _ (ToolCall _ name' _) _ _ ->
      if name' == toolFunctionName @a
        then Just c
        else findToolCall t cs
    Content _ (ToolReturn _ _ _) _ _ -> findToolCall t cs
