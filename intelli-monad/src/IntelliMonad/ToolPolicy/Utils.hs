{-# LANGUAGE RankNTypes #-}

{-# LANGUAGE TypeApplications #-}

{-# LANGUAGE ScopedTypeVariables #-}

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

module IntelliMonad.ToolPolicy.Utils
  (
    ToolRegistry,
    addTool,
    checkPolicy,
    changeToolPolicy,
    defaultRegistry,
    getTools
  ) where

import Control.Monad.IO.Class (liftIO, MonadIO)

import Control.Monad.Trans.State (get)
                                 
import Data.Aeson (FromJSON, ToJSON, encode, eitherDecode)

import Data.ByteString (ByteString, fromStrict, toStrict)

import qualified Data.Map as DM (insert, lookup)

import Data.Map (Map, empty, insert, assocs)

import Data.Proxy (Proxy)

import Data.Text (Text, pack, strip, toLower)

import IntelliMonad.BaseTypes (Content(Content), Context, HasFunctionObject(getFunctionDescription), Message(ToolCall, ToolReturn), Prompt, PromptEnv(inputCallback, outputCallback), ToolProxy(ToolProxy), contextToolbox, Tool(toolFunctionName), User(Tool))

import IntelliMonad.ToolPolicy.Types (ToolEntry(ToolEntry), ToolPolicy(Allow, Ask, Deny), ToolRegistry(ToolRegistry))

-- Add a tool to a registry.
addTool :: Text -> Text -> ToolPolicy -> ToolRegistry -> ToolRegistry
addTool name description policy (ToolRegistry rawRegistry) = ToolRegistry $ insert name (ToolEntry description policy) rawRegistry 

addToolProxy :: ToolProxy -> ToolRegistry -> ToolRegistry
addToolProxy (ToolProxy (_:: Proxy a)) registryIn = addTool (toolFunctionName @a) (pack $ getFunctionDescription @a) Ask registryIn

changeToolPolicy :: ToolRegistry -> Text -> ToolPolicy -> Maybe ToolRegistry
changeToolPolicy (ToolRegistry rawRegistry) toolName toolPolicy = do
  case DM.lookup toolName rawRegistry of
    Just (ToolEntry desc _) -> Just $ ToolRegistry $ DM.insert toolName (ToolEntry desc toolPolicy) rawRegistry
    Nothing -> Nothing

getTools :: Context -> [(Text, ToolEntry)]
getTools context = assocs $ (\(ToolRegistry rawRegistry) -> rawRegistry) $ contextToolbox context

defaultRegistry :: [ToolProxy] -> ToolRegistry
defaultRegistry [] = error "tried to build an empty default registry."
defaultRegistry (tool:[]) = addToolProxy tool $ ToolRegistry empty
defaultRegistry (tool:moreTools) = addToolProxy tool (defaultRegistry moreTools)

checkPolicy :: forall m. (MonadIO m, MonadFail m) => ToolRegistry -> Content -> Prompt m Content
checkPolicy (ToolRegistry rawRegistry) cin@(Content _ (ToolCall tcId tcName tcArgs) session timeout) = do
  env <- get
  case DM.lookup tcName rawRegistry of
    Just (ToolEntry _ Allow) -> do
      -- Match the style of:
      -- tool: [tool delete_key returning {"code":0,"stderr":""}]
      -- assistant: [assistant calling get_key with {"key":"missing_key"}]
      liftIO $ env.outputCallback $ "[" <> tcName <> " tool allowed by policy.]\n"
      return cin
    Just (ToolEntry _ (Deny reason)) -> do
      liftIO $ env.outputCallback $ "[" <> tcName <> " tool denied by policy.]\n"
      return $ Content Tool (ToolReturn tcId tcName ("Denied by policy. Reason: \"" <> reason <> "\"")) session timeout
    Just (ToolEntry _ Ask) -> do
      liftIO $ env.outputCallback $ "--- Tool Request ---\n"
                                 <> "Name: " <> tcName <> "\n"
                                 <> "Arguments: " <> tcArgs <> "\n"
                                 <> "--------------------\n"
                                 <> "Accept? [y/N]: "
      answer <- liftIO $ env.inputCallback ("" :: Text)
      case fmap (toLower . strip) answer of
        Just "y" -> return cin
        _        -> do
          liftIO $ env.outputCallback "Denied by user.\n"
          return $ Content Tool (ToolReturn tcId tcName "Denied by user.") session timeout
    Nothing -> do
      liftIO $ env.outputCallback $ "[ " <> tcName <> " Denied by default (no policy). ]\n"
      return $ Content Tool (ToolReturn tcId tcName "Denied by policy") session timeout
     
