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

module IntelliMonad.Cmd where

import Prelude (Bool(True), Either(Right), IO, ($), (<>), (<*>), (>>=), (<$>), pure, return)

import Data.Maybe (Maybe(Just, Nothing))

import Data.Text (pack)

import Database.Persist.Sqlite (SqliteConf)

import Options.Applicative (Parser, argument, command, customExecParser, fullDesc, helper, info, metavar, optional, prefs, progDesc, showHelpOnEmpty, str, subparser)

import System.Console.Haskeline (Settings(Settings, autoAddHistory, complete, historyFile), completeFilename, runInputT)

import System.Environment (lookupEnv)

import IntelliMonad.BaseTypes (PersistentBackend)

import IntelliMonad.Config (model, readConfig)

import IntelliMonad.Prompt (runPrompt)

import IntelliMonad.Repl (ReplCommand(Help, Clear, CopySession, DeleteKey, DeleteSession, Edit, EditContents, EditFooter, EditHeader, EditRequest, GetKey, ListKeys, ListSessions, ReadImage, Repl, SetKey, ShowContents, ShowContext, ShowRequest, ShowSession, ShowUsage, SwitchSession, UserInput), runCmd')

import IntelliMonad.Tools (defaultTools)

import IntelliMonad.Types (defaultRequest)

opts :: Options.Applicative.Parser ReplCommand
opts =
  subparser
    ( command "repl" (info (Repl <$> argument str (metavar "SESSION_NAME")) (progDesc "Start the repl"))
        <> command "clear" (info (pure Clear) (progDesc "Clear the contents"))
        <> command "show-contents" (info (pure ShowContents) (progDesc "Show the contents"))
        <> command "show-usage" (info (pure ShowUsage) (progDesc "Show the usage"))
        <> command "show-request" (info (pure ShowRequest) (progDesc "Show the request"))
        <> command "show-context" (info (pure ShowContext) (progDesc "Show the context"))
        <> command "show-session" (info (pure ShowSession) (progDesc "Show the session"))
        <> command "edit" (info (pure Edit) (progDesc "Edit the contents"))
        <> command "edit-request" (info (pure EditRequest) (progDesc "Edit the config of the current session"))
        <> command "edit-contents" (info (pure EditContents) (progDesc "Edit the contents of the current session"))
        <> command "edit-header" (info (pure EditHeader) (progDesc "Edit the header of the current session"))
        <> command "edit-footer" (info (pure EditFooter) (progDesc "Edit the footer of the current session"))
        <> command "list-sessions" (info (pure ListSessions) (progDesc "List all sessions"))
        <> command "copy-session" (info (CopySession <$> argument str (metavar "FROM") <*> argument str (metavar "TO")) (progDesc "Copy the session"))
        <> command "delete-session" (info (DeleteSession <$> argument str (metavar "SESSION_NAME")) (progDesc "Delete the session"))
        <> command "switch-session" (info (SwitchSession <$> argument str (metavar "SESSION_NAME")) (progDesc "Switch the session"))
        <> command "read-image" (info (ReadImage <$> argument str (metavar "IMAGE_PATH")) (progDesc "Read the image and call a prompt"))
        <> command "read-input" (info (UserInput <$> argument str (metavar "USER_INPUT")) (progDesc "User input as a text and call a prompt"))
        <> command "list-keys" (info (pure ListKeys) (progDesc "List all keys"))
        <> command "get-key" (info (GetKey <$> optional (argument str (metavar "NAMESPACE")) <*> argument str (metavar "KEY_NAME")) (progDesc "Get the key"))
        <> command "set-key" (info (SetKey <$> optional (argument str (metavar "NAMESPACE")) <*> argument str (metavar "KEY_NAME") <*> argument str (metavar "KEY_VALUE")) (progDesc "Set the key"))
        <> command "delete-key" (info (DeleteKey <$> optional (argument str (metavar "NAMESPACE")) <*> argument str (metavar "KEY_NAME")) (progDesc "Delete the key"))
        <> command "help" (info (pure Help) (progDesc "Show the help"))
    )

-- FIXME: hard coded defaults.
runCmd :: forall p. (PersistentBackend p) => ReplCommand -> IO ()
runCmd cmd = do
  let tools = defaultTools
      customs = []
      sessionName = "default"
      defaultReq = defaultRequest
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intelli-monad.history",
          autoAddHistory = True
        }
    )
    (runPrompt @p tools customs sessionName defaultReq (runCmd' @p (Right cmd) Nothing))

main :: IO ()
main = do
  -- Parse our config file. We only use the model, at this point.
  config <- readConfig

  -- FIXME: has no effect. thread this through.
  -- Determine what model to use by default. Use what's in the config file by default, but override with an environment variable (OPENAI_MODEL).
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ pack model
      Nothing -> return config.model

  cmd <- customExecParser (prefs showHelpOnEmpty) (info (helper <*> opts) (fullDesc <> progDesc "intelli-monad"))
  runCmd @SqliteConf cmd
