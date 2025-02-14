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

import Control.Monad (forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (MonadTrans, lift)
import Control.Monad.Trans.State (get, put)
import qualified Data.Aeson as A
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Void
import Database.Persist.Sqlite (SqliteConf)
import GHC.IO.Exception
import IntelliMonad.Config
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Repl
import IntelliMonad.Tools
import IntelliMonad.Types
import qualified OpenAI.Types as API
import Options.Applicative
import System.Console.Haskeline
import System.Environment (lookupEnv)
import System.IO (hClose)
import System.IO.Temp
import System.Process
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Char.Lexer as L

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

runCmd :: forall p. (PersistentBackend p) => ReplCommand -> IO ()
runCmd cmd = do
  config <- readConfig
  let tools = defaultTools
      customs = []
      sessionName = "default"
      defaultReq =
        defaultRequest
          { API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel config.model
          }
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
  model <- do
    lookupEnv "OPENAI_MODEL" >>= \case
      Just model -> return $ T.pack model
      --      Nothing -> return "gpt-4-vision-preview"
      Nothing -> return "gpt-4"
  cmd <- customExecParser (prefs showHelpOnEmpty) (info (helper <*> opts) (fullDesc <> progDesc "intelli-monad"))
  runCmd @SqliteConf cmd
