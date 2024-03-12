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

module IntelliMonad.Repl where

import Control.Monad (forM, forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (MonadTrans, lift)
import Control.Monad.Trans.State (put)
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Time
import Data.Void
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Tools
import IntelliMonad.Types
import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import System.Console.Haskeline
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Char.Lexer as L

import Database.Persist
import Database.Persist.PersistValue
import Database.Persist.Sqlite
import Database.Persist.TH

type Parser = Parsec Void Text

data ReplCommand
  = Quit
  | Clear
  | ShowContents
  | ShowUsage
  | ShowRequest
  | ShowContext
  | ShowSession
  | ListSessions
  | CopySession (Text, Text)
  | DeleteSession Text
  | SwitchSession Text
  | Help
  deriving (Eq, Show)

parseRepl :: Parser ReplCommand
parseRepl =
  (try (lexm (string ":quit")) >> pure Quit)
    <|> (try (lexm (string ":clear")) >> pure Clear)
    <|> (try (lexm (string ":show") >> lexm (string "contents")) >> pure ShowContents)
    <|> (try (lexm (string ":show") >> lexm (string "usage")) >> pure ShowUsage)
    <|> (try (lexm (string ":show") >> lexm (string "request")) >> pure ShowRequest)
    <|> (try (lexm (string ":show") >> lexm (string "context")) >> pure ShowContext)
    <|> (try (lexm (string ":show") >> lexm (string "session")) >> pure ShowSession)
    <|> (try (lexm (string ":list") >> lexm (string "sessions")) >> pure ListSessions)
    <|> (try (lexm (string ":list")) >> pure ListSessions)
    <|> ( try
            ( lexm (string ":copy") >> lexm (string "session") >> do
                from <- T.pack <$> lexm sessionName
                to <- T.pack <$> lexm sessionName
                return (from, to)
            )
            >>= pure . CopySession
        )
    <|> (try (lexm (string ":delete") >> lexm (string "session") >> lexm sessionName) >>= pure . DeleteSession . T.pack)
    <|> (try (lexm (string ":switch") >> lexm (string "session") >> lexm sessionName) >>= pure . SwitchSession . T.pack)
    <|> (try (lexm (string ":help")) >> pure Help)
  where
    sc = L.space space1 empty empty
    lexm = lexeme sc
    sessionName = many alphaNumChar

getTextInputLine :: (MonadTrans t) => String -> t (InputT IO) (Maybe T.Text)
getTextInputLine prompt = fmap (fmap T.pack) (lift $ getInputLine prompt)

runRepl :: Text -> API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl sessionName defaultReq contents = do
  let settings =
        toolAdd @BashInput $
          toolAdd @TextToSpeechInput $
            defaultReq
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intellimonad.history",
          autoAddHistory = True
        }
    )
    (runPrompt sessionName settings (push contents >> loop))
  where
    loop :: Prompt (InputT IO) ()
    loop = do
      minput <- getTextInputLine "% "
      case minput of
        Nothing -> return ()
        Just input ->
          if T.isPrefixOf ":" input then
            case runParser parseRepl "stdin" input of
              Left err -> do
                liftIO $ print err
                loop
              Right Quit -> return ()
              Right Clear -> do
                prev <- getContext
                setContext $ prev {contextBody = []}
                loop
              Right ShowContents -> do
                context <- getContext
                showContents context.contextBody
                loop
              Right ShowUsage -> do 
                context <- getContext
                liftIO $ do
                  print context.contextTotalTokens
                loop
              Right ShowRequest -> do 
                prev <- getContext
                let req = toRequest prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
                liftIO $ do
                  BS.putStr $ BS.toStrict $ encodePretty req
                  T.putStrLn ""
                loop
              Right ShowContext -> do 
                prev <- getContext
                liftIO $ do
                  putStrLn $ show prev
                loop
              Right ShowSession -> do 
                prev <- getContext
                liftIO $ do
                  T.putStrLn $ prev.contextSessionName
                loop
              Right ListSessions -> do
                liftIO $ do
                  list <- withDB $ \conn -> listSessions @SqliteConf conn
                  forM_ list $ \sessionName -> T.putStrLn sessionName
                loop
              Right (CopySession (from',to')) -> do
                liftIO $ do
                  withDB $ \conn -> do
                    mv <- load @SqliteConf conn from'
                    case mv of
                      Just v -> do
                        _ <- save @SqliteConf conn (v {contextSessionName = to'})
                        return ()
                      Nothing -> T.putStrLn $ "Failed to load " <> from'
                loop
              Right (DeleteSession session) -> do
                withDB $ \conn -> deleteSession @SqliteConf conn session
                loop
              Right (SwitchSession session) -> do
                mv <- withDB $ \conn -> load @SqliteConf conn session
                case mv of
                  Just v -> do
                    put v
                  Nothing -> liftIO $ T.putStrLn $ "Failed to load " <> session
                loop
              Right Help -> do
                liftIO $ do
                  putStrLn ":quit"
                  putStrLn ":clear"
                  putStrLn ":show contents"
                  putStrLn ":show usage"
                  putStrLn ":show request"
                  putStrLn ":show context"
                  putStrLn ":show session"
                  putStrLn ":list sessions"
                  putStrLn ":copy session <from> <to>"
                  putStrLn ":delete session <session name>"
                  putStrLn ":switch session <session name>"
                  putStrLn ":help"
                loop
          else do 
            time <- liftIO getCurrentTime
            context <- getContext
            let contents = [Content User (Message input) context.contextSessionName time]
            push contents
            ret <- call
            showContents ret
            loop
