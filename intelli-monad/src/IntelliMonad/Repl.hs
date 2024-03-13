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

import Control.Monad (forM_)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (MonadTrans, lift)
import Control.Monad.Trans.State (get, put)
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.ByteString as BS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Data.Time
import Data.Void
-- import Database.Persist hiding (get)
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import qualified OpenAI.Types as API
import System.Console.Haskeline
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Char.Lexer as L

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
  | ReadImage Text
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
    <|> (try (lexm (string ":read") >> lexm (string "image") >> lexm imagePath) >>= pure . ReadImage . T.pack)
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
    imagePath = many (alphaNumChar <|> char '.' <|> char '/' <|> char '-')

getTextInputLine :: (MonadTrans t) => String -> t (InputT IO) (Maybe T.Text)
getTextInputLine prompt = fmap (fmap T.pack) (lift $ getInputLine prompt)

runRepl :: forall p. (PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl tools customs sessionName defaultReq contents = do
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intelli-monad.history",
          autoAddHistory = True
        }
    )
    (runPrompt @p tools customs sessionName defaultReq (push @p contents >> loop))
  where
    loop :: Prompt (InputT IO) ()
    loop = do
      minput <- getTextInputLine "% "
      case minput of
        Nothing -> return ()
        Just input ->
          if T.isPrefixOf ":" input
            then case runParser parseRepl "stdin" input of
              Left err -> do
                liftIO $ print err
                loop
              Right Quit -> return ()
              Right Clear -> do
                clear @p
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
                  list <- withDB @p $ \conn -> listSessions @p conn
                  forM_ list $ \sessionName' -> T.putStrLn sessionName'
                loop
              Right (CopySession (from', to')) -> do
                liftIO $ do
                  withDB @p $ \conn -> do
                    mv <- load @p conn from'
                    case mv of
                      Just v -> do
                        _ <- save @p conn (v {contextSessionName = to'})
                        return ()
                      Nothing -> T.putStrLn $ "Failed to load " <> from'
                loop
              Right (DeleteSession session) -> do
                withDB @p $ \conn -> deleteSession @p conn session
                loop
              Right (SwitchSession session) -> do
                mv <- withDB @p $ \conn -> load @p conn session
                case mv of
                  Just v -> do
                    (env :: PromptEnv) <- get
                    put $ env {context = v}
                  Nothing -> liftIO $ T.putStrLn $ "Failed to load " <> session
                loop
              Right (ReadImage imagePath) -> do
                callWithImage @p imagePath >>= showContents
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
              callWithText @p input >>= showContents
              loop
