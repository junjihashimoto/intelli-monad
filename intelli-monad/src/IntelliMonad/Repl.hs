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
import qualified Data.Aeson as A
import qualified Data.ByteString as BS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Void
import IntelliMonad.Persist
import IntelliMonad.Prompt
import IntelliMonad.Types
import qualified OpenAI.Types as API
import System.Console.Haskeline
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Char.Lexer as L
import GHC.IO.Exception
import System.Process
import System.Environment (lookupEnv)
import System.IO.Temp
import System.IO (hClose)

type Parser = Parsec Void Text

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
    <|> (try (lexm (string ":edit") >> lexm (string "request")) >> pure EditRequest)
    <|> (try (lexm (string ":edit") >> lexm (string "contents")) >> pure EditContents)
    <|> (try (lexm (string ":edit") >> lexm (string "header")) >> pure EditHeader)
    <|> (try (lexm (string ":edit") >> lexm (string "footer")) >> pure EditFooter)
    <|> (try (lexm (string ":edit")) >> pure Edit)
  where
    sc = L.space space1 empty empty
    lexm = lexeme sc
    sessionName = many alphaNumChar
    imagePath = many (alphaNumChar <|> char '.' <|> char '/' <|> char '-')

getTextInputLine :: (MonadTrans t) => t (InputT IO) (Maybe T.Text)
getTextInputLine = fmap (fmap T.pack) (lift $ getInputLine "% ")

getUserCommand :: forall p t. (PersistentBackend p, MonadTrans t) => t (InputT IO) (Either (ParseErrorBundle Text Void) ReplCommand)
getUserCommand = do
  minput <- getTextInputLine
  case minput of
    Nothing -> return $ Right Quit
    Just input ->
      if T.isPrefixOf ":" input
        then case runParser parseRepl "stdin" input of
          Right v -> return $ Right v
          Left err -> return $ Left err
        else return $ Right (UserInput input)

editWithEditor :: forall m. ( MonadIO m, MonadFail m) => m (Maybe T.Text)
editWithEditor = do
  liftIO $ withSystemTempFile "tempfile.txt" $ \filePath fileHandle -> do
    hClose fileHandle
    editor <- do
      lookupEnv "EDITOR" >>= \case
        Just editor' -> return editor'
        Nothing -> return "vim"
    code <- system (editor <> " " <> filePath)
    case code of
      ExitSuccess -> Just <$> T.readFile filePath
      ExitFailure _ -> return Nothing

editRequestWithEditor :: forall m. ( MonadIO m, MonadFail m) => API.CreateChatCompletionRequest -> m (Maybe API.CreateChatCompletionRequest)
editRequestWithEditor req = do
  liftIO $ withSystemTempFile "tempfile.json" $ \filePath fileHandle -> do
    hClose fileHandle
    BS.writeFile filePath $ BS.toStrict $ encodePretty req
    editor <- do
      lookupEnv "EDITOR" >>= \case
        Just editor' -> return editor'
        Nothing -> return "vim"
    code <- system (editor <> " " <> filePath)
    case code of
      ExitSuccess -> do
        newReq <- A.decodeFileStrict @API.CreateChatCompletionRequest filePath
        case newReq of
          Just newReq' -> return $ Just newReq'
          Nothing -> return Nothing
      ExitFailure _ -> return Nothing

editContentsWithEditor :: forall m. ( MonadIO m, MonadFail m) => Contents -> m (Maybe Contents)
editContentsWithEditor contents = do
  liftIO $ withSystemTempFile "tempfile.json" $ \filePath fileHandle -> do
    hClose fileHandle
    BS.writeFile filePath $ BS.toStrict $ encodePretty contents
    editor <- do
      lookupEnv "EDITOR" >>= \case
        Just editor' -> return editor'
        Nothing -> return "vim"
    code <- system (editor <> " " <> filePath)
    case code of
      ExitSuccess -> do
        newContents <- A.decodeFileStrict @Contents filePath
        case newContents of
          Just newContents' -> return $ Just newContents'
          Nothing -> return Nothing
      ExitFailure _ -> return Nothing

runRepl' :: forall p. (PersistentBackend p) => Prompt (InputT IO) ()
runRepl' = do
  getUserCommand @p >>= \case
    Left err -> do
      liftIO $ print err
      runRepl' @p
    Right Quit -> return ()
    Right Clear -> do
      clear @p
      runRepl' @p
    Right ShowContents -> do
      context <- getContext
      showContents context.contextBody
      runRepl' @p
    Right ShowUsage -> do
      context <- getContext
      liftIO $ do
        print context.contextTotalTokens
      runRepl' @p
    Right ShowRequest -> do
      prev <- getContext
      let req = toRequest prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
      liftIO $ do
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""
      runRepl' @p
    Right ShowContext -> do
      prev <- getContext
      liftIO $ do
        putStrLn $ show prev
      runRepl' @p
    Right ShowSession -> do
      prev <- getContext
      liftIO $ do
        T.putStrLn $ prev.contextSessionName
      runRepl' @p
    Right ListSessions -> do
      liftIO $ do
        list <- withDB @p $ \conn -> listSessions @p conn
        forM_ list $ \sessionName' -> T.putStrLn sessionName'
      runRepl' @p
    Right (CopySession (from', to')) -> do
      liftIO $ do
        withDB @p $ \conn -> do
          mv <- load @p conn from'
          case mv of
            Just v -> do
              _ <- save @p conn (v {contextSessionName = to'})
              return ()
            Nothing -> T.putStrLn $ "Failed to load " <> from'
      runRepl' @p
    Right (DeleteSession session) -> do
      withDB @p $ \conn -> deleteSession @p conn session
      runRepl' @p
    Right (SwitchSession session) -> do
      mv <- withDB @p $ \conn -> load @p conn session
      case mv of
        Just v -> do
          (env :: PromptEnv) <- get
          put $ env {context = v}
        Nothing -> liftIO $ T.putStrLn $ "Failed to load " <> session
      runRepl' @p
    Right (ReadImage imagePath) -> do
      callWithImage @p imagePath >>= showContents
      runRepl' @p
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
      runRepl' @p
    Right (UserInput input) -> do
      callWithText @p input >>= showContents
      runRepl' @p
    Right Edit -> do
      -- Open a temporary file with the default editor of the system.
      -- Then send it as user input.
      editWithEditor >>= \case
        Just input -> do
          callWithText @p input >>= showContents
          runRepl' @p
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          runRepl' @p
    Right EditRequest -> do
      -- Open a json file of request and edit it with the default editor of the system.
      -- Then, read the file and parse it as a request.
      -- Finally, update the context with the new request.
      prev <- getContext
      let req = toRequest prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
      editRequestWithEditor req >>= \case
        Just req' -> do
          (env :: PromptEnv) <- get
          let newContext = prev {contextRequest = req'}
          put $ env {context = newContext}
          runRepl' @p
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          runRepl' @p
    Right EditContents -> do
      prev <- getContext
      editContentsWithEditor prev.contextBody >>= \case
        Just contents' -> do
          (env :: PromptEnv) <- get
          let newContext = prev {contextBody = contents'}
          put $ env {context = newContext}
          runRepl' @p
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          runRepl' @p
    Right EditHeader -> do
      prev <- getContext
      editContentsWithEditor prev.contextHeader >>= \case
        Just contents' -> do
          (env :: PromptEnv) <- get
          let newContext = prev {contextHeader = contents'}
          put $ env {context = newContext}
          runRepl' @p
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          runRepl' @p    
    Right EditFooter -> do
      prev <- getContext
      editContentsWithEditor prev.contextFooter >>= \case
        Just contents' -> do
          (env :: PromptEnv) <- get
          let newContext = prev {contextFooter = contents'}
          put $ env {context = newContext}
          runRepl' @p
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          runRepl' @p

runRepl :: forall p. (PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl tools customs sessionName defaultReq contents = do
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intelli-monad.history",
          autoAddHistory = True
        }
    )
    (runPrompt @p tools customs sessionName defaultReq (push @p contents >> runRepl' @p))
