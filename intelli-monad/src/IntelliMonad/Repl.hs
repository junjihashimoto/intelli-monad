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
import qualified Data.Aeson as A
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Data.Yaml as Y
import qualified Data.Yaml.Pretty as Y
import qualified Data.ByteString as BS
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.Void
import GHC.IO.Exception
import IntelliMonad.Persist
import IntelliMonad.Prompt hiding (user, system, assistant)
import IntelliMonad.Types
import qualified OpenAI.Types as API
import System.Console.Haskeline
import System.Environment (lookupEnv)
import System.IO (hClose)
import System.IO.Temp
import System.Process
import Text.Megaparsec
import Text.Megaparsec.Char
import Text.Megaparsec.Char.Lexer as L
import IntelliMonad.Config

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
                return $ CopySession from to
            )
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

editWithEditor :: forall m. (MonadIO m, MonadFail m) => m (Maybe T.Text)
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

editRequestWithEditor :: forall m. (MonadIO m, MonadFail m) => LLMRequest -> m (Maybe LLMRequest)
editRequestWithEditor req = do
  liftIO $ withSystemTempFile "tempfile.yaml" $ \filePath fileHandle -> do
    hClose fileHandle
    BS.writeFile filePath $ Y.encodePretty Y.defConfig req
    editor <- do
      lookupEnv "EDITOR" >>= \case
        Just editor' -> return editor'
        Nothing -> return "vim"
    code <- system (editor <> " " <> filePath)
    case code of
      ExitSuccess -> do
        newReq <- Y.decodeFileEither @LLMRequest filePath
        case newReq of
          Right newReq' -> return $ Just newReq'
          Left err -> do
            print err
            return Nothing
      ExitFailure _ -> return Nothing

editContentsWithEditor :: forall m. (MonadIO m, MonadFail m) => Contents -> m (Maybe Contents)
editContentsWithEditor contents = do
  liftIO $ withSystemTempFile "tempfile.yaml" $ \filePath fileHandle -> do
    hClose fileHandle
    BS.writeFile filePath $ Y.encodePretty Y.defConfig contents
    editor <- do
      lookupEnv "EDITOR" >>= \case
        Just editor' -> return editor'
        Nothing -> return "vim"
    code <- system (editor <> " " <> filePath)
    case code of
      ExitSuccess -> do
        newContents <- Y.decodeFileEither @Contents filePath
        case newContents of
          Right newContents' -> return $ Just newContents'
          Left err -> do
            print err
            return Nothing
      ExitFailure _ -> return Nothing

runCmd' :: forall p. (PersistentBackend p) => Either (ParseErrorBundle Text Void) ReplCommand -> Maybe (Prompt (InputT IO) ()) -> Prompt (InputT IO) ()
runCmd' cmd ret = do
  let repl = case ret of
        Just ret' -> ret'
        Nothing -> return ()
  case cmd of
    Left err -> do
      liftIO $ print err
      repl
    Right Quit -> return ()
    Right Clear -> do
      clear @p
      repl
    Right ShowContents -> do
      context <- getContext
      showContents context.contextBody
      repl
    Right ShowUsage -> do
      context <- getContext
      liftIO $ do
        print context.contextTotalTokens
      repl
    Right ShowRequest -> do
      prev <- getContext
      let req = toRequest prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
      liftIO $ do
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""
      repl
    Right ShowContext -> do
      prev <- getContext
      liftIO $ do
        putStrLn $ show prev
      repl
    Right ShowSession -> do
      prev <- getContext
      liftIO $ do
        T.putStrLn $ prev.contextSessionName
      repl
    Right ListSessions -> do
      liftIO $ do
        list <- withDB @p $ \conn -> listSessions @p conn
        forM_ list $ \sessionName' -> T.putStrLn sessionName'
      repl
    Right (CopySession from' to') -> do
      liftIO $ do
        withDB @p $ \conn -> do
          mv <- load @p conn from'
          case mv of
            Just v -> do
              _ <- save @p conn (v {contextSessionName = to'})
              return ()
            Nothing -> T.putStrLn $ "Failed to load " <> from'
      repl
    Right (DeleteSession session) -> do
      withDB @p $ \conn -> deleteSession @p conn session
      repl
    Right (SwitchSession session) -> do
      mv <- withDB @p $ \conn -> load @p conn session
      case mv of
        Just v -> do
          (env :: PromptEnv) <- get
          put $ env {context = v}
        Nothing -> liftIO $ T.putStrLn $ "Failed to load " <> session
      repl
    Right (ReadImage imagePath) -> do
      callWithImage @p imagePath >>= showContents
      repl
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
      repl
    Right (UserInput input) -> do
      callWithText @p input >>= showContents
      repl
    Right Edit -> do
      -- Open a temporary file with the default editor of the system.
      -- Then send it as user input.
      editWithEditor >>= \case
        Just input -> do
          callWithText @p input >>= showContents
          repl
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          repl
    Right EditRequest -> do
      -- Open a json file of request and edit it with the default editor of the system.
      -- Then, read the file and parse it as a request.
      -- Finally, update the context with the new request.
      prev <- getContext
      let req = toRequest prev.contextRequest (prev.contextHeader <> prev.contextBody <> prev.contextFooter)
      editRequestWithEditor req >>= \case
        Just req' -> do
          let newContext = prev {contextRequest = req'}
          setContext @p newContext
          repl
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          repl
    Right EditContents -> do
      prev <- getContext
      editContentsWithEditor prev.contextBody >>= \case
        Just contents' -> do
          let newContext = prev {contextBody = contents'}
          setContext @p newContext
          repl
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          repl
    Right EditHeader -> do
      prev <- getContext
      editContentsWithEditor prev.contextHeader >>= \case
        Just contents' -> do
          let newContext = prev {contextHeader = contents'}
          setContext @p newContext
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          repl
    Right EditFooter -> do
      prev <- getContext
      editContentsWithEditor prev.contextFooter >>= \case
        Just contents' -> do
          let newContext = prev {contextFooter = contents'}
          setContext @p newContext
          repl
        Nothing -> do
          liftIO $ putStrLn "Failed to open the editor."
          repl
    Right (Repl sessionName) -> do
      runRepl' @p
    Right (ListKeys) -> do
      liftIO $ do
        list <- withDB @p $ \conn -> listKeys @p conn
        forM_ list $ \(KeyName namespace keyName) -> T.putStrLn $ namespace <> ":" <> keyName
      repl
    Right (GetKey namespace keyName) -> do
      namespace' <- case namespace of
        Just v -> return v
        Nothing -> getSessionName
      mv <- withDB @p $ \conn -> getKey @p conn (KeyName namespace' keyName)
      case mv of
        Just v -> do
          liftIO $ T.putStrLn v
        Nothing -> do
          liftIO $ T.putStrLn $ "Failed to get " <> keyName
      repl
    Right (SetKey namespace keyName keyValue) -> do
      namespace' <- case namespace of
        Just v -> return v
        Nothing -> getSessionName
      withDB @p $ \conn -> setKey @p conn (KeyName namespace' keyName) keyValue
      repl
    Right (DeleteKey namespace keyName) -> do
      namespace' <- case namespace of
        Just v -> return v
        Nothing -> getSessionName
      withDB @p $ \conn -> deleteKey @p conn (KeyName namespace' keyName)
      repl

runRepl' :: forall p. (PersistentBackend p) => Prompt (InputT IO) ()
runRepl' = do
  cmd <- getUserCommand @p
  runCmd' @p cmd (Just (runRepl' @p))

runRepl :: forall p. (PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> LLMRequest -> Contents -> IO ()
runRepl tools customs sessionName defaultReq contents = do
  config <- readConfig
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intelli-monad.history",
          autoAddHistory = True
        }
    )
    (runPrompt @p tools customs sessionName defaultReq (push @p contents >> runRepl' @p))
