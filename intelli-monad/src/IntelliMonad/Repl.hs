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

import Prelude (Bool(True), Either(Left, Right), Eq, IO, Int, Show, (.), ($), (<>), (>>), (<$>), (>>=), fmap, print, pure, putStrLn, return, show)

import Control.Monad (forM_)

import Control.Monad.Fail (MonadFail)

import Control.Monad.IO.Class (MonadIO, liftIO)

import Control.Monad.Trans.Class (MonadTrans, lift)

import Control.Monad.Trans.State (get, put)

import Data.Aeson.Encode.Pretty (encodePretty)

import Data.Maybe (Maybe(Just, Nothing))

import qualified Data.ByteString as BS (putStr, toStrict, writeFile)

import Data.Text (Text)
import qualified Data.Text as T (isPrefixOf, pack, unpack)
import qualified Data.Text.IO as T (putStr, putStrLn, readFile)

import Data.Void (Void)

import qualified Data.Yaml as Y (decodeFileEither)
import qualified Data.Yaml.Pretty as Y (defConfig, encodePretty)

import GHC.IO.Exception (ExitCode(ExitFailure, ExitSuccess))

import qualified Louter.Types.Request as Louter (ChatRequest, reqModel)

import System.Console.Haskeline (InputT, Settings(Settings, autoAddHistory, complete, historyFile), completeFilename, defaultSettings, getExternalPrint, getInputLine, runInputT)

import System.Environment (lookupEnv)

import System.IO (hClose)

import System.IO.Temp (withSystemTempFile)

import System.Process (system)

import Text.Megaparsec (Parsec, ParseErrorBundle, (<|>), anySingle, empty, many, runParser, try)

import Text.Megaparsec.Char (alphaNumChar, char, space1, string)

import Text.Megaparsec.Char.Lexer as L (decimal, lexeme, space)

import IntelliMonad.BaseTypes (ChatCompletion(toRequest), Content(Content), Contents, Context(contextBody, contextFooter, contextHeader, contextRequest, contextSessionName, contextTotalTokens, contextToolbox), CustomInstructionProxy, Message(Message), ToolProxy, PersistentBackend(deleteKey, deleteSession, getKey, listKeys, listSessions, load, save, setKey), Prompt, PromptEnv(context, inputCallback, outputCallback, timeoutSeconds), Unique(KeyName))

import IntelliMonad.Persist (withDB)

import IntelliMonad.Prompt (callWithImage, callWithText, clear, getContext, getSessionName, push, runPrompt, setContext, showContents)

import IntelliMonad.Config (readConfig)
import qualified IntelliMonad.Config as Config (getUseStreaming)

import IntelliMonad.ToolPolicy (getTools, changeToolPolicy)

import IntelliMonad.ToolPolicy.Types (ToolEntry(ToolEntry), ToolPolicy(Allow,Ask,Deny))

type Parser = Parsec Void Text

data ReplCommand
  = Quit
  | Clear
  | ShowContents
  | ShowUsage
  | ShowRequest
  | ShowContext
  | ShowSession
  | Edit
  | EditRequest
  | EditContents
  | EditHeader
  | EditFooter
  | ListSessions
  | ListTools
  | SetModel Text
  | SetToolPolicy ToolPolicy Text
  | SetTimeout Int
  | CopySession
      { sessionNameFrom :: Text,
        sessionNameTo :: Text
      }
  | DeleteSession
      { sessionName :: Text
      }
  | SwitchSession
      { sessionName :: Text
      }
  | ReadImage Text
  | UserInput Text
  | Help
  | Repl
      { sessionName :: Text
      }
  | ListKeys
  | GetKey
      { nameSpace :: Maybe Text,
        keyName :: Text
      }
  | SetKey
      { nameSpace :: Maybe Text,
        keyName :: Text,
        value :: Text
      }
  | DeleteKey
      { nameSpace :: Maybe Text,
        keyName :: Text
      }
  deriving (Eq, Show)

parseRepl :: Parser ReplCommand
parseRepl =
  (try (lexm (string ":quit")) >> pure Quit)
    <|> (try (lexm (string ":clear")) >> pure Clear)
    <|> (try (lexm (string ":model") >> T.pack <$> lexm modelName) >>= pure . SetModel)
    <|> (try (lexm (string ":show") >> lexm (string "contents")) >> pure ShowContents)
    <|> (try (lexm (string ":show") >> lexm (string "usage")) >> pure ShowUsage)
    <|> (try (lexm (string ":show") >> lexm (string "request")) >> pure ShowRequest)
    <|> (try (lexm (string ":show") >> lexm (string "context")) >> pure ShowContext)
    <|> (try (lexm (string ":show") >> lexm (string "session")) >> pure ShowSession)
    <|> (try (lexm (string ":set" ) >> lexm (string "timeout") >> L.decimal) >>= pure . SetTimeout)
    <|> (try (lexm (string ":set" ) >> lexm (string "tool") >> (T.pack <$> lexm toolName >>= \name -> lexm (string "allow") >> return (SetToolPolicy Allow name)) ))
    <|> (try (lexm (string ":set" ) >> lexm (string "tool") >> (T.pack <$> lexm toolName >>= \name -> lexm (string "ask") >> return (SetToolPolicy Ask name)) ))
    <|> (try (lexm (string ":set" ) >> lexm (string "tool") >> (T.pack <$> lexm toolName >>= \name -> lexm (string "deny") >> lexm (many anySingle) >>= \reason -> return (SetToolPolicy (Deny $ T.pack reason) name)) ))
    <|> (try (lexm (string ":read") >> lexm (string "image") >> lexm imagePath) >>= pure . ReadImage . T.pack)
    <|> (try (lexm (string ":list") >> lexm (string "sessions")) >> pure ListSessions)
    <|> (try (lexm (string ":list") >> lexm (string "tools")) >> pure ListTools)
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
    modelName = many (alphaNumChar <|> char '-' <|> char '.' <|> char ':' <|> char '/')
    toolName = many (alphaNumChar <|> char '-' <|> char '.' <|> char ':' <|> char '/' <|> char '_')

getTextInputLine :: (MonadTrans t) => t (InputT IO) (Maybe Text)
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

editWithEditor :: forall m. (MonadIO m, MonadFail m) => m (Maybe Text)
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

editRequestWithEditor :: forall m. (MonadIO m, MonadFail m) => Louter.ChatRequest -> m (Maybe Louter.ChatRequest)
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
        newReq <- Y.decodeFileEither @Louter.ChatRequest filePath
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
    Right (SetModel modelName) -> do
      prev <- getContext
      let req = prev.contextRequest { Louter.reqModel = modelName }
          newContext = prev {contextRequest = req}
      setContext @p newContext
      liftIO $ T.putStrLn $ "Model set to: " <> modelName
      repl
    Right (SetTimeout timeout) -> do
      env <- get
      put $ env { timeoutSeconds = Just timeout }
      liftIO $ T.putStrLn $ "Timeout set to: " <> T.pack (show timeout) <> " seconds"
      repl
    Right (SetToolPolicy policy toolName) -> do
      prev <- getContext
      let
        maybeNewRegistry = changeToolPolicy prev.contextToolbox toolName policy
      case maybeNewRegistry of
        Nothing -> do
          liftIO $ T.putStrLn $ "Tool Policy set failed. Could not find tool: " <> toolName <> "."
        Just newRegistry -> do
          liftIO $ T.putStrLn $ "\"" <> toolName <> "\" policy set to: " <> T.pack (show policy)
          env <- get
          put $ env { context = prev { contextToolbox = newRegistry } }
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
      context <- getContext
      let req = toRequest context.contextRequest (context.contextHeader <> context.contextBody <> context.contextFooter)
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
    Right ListTools -> do
      context <- getContext
      liftIO $ do
        putStrLn $ "Policy | Name -- Description" <> "\n" <> "----------------------------"
        let
          list = getTools context
          showToolEntry (name, (ToolEntry desc policy)) =
            case policy of
              Allow -> T.putStrLn $ "ALLOW  | " <> name <> " -- " <> desc
              Ask -> T.putStrLn $ "ASK    | " <> name <> " -- " <> desc
              (Deny reason) -> T.putStrLn $ "DENY   | " <> name <> " -- " <> desc <> " -- Deny reason: " <> reason
        forM_ list $ \a -> showToolEntry a
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
    Right (UserInput input) -> callInput input >> repl
    Right Edit -> do
      -- Open a temporary file with the default editor of the system.
      -- Then send it as user input.
      editWithEditor >>= \case
        Just input -> callInput input
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
    Right (Repl _) -> do
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
  where
    callInput :: Text -> Prompt (InputT IO) ()
    callInput input = do
      config <- liftIO $ readConfig
      if Config.getUseStreaming config
        then do
          liftIO $ T.putStr "assistant: "
          _ <- callWithText @p input
          liftIO $ T.putStrLn ""
        else do
          result <- callWithText @p input
          showContents [con | con@(Content _ (Message _) _ _) <- result]


runRepl' :: forall p. (PersistentBackend p) => Prompt (InputT IO) ()
runRepl' = do
  cmd <- getUserCommand @p
  runCmd' @p cmd (Just (runRepl' @p))

runRepl :: forall p. (PersistentBackend p) => [ToolProxy] -> [CustomInstructionProxy] -> Text -> Louter.ChatRequest -> Contents -> IO ()
runRepl tools customs sessionName defaultReq contents = do
  runInputT
    ( Settings
        { complete = completeFilename,
          historyFile = Just "intelli-monad.history",
          autoAddHistory = True
        }
    ) $
    do
      output <- getExternalPrint
      let callbackOut = \text -> output (T.unpack text)
          callbackIn = \prompt -> runInputT defaultSettings $
                                  fmap (fmap T.pack) (getInputLine $ T.unpack prompt)
      runPrompt @p tools customs sessionName defaultReq $ do
        prev <- get
        put $ prev { outputCallback = callbackOut
                   , inputCallback = callbackIn
                   }
        push @p contents
        runRepl' @p
