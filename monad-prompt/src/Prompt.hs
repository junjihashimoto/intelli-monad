{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}

module Prompt where

import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.ByteString.Lazy as BS
import qualified Codec.Picture as P

import           Control.Monad.Trans.State (StateT, get, put, runStateT)
import           Control.Monad.Trans.Class (lift, MonadTrans)
import           Control.Monad.IO.Class

import           Network.HTTP.Client     (newManager, managerResponseTimeout, responseTimeoutMicro)
import           Network.HTTP.Client.TLS (tlsManagerSettings)
import           Servant.Client          (ClientEnv, mkClientEnv, parseBaseUrl, responseBody)
import           System.Environment      (getEnv, lookupEnv)
import           System.Console.Haskeline
import           Control.Monad (forM_, forM)
import           Control.Monad.IO.Class (liftIO)
import           Data.Aeson (encode)
import           System.Process

data User = User | System | Assistant deriving (Eq, Show)

userToText :: User -> T.Text
userToText = \case
  User -> "user"
  System -> "system"
  Assistant -> "assistant"
  
textToUser :: T.Text -> User
textToUser = \case
  "user" -> User
  "system" -> System
  "assistant" -> Assistant
  
data Message = Text { unText :: T.Text } | Image { unImage :: P.Image P.PixelRGB8 } deriving (Eq)

instance Show Message where
  show (Text t) = show $ t
  show (Image _) = show $ "Imaage: ..."

newtype Model = Model T.Text deriving (Eq, Show)

newtype Content = Content { unContent :: (User, Message) } deriving (Eq, Show)

newtype Contents = Contents { unContents :: [Content] } deriving (Eq, Show, Semigroup, Monoid)

class ChatCompletion a where
  toRequest :: API.CreateChatCompletionRequest -> a -> API.CreateChatCompletionRequest
  fromResponse :: API.CreateChatCompletionResponse -> a

class ChatCompletion a => Validate a b where
  tryConvert :: a -> Either a b

data Context = Context
  { request :: API.CreateChatCompletionRequest
  , response :: Maybe API.CreateChatCompletionResponse
  , contents :: Contents
  } deriving (Eq, Show)

type Prompt = StateT Context

getContext :: (MonadIO m, MonadFail m) => Prompt m Context
getContext = get
  
setContext :: (MonadIO m, MonadFail m) => Context -> Prompt m ()
setContext = put
  
-- last :: (MonadIO m, MonadFail m) => Prompt m (Maybe Contents)
-- last = do
--   prev <- getContext
--   let contents' = unContents prev.contents
--   if length contents' == 0
--     then return Nothing
--     else return $ Just $ head contents'

push :: (MonadIO m, MonadFail m) => Contents -> Prompt m ()
push contents = do
  prev <- getContext
  let nextContents = prev.contents <> contents
      next = prev { contents = nextContents, request = toRequest prev.request nextContents }
  setContext next

call :: (MonadIO m, MonadFail m) => Prompt m Contents
call = do
  prev <- getContext
  (contents, res) <- liftIO $ runRequest prev.request prev.contents
  let next = prev { response = Just res }
  setContext next
  push contents
  return contents

-- instance (TypedChatCompletion i0 o0, TypedChatCompletion o0 o1) => TypedChatCompletion i0 o1

runPrompt :: (MonadIO m, MonadFail m) => API.CreateChatCompletionRequest -> Prompt m a -> m a
runPrompt req func = do
  let context = Context req Nothing mempty 
  fst <$> runStateT func context
  

instance ChatCompletion Contents where
  toRequest orgRequest (Contents contents) =
    let messages = flip map contents $ \case
          Content (user, Text message) -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = API.ChatCompletionRequestMessageContentText message
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecalls = Nothing
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content (user, Image message) -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = undefined
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecalls = Nothing
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
    in orgRequest { API.createChatCompletionRequestMessages = messages }
  fromResponse response = Contents $
    flip map (API.createChatCompletionResponseChoices response) $ \res -> 
      let message = API.createChatCompletionResponseChoicesInnerMessage res
          role = textToUser $ API.chatCompletionResponseMessageRole message
          content = API.chatCompletionResponseMessageContent message
      in Content (role, Text content)
      
      
defaultRequest :: API.CreateChatCompletionRequest
defaultRequest =
  API.CreateChatCompletionRequest
                    { API.createChatCompletionRequestMessages = []
                    , API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel "gpt-3.5-turbo"
                    , API.createChatCompletionRequestFrequencyUnderscorepenalty = Nothing
                    , API.createChatCompletionRequestLogitUnderscorebias = Nothing
                    , API.createChatCompletionRequestLogprobs = Nothing
                    , API.createChatCompletionRequestTopUnderscorelogprobs  = Nothing
                    , API.createChatCompletionRequestMaxUnderscoretokens  = Nothing
                    , API.createChatCompletionRequestN  = Nothing
                    , API.createChatCompletionRequestPresenceUnderscorepenalty  = Nothing
                    , API.createChatCompletionRequestResponseUnderscoreformat  = Nothing
                    , API.createChatCompletionRequestSeed = Just 0
                    , API.createChatCompletionRequestStop = Nothing
                    , API.createChatCompletionRequestStream = Nothing
                    , API.createChatCompletionRequestTemperature = Nothing
                    , API.createChatCompletionRequestTopUnderscorep = Nothing
                    , API.createChatCompletionRequestTools = Nothing
                    , API.createChatCompletionRequestToolUnderscorechoice = Nothing
                    , API.createChatCompletionRequestUser = Nothing
                    , API.createChatCompletionRequestFunctionUnderscorecall = Nothing
                    , API.createChatCompletionRequestFunctions = Nothing
                    }

runRequest :: ChatCompletion a => API.CreateChatCompletionRequest -> a -> IO (a, API.CreateChatCompletionResponse)
runRequest defaultReq request = do
  api_key <- (API.clientAuth . T.pack) <$> getEnv "OPENAI_API_KEY"
  url <- do
    lookupEnv "OPENAI_ENDPOINT" >>= \case
      Just url -> parseBaseUrl url
      Nothing -> parseBaseUrl "https://api.openai.com/v1/"
  manager <- newManager (
    tlsManagerSettings {
        managerResponseTimeout = responseTimeoutMicro (120 * 1000 * 1000) 
    })
  let API.OpenAIBackend{..} = API.createOpenAIClient
      req = (toRequest defaultReq request)
  -- print $ encode req
  res <- API.callOpenAI (mkClientEnv manager url) $ createChatCompletion api_key req
  return (fromResponse res, res)

getTextInputLine :: MonadTrans t => String -> t (InputT IO) (Maybe T.Text)
getTextInputLine prompt = fmap (fmap T.pack) (lift $ getInputLine prompt)

shAgent :: Contents -> IO Contents
shAgent inputs = do
  let spec = Contents [
          Content (System, Text "You are a agent to run bash command of your tasks. Input messages are tasks. You should generate bash scripts to resolve the tasks.")
        ]
  contents <- runPrompt defaultRequest (push (spec <> inputs) >> call)
  outs <- forM (unContents contents) $ \(Content (user, message))-> do
    v <- readCreateProcess (shell (T.unpack $ unText message)) ""
    return $ Content (System, Text (T.pack v))
  return $ Contents outs

shAgent' :: Contents -> IO Contents
shAgent' inputs = do
  let spec = Contents [
          Content (System, Text "You can run a script with a message '%script-type <script>', e.g. '%bash ls' or '%python print(1+3), then it returns a system prompt as the result.")
        ]
  contents <- runPrompt defaultRequest (push (spec <> inputs) >> call)
  outs <- forM (unContents contents) $ \(Content (user, message))-> do
    v <- readCreateProcess (shell (T.unpack $ unText message)) ""
    return $ Content (System, Text (T.pack v))
  return $ Contents outs

showContents :: MonadIO m => Contents -> m ()
showContents (Contents res) = do
  forM_ res $ \(Content (user, message)) ->
    liftIO $ T.putStrLn $ userToText user <> ": " <> unText message
  
runRepl :: API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl defaultReq contents = do
  runInputT defaultSettings (runPrompt defaultReq (push contents >> loop))
  where
    loop :: Prompt (InputT IO) ()
    loop = do
      minput <- getTextInputLine "% "
      case minput of
        Nothing -> return ()
        Just "quit" -> return ()
        Just "show-usage" -> do
          context <- getContext
          case (encode <$> (API.createChatCompletionResponseUsage <$> context.response)) of
            Just v -> liftIO $ do
              BS.putStr v
              BS.putStr "\n"
            Nothing -> return ()
          loop
        Just input -> do
          let contents = Contents [Content (User, Text input)]
          push contents
          call >>= showContents
          loop
