{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes  #-}
{-# LANGUAGE DeriveGeneric  #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE FlexibleInstances  #-}


module Prompt where

import qualified OpenAI.API as API
import qualified OpenAI.Types as API
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Encoding as T
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString as BS
import qualified Codec.Picture as P
import           Codec.Picture.Png (encodePng)


import           Control.Monad.Trans.State (StateT, get, put, runStateT)
import           Control.Monad.Trans.Class (lift, MonadTrans)
import           Control.Monad.IO.Class

import           Network.HTTP.Client     (newManager, managerResponseTimeout, responseTimeoutMicro)
import           Network.HTTP.Client.TLS (tlsManagerSettings)
import           Servant.Client          (mkClientEnv, parseBaseUrl)
import           System.Environment      (getEnv, lookupEnv)
import           System.Console.Haskeline
import           Control.Monad (forM_, forM)
import           Data.Aeson (encode)
import qualified Data.Aeson as A
import           System.Process
import           GHC.Generics
import           GHC.IO.Exception
import           Data.Maybe (fromMaybe, catMaybes)
import qualified Data.ByteString.Base64 as Base64


data User = User | System | Assistant | Tool deriving (Eq, Show)

userToText :: User -> T.Text
userToText = \case
  User -> "user"
  System -> "system"
  Assistant -> "assistant"
  Tool -> "tool"
  
textToUser :: T.Text -> User
textToUser = \case
  "user" -> User
  "system" -> System
  "assistant" -> Assistant
  "tool" -> Tool
  v -> error $ T.unpack $ "Undefined role:" <> v

instance Show (P.Image P.PixelRGB8) where
  show _ = "Image: ..."
  
data Message
  = Text { unText :: T.Text }
  | Image { unImage :: P.Image P.PixelRGB8 }
  | ToolCall { toolId :: T.Text
             , toolName :: T.Text
             , toolArguments :: T.Text
             }
  | ToolReturn { toolId :: T.Text
               , toolName :: T.Text
               , toolContent :: T.Text
               }
  deriving (Eq, Show)

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
  , tokens :: [Int]
  , total_tokens :: Int
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
  let current_total_tokens = fromMaybe 0 $ API.completionUsageTotalUnderscoretokens <$> API.createChatCompletionResponseUsage res
      next = prev { response = Just res, tokens = prev.tokens ++ [current_total_tokens - total_tokens prev] , total_tokens = current_total_tokens}
  setContext next
  push contents
  if hasToolCall contents
    then do
      showContents contents
      retTool <- tryToolExec contents
      showContents retTool
      push retTool
      call
    else return contents
  

hasToolCall :: Contents -> Bool
hasToolCall (Contents cs) =
  let loop [] = False
      loop (m@(Content (user, (ToolCall _ _ _))) : cs) = True
      loop (_ : cs) = loop cs
  in loop cs

filterToolCall :: Contents -> Contents
filterToolCall (Contents cs) =
  let loop [] = []
      loop (m@(Content (user, (ToolCall _ _ _))) : cs) = m: loop cs
      loop (_ : cs) = loop cs
  in Contents $ loop cs

tryToolExec :: (MonadIO m, MonadFail m) => Contents -> Prompt m Contents
tryToolExec contents = do
  cs <- forM (unContents (filterToolCall contents)) $ \c@(Content (user, (ToolCall id' name' args'))) ->
    if name' == toolFunctionName @BashInput
      then do
        case (A.eitherDecode (BS.fromStrict (T.encodeUtf8 args')) :: Either String BashInput) of
          Left err -> return Nothing
          Right input -> do
            output <- liftIO $ toolExec input
            return $ Just $ (Content (Tool, ToolReturn id' name' (T.decodeUtf8Lenient (BS.toStrict (encode output)))))
      else
        return Nothing
  return $ Contents $ catMaybes cs
    
runPrompt :: (MonadIO m, MonadFail m) => API.CreateChatCompletionRequest -> Prompt m a -> m a
runPrompt req func = do
  let context = Context req Nothing mempty [] 0
  fst <$> runStateT func context

class (A.ToJSON a, A.FromJSON a, A.ToJSON b, A.FromJSON b) => Tool a b | a -> b where
  toolFunctionName :: T.Text
  toolSchema :: API.ChatCompletionTool
  toolExec :: a -> IO b
  toolAdd :: API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest 
  toolAdd req =
    let prevTools = case API.createChatCompletionRequestTools req of
          Nothing -> []
          Just v -> v
        newTools = prevTools ++ [toolSchema @a @b]
    in req { API.createChatCompletionRequestTools = Just newTools } 

data BashInput = BashInput
  { script :: String
  } deriving (Eq, Show, Generic)

data BashOutput = BashOutput
  { code :: Int
  , stdout :: String
  , stderr :: String
  } deriving (Eq, Show, Generic)

instance A.FromJSON BashInput
instance A.ToJSON BashInput
instance A.FromJSON BashOutput
instance A.ToJSON BashOutput

instance Tool BashInput BashOutput where
  toolFunctionName = "call_bash_script"
  toolSchema = API.ChatCompletionTool
    { chatCompletionToolType = "function"
    , chatCompletionToolFunction = API.FunctionObject
      { functionObjectDescription = Just "Call a bash script in a local environment"
      , functionObjectName = "call_bash_script"
      , functionObjectParameters = Just $
        [ ("type", "object")
        , ("properties", A.Object [
              ("script", A.Object [
                  ("type", "string"),
                  ("description", "A script executing in a local environment ")
                  ])
              ])
        , ("required", A.Array ["script"])
        ]
      }
    }
  toolExec args = do
    (code, stdout, stderr) <- readCreateProcessWithExitCode (shell args.script) ""
    let code' = case code of
          ExitSuccess -> 0
          ExitFailure v -> v
    return $ BashOutput code' stdout stderr

instance ChatCompletion Contents where
  toRequest orgRequest (Contents contents) =
    let messages = flip map contents $ \case
          Content (user, Text message) -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Just $ API.ChatCompletionRequestMessageContentText message
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecalls = Nothing
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content (user, Image img) -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Just $
                API.ChatCompletionRequestMessageContentParts [
                  API.ChatCompletionRequestMessageContentPart
                    { API.chatCompletionRequestMessageContentPartType = "image_url"
                    , API.chatCompletionRequestMessageContentPartImageUnderscoreurl =
                      Just $ API.ChatCompletionRequestMessageContentPartImageImageUrl
                              { API.chatCompletionRequestMessageContentPartImageImageUrlUrl =
                                  let b64 = (Base64.encode (BS.toStrict (encodePng img)) :: BS.ByteString)
                                  in "data:image/png;base64," <>  (T.decodeUtf8 b64)
                              , API.chatCompletionRequestMessageContentPartImageImageUrlDetail = Nothing  
                              }
                    }
                  ]
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecalls = Nothing
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content (user, ToolCall id' name' args') -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Nothing
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecalls = Just [
                  API.ChatCompletionMessageToolCall
                    { API.chatCompletionMessageToolCallId = id'
                    , API.chatCompletionMessageToolCallType = "function"
                    , API.chatCompletionMessageToolCallFunction =
                        API.ChatCompletionMessageToolCallFunction
                          { API.chatCompletionMessageToolCallFunctionName = name'
                          , API.chatCompletionMessageToolCallFunctionArguments = args'
                          }
                    }
                  ]
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
          Content (user, ToolReturn id' name' ret') -> 
            API.ChatCompletionRequestMessage
              { API.chatCompletionRequestMessageContent = Just $ API.ChatCompletionRequestMessageContentText ret'
              , API.chatCompletionRequestMessageRole = userToText user
              , API.chatCompletionRequestMessageName = Just name'
              , API.chatCompletionRequestMessageToolUnderscorecalls = Nothing
              , API.chatCompletionRequestMessageFunctionUnderscorecall = Nothing
              , API.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
    in orgRequest { API.createChatCompletionRequestMessages = messages }
  fromResponse response = Contents $
    concat $ flip map (API.createChatCompletionResponseChoices response) $ \res -> 
      let message = API.createChatCompletionResponseChoicesInnerMessage res
          role = textToUser $ API.chatCompletionResponseMessageRole message
          content = API.chatCompletionResponseMessageContent message
      in case API.chatCompletionResponseMessageToolUnderscorecalls message of
         Just toolcalls -> map (\(API.ChatCompletionMessageToolCall id' _ (API.ChatCompletionMessageToolCallFunction name' args')) -> Content (role, ToolCall id' name' args')) toolcalls
         
         Nothing -> [Content (role, Text (fromMaybe "" content))]
      
      
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

showContents :: MonadIO m => Contents -> m ()
showContents (Contents res) = do
  forM_ res $ \(Content (user, message)) ->
    liftIO $ T.putStrLn $ userToText user <> ": " <>
               case message of
                 Text t -> t
                 Image _ -> "Image: ..."
                 c@(ToolCall _ _ _) -> T.pack $ show c
                 c@(ToolReturn _ _ _) -> T.pack $ show c
    
  
runRepl :: API.CreateChatCompletionRequest -> Contents -> IO ()
runRepl defaultReq contents = do
  let settings = toolAdd @BashInput defaultReq
  runInputT defaultSettings (runPrompt settings (push contents >> loop))
  where
    loop :: Prompt (InputT IO) ()
    loop = do
      minput <- getTextInputLine "% "
      case minput of
        Nothing -> return ()
        Just "quit" -> return ()
        Just "clear" -> do
          loop
        Just "show-usage" -> do
          context <- getContext
          liftIO $ do
            print context.total_tokens
          loop
        Just input -> do
          let contents = Contents [Content (User, Text input)]
          push contents
          ret <- call
          showContents ret
          loop
