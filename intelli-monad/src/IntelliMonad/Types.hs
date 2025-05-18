{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module IntelliMonad.Types where

import qualified Codec.Picture as P
import Control.Monad.IO.Class
import Control.Monad.Trans.State (StateT)
import Control.Monad.Except (runExceptT, ExceptT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode, Value)
import Data.Map (Map)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as A
import qualified Data.Aeson.KeyMap as A
import qualified Data.Aeson.Text as A
import Data.ByteString (ByteString, fromStrict, toStrict)
import Data.Coerce
import Data.Kind (Type)
import qualified Data.Map as M
import Data.Proxy
import Data.Text (Text)
import Data.Maybe (fromMaybe)
import qualified Data.Text as T
import qualified Data.Text.Lazy as TL
import Data.Time
import qualified Data.Vector as V
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import GHC.Generics
import qualified OpenAI.API as OpenAI
import qualified OpenAI.Types as OpenAI
import qualified IntelliMonad.ExternalApis.Ollama as Ollama
import IntelliMonad.Config
import Network.HTTP.Client (managerResponseTimeout, newManager, responseTimeoutMicro, defaultManagerSettings)
import Network.HTTP.Client.TLS (tlsManagerSettings)
import qualified Data.ByteString as BS
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import Servant.Client (mkClientEnv, parseBaseUrl)
import System.Environment (getEnv, lookupEnv)
import Data.Aeson.Encode.Pretty (encodePretty)
import qualified Servant.Types.SourceT as Servant

data User = User | System | Assistant | Tool deriving (Eq, Show, Ord, Generic)

instance ToJSON User

instance FromJSON User

userToText :: User -> Text
userToText = \case
  User -> "user"
  System -> "system"
  Assistant -> "assistant"
  Tool -> "tool"

textToUser :: Text -> User
textToUser = \case
  "user" -> User
  "system" -> System
  "assistant" -> Assistant
  "tool" -> Tool
  v -> error $ T.unpack $ "Undefined role:" <> v

instance Show (P.Image P.PixelRGB8) where
  show _ = "Image: ..."

data Message
  = Message
      {unText :: Text}
  | Image
      { imageType :: Text,
        imageData :: Text
      }
  | ToolCall
      { toolId :: Text,
        toolName :: Text,
        toolArguments :: Text
      }
  | ToolReturn
      { toolId :: Text,
        toolName :: Text,
        toolContent :: Text
      }
  deriving (Eq, Show, Ord, Generic)

data FinishReason
  = Stop
  | Length
  | ToolCalls
  | FunctionCall
  | ContentFilter
  | Null
  deriving (Eq, Show)

finishReasonToText :: FinishReason -> Text
finishReasonToText = \case
  Stop -> "stop"
  Length -> "length"
  ToolCalls -> "tool_calls"
  FunctionCall -> "function_call"
  ContentFilter -> "content_filter"
  Null -> "null"

textToFinishReason :: Text -> FinishReason
textToFinishReason = \case
  "stop" -> Stop
  "length" -> Length
  "tool_calls" -> ToolCalls
  "function_call" -> FunctionCall
  "content_filter" -> ContentFilter
  "null" -> Null
  _ -> Null

instance ToJSON Message

instance FromJSON Message

newtype Model = Model Text deriving (Eq, Show)

class HasFunctionObject r where
  getFunctionName :: String
  getFunctionDescription :: String
  getFieldDescription :: String -> String

data Schema
  = Maybe' Schema
  | String'
  | Number'
  | Integer'
  | Object' [(String, String, Schema)]
  | Array' Schema
  | Boolean'
  | Null'

class GSchema s f where
  gschema :: forall a. f a -> Schema

class JSONSchema r where
  schema :: Schema
  default schema :: (HasFunctionObject r, Generic r, GSchema r (Rep r)) => Schema
  schema = gschema @r (from (undefined :: r))

data OpenAI
data Ollama

data LLMProtocol
  = OpenAI
  | Ollama

data LLMRequest
  = OpenAIRequest OpenAI.CreateChatCompletionRequest
  | OllamaRequest Ollama.ChatRequest
  deriving (Show, Eq, Ord, Generic)

data LLMResponse
  = OpenAIResponse OpenAI.CreateChatCompletionResponse
  | OllamaResponse Ollama.ChatResponses
  deriving (Show, Eq, Ord, Generic)

data LLMTool
  = OpenAITool OpenAI.ChatCompletionTool
  | OllamaTool Ollama.Tool
  deriving (Show, Eq, Ord, Generic)

instance ToJSON LLMRequest
instance FromJSON LLMRequest
instance ToJSON LLMResponse
instance FromJSON LLMResponse

class LLMApi api where
  type LLMRequest' api
  type LLMResponse' api
  type LLMTool' api
  defaultRequest :: LLMRequest' api
  newTool :: forall a. (HasFunctionObject a, JSONSchema a) => Proxy a -> LLMTool' api
  toTools :: LLMRequest' api -> [LLMTool' api]
  fromTools :: LLMRequest' api -> [LLMTool' api] -> LLMRequest' api
  fromModel_ :: Text -> LLMRequest' api

class ChatCompletion b where
  toRequest :: LLMRequest -> b -> LLMRequest
  fromResponse :: Text -> LLMResponse -> (b, FinishReason)

toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> T.pack err

instance PersistField OpenAI.CreateChatCompletionRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql OpenAI.CreateChatCompletionRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField OpenAI.CreateChatCompletionResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql OpenAI.CreateChatCompletionResponse where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField Ollama.ChatRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Ollama.ChatRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField Ollama.ChatResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Ollama.ChatResponse where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField User where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql User where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField Message where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Message where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField LLMRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistField LLMResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql LLMRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistFieldSql LLMResponse where
  sqlType _ = sqlType (Proxy @ByteString)



share
  [mkPersist sqlSettings, mkMigrate "migrateAll"]
  [persistLowerCase|
Content
    user User
    message Message
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
    deriving Ord
    deriving ToJSON
    deriving FromJSON
    deriving Generic
Context
    request LLMRequest
    response LLMResponse Maybe
    header [Content]
    body [Content]
    footer [Content]
    totalTokens Int
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
    deriving Ord
KeyValue
    namespace Text
    key Text
    value Text
    KeyName namespace key
    deriving Show
    deriving Eq
    deriving Ord
|]

data ToolProxy = forall t. (Tool t, A.FromJSON t, A.ToJSON t, A.FromJSON (Output t), A.ToJSON (Output t), HasFunctionObject t, JSONSchema t) => ToolProxy (Proxy t)

class CustomInstruction a where
  customHeader :: a -> Contents
  customFooter :: a -> Contents

data CustomInstructionProxy = forall t. (CustomInstruction t) => CustomInstructionProxy t

class Hook a where
  preHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m ()
  postHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m ()

data HookProxy = forall t. (Hook t) => HookProxy t

data PersistProxy = forall t. (PersistentBackend t) => PersistProxy t

data PromptEnv = PromptEnv
  { tools :: [ToolProxy]
  -- ^ The list of function calling
  , customInstructions :: [CustomInstructionProxy]
  -- ^ This system sends a prompt that includes headers, bodies and footers. Then the message that LLM outputs is added to bodies. customInstructions generates headers and footers.
  , context :: Context
  -- ^ The request settings like model and prompt logs
  , backend :: PersistProxy
  -- ^ The backend for prompt logging
  , hooks :: [HookProxy]
  -- ^ The hook functions before or after calling LLM
  }

type Contents = [Content]

type Prompt = StateT PromptEnv

-- data TypedPrompt tools task output =

type SessionName = Text

class Tool a where
  data Output a :: Type

  toolFunctionName :: Text
  default toolFunctionName :: (HasFunctionObject a) => Text
  toolFunctionName = T.pack $ getFunctionName @a

  toolExec :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m (Output a)

  toolHeader :: Contents
  toolHeader = []
  toolFooter :: Contents
  toolFooter = []

toAeson :: Schema -> A.Value
toAeson = \case
  Maybe' s -> toAeson s
  String' -> A.Object [("type", "string")]
  Number' -> A.Object [("type", "number")]
  Integer' -> A.Object [("type", "integer")]
  Object' properties ->
    let notMaybes' :: [A.Value]
        notMaybes' =
          concat $
            map
              ( \(name, desc, schema) ->
                  case schema of
                    Maybe' _ -> []
                    _ -> [A.String $ T.pack name]
              )
              properties
     in A.Object
          [ ("type", "object"),
            ( "properties",
              A.Object $
                A.fromList $
                  map
                    ( \(name, desc, schema) ->
                        (A.fromString name, append (toAeson schema) (A.Object [("description", A.String $ T.pack desc)]))
                    )
                    properties
            ),
            ("required", A.Array (V.fromList notMaybes'))
          ]
  Array' s ->
    A.Object
      [ ("type", "array"),
        ("items", toAeson s)
      ]
  Boolean' -> A.Object [("type", "boolean")]
  Null' -> A.Object [("type", "null")]

instance Semigroup Schema where
  (<>) (Object' a) (Object' b) = Object' (a <> b)
  (<>) (Array' a) (Array' b) = Array' (a <> b)
  (<>) _ _ = error "Can not concat json value."

append :: A.Value -> A.Value -> A.Value
append (A.Object a) (A.Object b) = A.Object (a <> b)
append (A.Array a) (A.Array b) = A.Array (a <> b)
append _ _ = error "Can not concat json value."

instance {-# OVERLAPS #-} JSONSchema String where
  schema = String'

instance JSONSchema Text where
  schema = String'

instance (JSONSchema a) => JSONSchema (Maybe a) where
  schema = Maybe' (schema @a)

instance JSONSchema Integer where
  schema = Integer'

instance JSONSchema Int where
  schema = Integer'

instance JSONSchema Double where
  schema = Number'

instance JSONSchema Bool where
  schema = Boolean'

instance (JSONSchema a) => JSONSchema [a] where
  schema = Array' (schema @a)

instance JSONSchema () where
  schema = Null'

instance (HasFunctionObject s, JSONSchema c) => GSchema s U1 where
  gschema _ = Null'

instance (HasFunctionObject s, JSONSchema c) => GSchema s (K1 i c) where
  gschema _ = schema @c

instance (HasFunctionObject s, GSchema s a, GSchema s b) => GSchema s (a :*: b) where
  gschema _ = gschema @s @a undefined <> gschema @s @b undefined

instance (HasFunctionObject s, GSchema s a, GSchema s b) => GSchema s (a :+: b) where
  gschema _ = gschema @s @a undefined
  gschema _ = gschema @s @b undefined

-- | Datatype
instance (HasFunctionObject s, GSchema s f) => GSchema s (M1 D c f) where
  gschema _ = gschema @s @f undefined

-- | Constructor Metadata
instance (HasFunctionObject s, GSchema s f, Constructor c) => GSchema s (M1 C c f) where
  gschema _ = gschema @s @f undefined

-- | Selector Metadata
instance (HasFunctionObject s, GSchema s f, Selector c) => GSchema s (M1 S c f) where
  gschema a =
    let name = selName a
        desc = getFieldDescription @s name
     in Object' [(name, desc, (gschema @s @f undefined))]

toolAdd :: forall api a. (LLMApi api, Tool a, HasFunctionObject a, JSONSchema a) => LLMRequest' api -> LLMRequest' api
toolAdd req =
  let prevTools = case toTools @api req of
        [] -> []
        v -> v
      newTools = prevTools ++ [newTool @api @a Proxy]
   in fromTools @api req newTools

defaultUTCTime :: UTCTime
defaultUTCTime = UTCTime (coerce (0 :: Integer)) 0

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

class PersistentBackend p where
  type Conn p
  config :: p
  setup :: (MonadIO m, MonadFail m) => p -> m (Maybe (Conn p))
  initialize :: (MonadIO m, MonadFail m) => Conn p -> Context -> m ()
  load :: (MonadIO m, MonadFail m) => Conn p -> SessionName -> m (Maybe Context)
  loadByKey :: (MonadIO m, MonadFail m) => Conn p -> (Key Context) -> m (Maybe Context)
  save :: (MonadIO m, MonadFail m) => Conn p -> Context -> m (Maybe (Key Context))
  saveContents :: (MonadIO m, MonadFail m) => Conn p -> [Content] -> m ()
  listSessions :: (MonadIO m, MonadFail m) => Conn p -> m [Text]
  deleteSession :: (MonadIO m, MonadFail m) => Conn p -> SessionName -> m ()
  listKeys :: (MonadIO m, MonadFail m) => Conn p -> m [Unique KeyValue]
  getKey :: (MonadIO m, MonadFail m) => Conn p -> Unique KeyValue -> m (Maybe Text)
  setKey :: (MonadIO m, MonadFail m) => Conn p -> Unique KeyValue -> Text -> m ()
  deleteKey :: (MonadIO m, MonadFail m) => Conn p -> Unique KeyValue -> m ()


instance LLMApi OpenAI where
  type LLMRequest' OpenAI = OpenAI.CreateChatCompletionRequest
  type LLMResponse' OpenAI = OpenAI.CreateChatCompletionResponse
  type LLMTool' OpenAI = OpenAI.ChatCompletionTool
  defaultRequest =
    OpenAI.CreateChatCompletionRequest
      { OpenAI.createChatCompletionRequestMessages = [],
        OpenAI.createChatCompletionRequestModel = OpenAI.CreateChatCompletionRequestModel "gpt-4",
        OpenAI.createChatCompletionRequestFrequencyUnderscorepenalty = Nothing,
        OpenAI.createChatCompletionRequestLogitUnderscorebias = Nothing,
        OpenAI.createChatCompletionRequestLogprobs = Nothing,
        OpenAI.createChatCompletionRequestTopUnderscorelogprobs = Nothing,
        OpenAI.createChatCompletionRequestMaxUnderscoretokens = Nothing,
        OpenAI.createChatCompletionRequestN = Nothing,
        OpenAI.createChatCompletionRequestPresenceUnderscorepenalty = Nothing,
        OpenAI.createChatCompletionRequestResponseUnderscoreformat = Nothing,
        OpenAI.createChatCompletionRequestSeed = Just 0,
        OpenAI.createChatCompletionRequestStop = Nothing,
        OpenAI.createChatCompletionRequestStream = Nothing,
        OpenAI.createChatCompletionRequestTemperature = Nothing,
        OpenAI.createChatCompletionRequestTopUnderscorep = Nothing,
        OpenAI.createChatCompletionRequestTools = Nothing,
        OpenAI.createChatCompletionRequestToolUnderscorechoice = Nothing,
        OpenAI.createChatCompletionRequestUser = Nothing,
        OpenAI.createChatCompletionRequestFunctionUnderscorecall = Nothing,
        OpenAI.createChatCompletionRequestFunctions = Nothing
      }
  newTool (Proxy :: Proxy a) =
    OpenAI.ChatCompletionTool
      { chatCompletionToolType = "function",
        chatCompletionToolFunction =
          OpenAI.FunctionObject
            { functionObjectDescription = Just (T.pack $ getFunctionDescription @a),
              functionObjectName = T.pack $ getFunctionName @a,
              functionObjectParameters = Just $
                case toAeson (schema @a) of
                  A.Object kv -> M.fromList $ map (\(k, v) -> (A.toString k, v)) $ A.toList kv
                  _ -> []
            }
      }
  toTools req =
    case OpenAI.createChatCompletionRequestTools req of
      Nothing -> []
      Just v -> v
  fromTools req [] = req {OpenAI.createChatCompletionRequestTools = Nothing}
  fromTools req tools = req {OpenAI.createChatCompletionRequestTools = Just tools}
  fromModel_ model =
    (defaultRequest @OpenAI :: LLMRequest' OpenAI) 
      { OpenAI.createChatCompletionRequestModel = OpenAI.CreateChatCompletionRequestModel model
      }

-- | Read the JSON object and convert it to a Map
toMap :: Text -> Map Text Value
toMap json =
  case A.decodeStrictText json of
    Just v -> v
    Nothing -> error $ T.unpack $ "Decoding JSON fails"
  

fromMap :: Map Text Value -> Text
fromMap txt = TL.toStrict $ A.encodeToLazyText txt

instance ChatCompletion Contents where
  toRequest (OpenAIRequest orgRequest) contents =
    let messages = flip map contents $ \case
          Content user (Message message) _ _ ->
            OpenAI.ChatCompletionRequestMessage
              { OpenAI.chatCompletionRequestMessageContent = Just $ OpenAI.ChatCompletionRequestMessageContentText message,
                OpenAI.chatCompletionRequestMessageRole = userToText user,
                OpenAI.chatCompletionRequestMessageName = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                OpenAI.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content user (Image type' img) _ _ ->
            OpenAI.ChatCompletionRequestMessage
              { OpenAI.chatCompletionRequestMessageContent =
                  Just $
                    OpenAI.ChatCompletionRequestMessageContentParts
                      [ OpenAI.ChatCompletionRequestMessageContentPart
                          { OpenAI.chatCompletionRequestMessageContentPartType = "image_url",
                            OpenAI.chatCompletionRequestMessageContentPartText = Nothing,
                            OpenAI.chatCompletionRequestMessageContentPartImageUnderscoreurl =
                              Just $
                                OpenAI.ChatCompletionRequestMessageContentPartImageImageUrl
                                  { OpenAI.chatCompletionRequestMessageContentPartImageImageUrlUrl =
                                      "data:image/" <> type' <> ";base64," <> img,
                                    OpenAI.chatCompletionRequestMessageContentPartImageImageUrlDetail = Nothing
                                  }
                          }
                      ],
                OpenAI.chatCompletionRequestMessageRole = userToText user,
                OpenAI.chatCompletionRequestMessageName = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                OpenAI.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Nothing
              }
          Content user (ToolCall id' name' args') _ _ ->
            OpenAI.ChatCompletionRequestMessage
              { OpenAI.chatCompletionRequestMessageContent = Nothing,
                OpenAI.chatCompletionRequestMessageRole = userToText user,
                OpenAI.chatCompletionRequestMessageName = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecalls =
                  Just
                    [ OpenAI.ChatCompletionMessageToolCall
                        { OpenAI.chatCompletionMessageToolCallId = id',
                          OpenAI.chatCompletionMessageToolCallType = "function",
                          OpenAI.chatCompletionMessageToolCallFunction =
                            OpenAI.ChatCompletionMessageToolCallFunction
                              { OpenAI.chatCompletionMessageToolCallFunctionName = name',
                                OpenAI.chatCompletionMessageToolCallFunctionArguments = args'
                              }
                        }
                    ],
                OpenAI.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
          Content user (ToolReturn id' name' ret') _ _ ->
            OpenAI.ChatCompletionRequestMessage
              { OpenAI.chatCompletionRequestMessageContent = Just $ OpenAI.ChatCompletionRequestMessageContentText ret',
                OpenAI.chatCompletionRequestMessageRole = userToText user,
                OpenAI.chatCompletionRequestMessageName = Just name',
                OpenAI.chatCompletionRequestMessageToolUnderscorecalls = Nothing,
                OpenAI.chatCompletionRequestMessageFunctionUnderscorecall = Nothing,
                OpenAI.chatCompletionRequestMessageToolUnderscorecallUnderscoreid = Just id'
              }
     in OpenAIRequest $ orgRequest {OpenAI.createChatCompletionRequestMessages = messages}
  toRequest (OllamaRequest orgRequest) contents =
    let messages = flip map contents $ \case
          Content user (Message message) _ _ ->
            Ollama.ChatMessage
              { Ollama.msgContent = message,
                Ollama.msgRole = userToText user,
                Ollama.msgImages = Nothing,
                Ollama.msgName = Nothing,
                Ollama.msgToolCalls = Nothing
              }
          Content user (Image type' img) _ _ ->
            Ollama.ChatMessage
              { Ollama.msgContent = "",
                Ollama.msgRole = userToText user,
                Ollama.msgImages = Just [img],
                Ollama.msgName = Nothing,
                Ollama.msgToolCalls = Nothing
              }
          Content user (ToolCall id' name' args') _ _ -> 
            Ollama.ChatMessage
              { Ollama.msgContent = "",
                Ollama.msgRole = userToText Assistant,
                Ollama.msgImages = Nothing,
                Ollama.msgName = Nothing,
                Ollama.msgToolCalls = Just [Ollama.ToolCall (Ollama.ToolFunction
                                            { Ollama.tfName = name'
                                            , Ollama.tfArguments = toMap args'
                                            })]
              }
          Content user (ToolReturn id' name' ret') _ _ ->
            Ollama.ChatMessage
              { Ollama.msgContent = ret',
                Ollama.msgRole = userToText Tool,
                Ollama.msgName = Just name',
                Ollama.msgImages = Nothing,
                Ollama.msgToolCalls = Nothing
              }
     in OllamaRequest $ orgRequest {Ollama.crMessages = messages}

  fromResponse sessionName (OpenAIResponse response) =
    let res = head (OpenAI.createChatCompletionResponseChoices response)
        message = OpenAI.createChatCompletionResponseChoicesInnerMessage res
        role = textToUser $ OpenAI.chatCompletionResponseMessageRole message
        content = OpenAI.chatCompletionResponseMessageContent message
        finishReason = textToFinishReason $ OpenAI.createChatCompletionResponseChoicesInnerFinishUnderscorereason res
        v = case OpenAI.chatCompletionResponseMessageToolUnderscorecalls message of
          Just toolcalls -> map (\(OpenAI.ChatCompletionMessageToolCall id' _ (OpenAI.ChatCompletionMessageToolCallFunction name' args')) -> Content role (ToolCall id' name' args') sessionName defaultUTCTime) toolcalls
          Nothing -> [Content role (Message (fromMaybe "" content)) sessionName defaultUTCTime]
     in (v, finishReason)
  fromResponse sessionName (OllamaResponse (response:_)) =
    let message = Ollama.chatRespMessage response
        role = textToUser $ Ollama.msgRole message
        content = Ollama.msgContent message
        finishReason = Ollama.msgContent message
    in case Ollama.msgToolCalls message of
         Just toolcalls -> (map (\func -> Content role (ToolCall "" func.tcFunction.tfName (fromMap func.tcFunction.tfArguments)) sessionName defaultUTCTime) toolcalls, ToolCalls)
         Nothing -> ([Content role (Message (fromMaybe "" (Just content))) sessionName defaultUTCTime], Stop)

instance LLMApi Ollama where
  type LLMRequest' Ollama = Ollama.ChatRequest
  type LLMResponse' Ollama = Ollama.ChatResponses
  type LLMTool' Ollama = Ollama.Tool
  defaultRequest = Ollama.ChatRequest
    { Ollama.crModel = "gemma3"
    , Ollama.crMessages = []
    , Ollama.crTools = Nothing
    , Ollama.crFormat = Nothing
    , Ollama.crOptions = Nothing
    , Ollama.crStream = Just False
    , Ollama.crKeepAlive = Just "5m"
    }
  newTool (Proxy :: Proxy a) =
    Ollama.Tool
      { Ollama.toolType = "function",
        Ollama.toolFunction =
          Ollama.ToolFunctionDefinition
            { Ollama.tfdDescription = Just (T.pack $ getFunctionDescription @a),
              Ollama.tfdName = T.pack $ getFunctionName @a,
              Ollama.tfdParameters = toAeson (schema @a)
            }
      }
  toTools req =
    case Ollama.crTools req of
      Nothing -> []
      Just v -> v
  fromTools req [] = req {Ollama.crTools = Nothing}
  fromTools req tools = req {Ollama.crTools = Just tools}
  fromModel_ model =
    (defaultRequest @Ollama :: LLMRequest' Ollama) 
      { Ollama.crModel = model
      }
    
data LLMProxy api =
  LLMProxy
  { toRequest' :: LLMRequest' api -> LLMRequest
  , toResponse' :: LLMResponse' api -> LLMResponse
  }

withLLMRequest
  :: forall a. LLMRequest
  -> (forall (api :: Type)
      . (LLMApi api)
      => LLMProxy api
      -> LLMRequest' api
      -> a)
  -> a
withLLMRequest req func =
  case req of
    OpenAIRequest req' -> func (LLMProxy @OpenAI OpenAIRequest OpenAIResponse)  req'
    OllamaRequest req' -> func (LLMProxy @Ollama OllamaRequest OllamaResponse)  req'

updateRequest :: LLMRequest -> Contents -> LLMRequest
updateRequest = toRequest

--  withLLMRequest req' $ \(p :: LLMProxy api) req -> (toRequest' @api p) (toRequest req cs)

addTools :: [ToolProxy] -> LLMRequest -> LLMRequest
addTools [] req' = req'
addTools (tool : tools') req' =
  case tool of
    (ToolProxy (_ :: Proxy a)) ->
      withLLMRequest req' $ \(p :: LLMProxy api) req -> addTools tools' ((toRequest' @api p) $ toolAdd @api @a req)
      
fromModel :: Text -> LLMRequest
fromModel = OpenAIRequest . fromModel_ @OpenAI

runRequest :: forall a. (ChatCompletion a) => Text -> LLMRequest -> a -> IO ((a, FinishReason), LLMResponse)
runRequest sessionName (OpenAIRequest defaultReq) request = do
  config <- readConfig
  let api_key = OpenAI.clientAuth config.apiKey
  url <- case parseBaseUrl (T.unpack config.endpoint) of
           Just url' -> pure url'
           Nothing -> error $ T.unpack $ "Can not parse the endpoint: " <> config.endpoint
  manager <-
    newManager
      ( tlsManagerSettings
          { managerResponseTimeout = responseTimeoutMicro (120 * 1000 * 1000)
          }
      )
  let OpenAI.OpenAIBackend {..} = OpenAI.createOpenAIClient
      (OpenAIRequest req) = (toRequest (OpenAIRequest defaultReq) request)

  lookupEnv "OPENAI_DEBUG" >>= \case
    Just "1" -> do
      liftIO $ do
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""
    _ -> return ()
  res <- OpenAI.callOpenAI (mkClientEnv manager url) $ createChatCompletion api_key req
  return $ (fromResponse sessionName (OpenAIResponse res), OpenAIResponse res)

runRequest sessionName (OllamaRequest defaultReq) request = do
  config <- readConfig
  url <- case parseBaseUrl (T.unpack config.endpoint) of
           Just url' -> pure url'
           Nothing -> error $ T.unpack $ "Can not parse the endpoint: " <> config.endpoint
  liftIO $ print "hello"
  manager <-
    newManager
      ( defaultManagerSettings
          { managerResponseTimeout = responseTimeoutMicro (120 * 1000 * 1000)
          }
      )
  let Ollama.OllamaBackend {..} = Ollama.createOllamaClient
      (OllamaRequest req) = (toRequest (OllamaRequest defaultReq) request)

  lookupEnv "OPENAI_DEBUG" >>= \case
    Just "1" -> do
      liftIO $ do
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""
    _ -> return ()
  (res :: [Ollama.ChatResponse]) <- Ollama.callOllama (mkClientEnv manager url) $ do
    (r0 :: Servant.SourceT IO Ollama.ChatResponse) <- chatClient (Ollama.clientAuth "") (req {Ollama.crStream = Just False})
    liftIO $ do
      runExceptT (Servant.runSourceT r0) >>= \case
        Left err -> do
          print "-------"
          print err
          return []
--          fail err
        Right v -> return v
    
  return $ (fromResponse sessionName (OllamaResponse res), OllamaResponse res)
