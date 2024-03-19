{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
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
{-# OPTIONS_GHC -fno-warn-orphans #-}

module IntelliMonad.Types where

import qualified Codec.Picture as P
import Control.Monad.Trans.State (StateT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as A
import qualified Data.Aeson.KeyMap as A
import Data.ByteString (ByteString, fromStrict, toStrict)
import Data.Coerce
import Data.Kind (Type)
import qualified Data.Map as M
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import Data.Time
import qualified Data.Vector as V
import Database.Persist
import Database.Persist.Sqlite
import Database.Persist.TH
import GHC.Generics
import qualified OpenAI.Types as API

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
  ContentFilter -> "content_fileter"
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

class ChatCompletion a where
  toRequest :: API.CreateChatCompletionRequest -> a -> API.CreateChatCompletionRequest
  fromResponse :: Text -> API.CreateChatCompletionResponse -> (a, FinishReason)

class (ChatCompletion a) => Validate a b where
  tryConvert :: a -> Either a b

toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> T.pack err

instance PersistField API.CreateChatCompletionRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql API.CreateChatCompletionRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField API.CreateChatCompletionResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql API.CreateChatCompletionResponse where
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
    request API.CreateChatCompletionRequest
    response API.CreateChatCompletionResponse Maybe
    header [Content]
    body [Content]
    footer [Content]
    totalTokens Int
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
    deriving Ord
|]

data ToolProxy = forall t. (Tool t, A.FromJSON t, A.ToJSON t, A.FromJSON (Output t), A.ToJSON (Output t)) => ToolProxy (Proxy t)

class CustomInstruction a where
  customHeader :: Contents
  customFooter :: Contents

data CustomInstructionProxy = forall t. (CustomInstruction t) => CustomInstructionProxy (Proxy t)

data PromptEnv = PromptEnv
  { tools :: [ToolProxy],
    customInstructions :: [CustomInstructionProxy],
    context :: Context
  }

type Contents = [Content]

type Prompt = StateT PromptEnv

-- data TypedPrompt tools task output =

type SessionName = Text

defaultRequest :: API.CreateChatCompletionRequest
defaultRequest =
  API.CreateChatCompletionRequest
    { API.createChatCompletionRequestMessages = [],
      API.createChatCompletionRequestModel = API.CreateChatCompletionRequestModel "gpt-4",
      API.createChatCompletionRequestFrequencyUnderscorepenalty = Nothing,
      API.createChatCompletionRequestLogitUnderscorebias = Nothing,
      API.createChatCompletionRequestLogprobs = Nothing,
      API.createChatCompletionRequestTopUnderscorelogprobs = Nothing,
      API.createChatCompletionRequestMaxUnderscoretokens = Nothing,
      API.createChatCompletionRequestN = Nothing,
      API.createChatCompletionRequestPresenceUnderscorepenalty = Nothing,
      API.createChatCompletionRequestResponseUnderscoreformat = Nothing,
      API.createChatCompletionRequestSeed = Just 0,
      API.createChatCompletionRequestStop = Nothing,
      API.createChatCompletionRequestStream = Nothing,
      API.createChatCompletionRequestTemperature = Nothing,
      API.createChatCompletionRequestTopUnderscorep = Nothing,
      API.createChatCompletionRequestTools = Nothing,
      API.createChatCompletionRequestToolUnderscorechoice = Nothing,
      API.createChatCompletionRequestUser = Nothing,
      API.createChatCompletionRequestFunctionUnderscorecall = Nothing,
      API.createChatCompletionRequestFunctions = Nothing
    }

class Tool a where
  data Output a :: Type

  toolFunctionName :: Text
  default toolFunctionName :: (HasFunctionObject a) => Text
  toolFunctionName = T.pack $ getFunctionName @a

  toolSchema :: API.ChatCompletionTool
  default toolSchema :: (HasFunctionObject a, JSONSchema a, Generic a, GSchema a (Rep a)) => API.ChatCompletionTool
  toolSchema = toChatCompletionTool @a

  toolExec :: a -> IO (Output a)

toChatCompletionTool :: forall a. (HasFunctionObject a, JSONSchema a) => API.ChatCompletionTool
toChatCompletionTool =
  API.ChatCompletionTool
    { chatCompletionToolType = "function",
      chatCompletionToolFunction =
        API.FunctionObject
          { functionObjectDescription = Just (T.pack $ getFunctionDescription @a),
            functionObjectName = T.pack $ getFunctionName @a,
            functionObjectParameters = Just $
              case toAeson (schema @a) of
                A.Object kv -> M.fromList $ map (\(k, v) -> (A.toString k, v)) $ A.toList kv
                _ -> []
          }
    }

class HasFunctionObject r where
  getFunctionName :: String
  getFunctionDescription :: String
  getFieldDescription :: String -> String

class JSONSchema r where
  schema :: Schema
  default schema :: (HasFunctionObject r, Generic r, GSchema r (Rep r)) => Schema
  schema = gschema @r (from (undefined :: r))

class GSchema s f where
  gschema :: forall a. f a -> Schema

data Schema
  = Maybe' Schema
  | String'
  | Number'
  | Integer'
  | Object' [(String, String, Schema)]
  | Array' Schema
  | Boolean'
  | Null'

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

toolAdd :: forall a. (Tool a) => API.CreateChatCompletionRequest -> API.CreateChatCompletionRequest
toolAdd req =
  let prevTools = case API.createChatCompletionRequestTools req of
        Nothing -> []
        Just v -> v
      newTools = prevTools ++ [toolSchema @a]
   in req {API.createChatCompletionRequestTools = Just newTools}

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
  { sessionNameFrom :: Text
  , sessionNameTo :: Text
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
  deriving (Eq, Show)

