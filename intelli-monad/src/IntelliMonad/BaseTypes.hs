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
-- Types.hs, but cleaner.

-- used to break the compile order between toolPolicy, and the remains of Types.hs.

module IntelliMonad.BaseTypes
  (
    ChatCompletion(toRequest, fromResponse),
    Content(Content, contentUser),
    Contents,
    Context(Context, contextBody, contextCreated, contextHeader, contextFooter, contextRequest, contextResponse, contextSessionName, contextToolbox, contextTotalTokens),
    ConstructorSchema(ConstructorSchema),
    CustomInstruction(customHeader, customFooter),
    CustomInstructionProxy(CustomInstructionProxy),
    defaultUTCTime,
    EntityField(ContextSessionName, ContextId),
    FinishReason(FunctionCall, Length, Stop, ToolCalls),
    GSchema(gschema),
    HasFunctionObject(getFieldDescription, getFunctionDescription, getFunctionName),
    Hook(preHook, postHook),
    HookProxy(HookProxy),
    JSONSchema(schema),
    KeyValue(KeyValue, keyValueValue),
    Message(Message, ToolCall, ToolReturn, Image),
    migrateAll,
    PersistProxy(PersistProxy),
    PersistentBackend(Conn, config, deleteKey, deleteSession, getKey, initialize, listKeys, listSessions, load, loadByKey, save, saveContents, setKey, setup),
    Prompt,
    PromptEnv(PromptEnv, backend, context, customInstructions, hooks, inputCallback, outputCallback, timeoutSeconds, tools),
    Schema(Maybe', String', Number', Integer', Object', Array', Boolean', Null', Enum', OneOfUntagged, OneOfTagged),
    SessionName,
    Tool(Output, toolExec, toolFooter, toolFunctionName, toolHeader),
    ToolProxy(ToolProxy),
    Unique(KeyName),
    User(Assistant, System, Tool, User),
    userToText
  ) where

import Prelude (Bool(False, True), Double, Either(Left, Right), Eq, Int, Integer, IO, Ord(compare), Semigroup, Show, String, all, error, flip, fmap, head, length, map, not, null, otherwise, return, undefined, (.), ($), (<>), (++), (==))

import Control.Monad.Trans.State (StateT)

import Control.Monad.IO.Class (MonadIO)

import Control.Monad.Fail (MonadFail)

import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode)

import Data.ByteString (ByteString, fromStrict, toStrict)

import Data.List (nub)

import Data.Kind (Type)

import Data.Maybe (Maybe(Just, Nothing))

import Data.Proxy (Proxy(Proxy))

import Data.Text (Text, intercalate, pack, toLower, unpack)

import Data.Time (Day(ModifiedJulianDay), UTCTime(UTCTime))

import Database.Persist (EntityField, PersistField, PersistValue, Unique, Key, toPersistValue, fromPersistValue)

import Database.Persist.Sqlite (PersistFieldSql, sqlType)

import Database.Persist.TH (persistLowerCase,share, mkPersist, sqlSettings, mkMigrate)

import qualified Louter.Types.Request as Louter (ContentPart(TextPart, ToolCallPart, ToolResultPart, ImagePart), ChatRequest(reqMessages), MessageRole(RoleAssistant, RoleTool, RoleUser, RoleSystem), Message(Message, msgRole, msgContent))
import qualified Louter.Types.Response as Louter (ChatResponse, FinishReason(FinishContentFilter, FinishLength, FinishStop, FinishToolCalls), choiceFinishReason, choiceMessage, choiceToolCalls, functionArguments, functionName, respChoices, rtcFunction, rtcId)

import GHC.Generics (C, Constructor, D, Generic, K1, M1, Rep, S, Selector, U1, (:*:), (:+:), conName, from, selName)

-- For defining the registry of tools and their policies.
import IntelliMonad.ToolPolicy.Types (ToolRegistry)

type SessionName = Text

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
  v -> error $ unpack $ "Undefined role:" <> v

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
    request Louter.ChatRequest
    response Louter.ChatResponse Maybe
    toolbox ToolRegistry
    header [Content]
    body [Content]
    footer [Content]
    totalTokens Int
    sessionName Text
    created UTCTime default=CURRENT_TIME
    deriving Show
    deriving Eq
KeyValue
    namespace Text
    key Text
    value Text
    KeyName namespace key
    deriving Show
    deriving Eq
    deriving Ord
|]

-- Manual Ord instance for Context
-- We compare only the fields that have Ord, ignoring request and response
instance Ord Context where
  compare c1 c2 =
    compare (contextHeader c1, contextBody c1, contextFooter c1,
             contextTotalTokens c1, contextSessionName c1, contextCreated c1)
            (contextHeader c2, contextBody c2, contextFooter c2,
             contextTotalTokens c2, contextSessionName c2, contextCreated c2)

toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> pack err

instance PersistField Louter.ChatRequest where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Louter.ChatRequest where
  sqlType _ = sqlType (Proxy @ByteString)

instance PersistField Louter.ChatResponse where
  toPersistValue = toPV
  fromPersistValue = fromPV

instance PersistFieldSql Louter.ChatResponse where
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

type Contents = [Content]

instance ChatCompletion Contents where
  toRequest orgRequest contents =
    let userToRole = \case
          User -> Louter.RoleUser
          System -> Louter.RoleSystem
          Assistant -> Louter.RoleAssistant
          Tool -> Louter.RoleTool
        messages = flip map contents $ \case
          Content user (Message message) _ _ ->
            Louter.Message (userToRole user) [Louter.TextPart message]
          Content user (Image type' img) _ _ ->
            Louter.Message (userToRole user) [Louter.ImagePart type' img]
          Content user (ToolCall id name args) _ _ ->
            Louter.Message (userToRole user) [Louter.ToolCallPart id name args]
          Content user (ToolReturn id _ res) _ _ ->
            Louter.Message (userToRole user) [Louter.ToolResultPart id res]
     in orgRequest { Louter.reqMessages = messages }

  fromResponse sessionName response =
    let choice = head (Louter.respChoices response)
        message = Louter.choiceMessage choice
        toolCalls = Louter.choiceToolCalls choice
        -- Some LLMs emit tool calls, but still give a finish of Stop.
        finishReason = if not (null toolCalls)
                       then ToolCalls
                       else case Louter.choiceFinishReason choice of
                              Just Louter.FinishStop -> Stop
                              Just Louter.FinishLength -> Length
                              Just Louter.FinishToolCalls -> ToolCalls
                              Just Louter.FinishContentFilter -> ContentFilter
                              Nothing -> Null
        -- If there are tool calls, convert them to Content
        contents = if null toolCalls
                   then [Content Assistant (Message message) sessionName defaultUTCTime]
                   else map (\tc -> Content Assistant
                                      (ToolCall (Louter.rtcId tc)
                                                (Louter.functionName $ Louter.rtcFunction tc)
                                                (Louter.functionArguments $ Louter.rtcFunction tc))
                                      sessionName
                                      defaultUTCTime) toolCalls
     in (contents, finishReason)

    
class ChatCompletion b where
  toRequest :: Louter.ChatRequest -> b -> Louter.ChatRequest
  fromResponse :: Text -> Louter.ChatResponse -> (b, FinishReason)

defaultUTCTime :: UTCTime
defaultUTCTime = UTCTime (ModifiedJulianDay 0) 0

--------------------
-- Prompt Related --
--------------------

type Prompt = StateT PromptEnv

data PromptEnv = PromptEnv
  { tools :: [ToolProxy]
  -- ^ The list of tools that can be used by the LLM via tool calling
  , customInstructions :: [CustomInstructionProxy]
  -- ^ This system sends a prompt that includes headers, bodies and footers. Then the message that LLM outputs is added to bodies. customInstructions generates headers and footers.
  , context :: Context
  -- ^ The request settings like model and prompt logs
  , backend :: PersistProxy
  -- ^ The backend for prompt logging
  , hooks :: [HookProxy]
  -- ^ The hook functions before or after calling LLM
  , timeoutSeconds :: Maybe Int
  -- ^ The timeout in seconds to wait for results. Given to Louter.
  , inputCallback :: Text -> IO (Maybe Text)
  -- ^ The function to call to get IO from the user.
  , outputCallback :: Text -> IO ()
  -- ^ The function to call to get IO back to the user.
  }

------------------
-- Tool Related --
------------------

data ToolProxy = forall t. (Tool t, FromJSON t, ToJSON t, FromJSON (Output t), ToJSON (Output t), HasFunctionObject t, JSONSchema t) => ToolProxy (Proxy t)

class HasFunctionObject r where
  getFunctionName :: String
  getFunctionDescription :: String
  getFieldDescription :: String -> String

-------------------------
-- JSON Schema Related --
-------------------------

-- | Constructor schema for sum types
data ConstructorSchema = ConstructorSchema
  { csName :: Text           -- ^ Constructor name
  , csPayload :: Schema      -- ^ Payload schema
  , csIsNullary :: Bool      -- ^ True for zero-field constructors
  } deriving (Show, Eq)

data Schema
  = Maybe' Schema
  | String'
  | Number'
  | Integer'
  | Object' [(String, String, Schema)]
  | Array' Schema
  | Boolean'
  | Null'
  -- Sum type schemas
  | Enum' [Text]                       -- ^ String enum for nullary constructors
  | OneOfUntagged [ConstructorSchema]  -- ^ Untagged union (distinguishable shapes)
  | OneOfTagged [ConstructorSchema]    -- ^ Tagged union with @tag/@value
  deriving (Show, Eq)

instance Semigroup Schema where
  (<>) (Object' a) (Object' b) = Object' (a <> b)
  (<>) (Array' a) (Array' b) = Array' (a <> b)
  (<>) _ _ = error "Can not concat json value."

-- | Extract schema shape for distinguishability check
-- Two schemas are distinguishable if they have different shapes
schemaShape :: Schema -> Text
schemaShape (Maybe' s) = "maybe:" <> schemaShape s
schemaShape String' = "string"
schemaShape Number' = "number"
schemaShape Integer' = "integer"
schemaShape (Object' fields) = "object:" <> intercalate "," (map (\(n,_,_) -> pack n) fields)
schemaShape (Array' _) = "array"
schemaShape Boolean' = "boolean"
schemaShape Null' = "null"
schemaShape (Enum' _) = "enum"
schemaShape (OneOfUntagged _) = "oneof-untagged"
schemaShape (OneOfTagged _) = "oneof-tagged"

class JSONSchema r where
  schema :: Schema
  default schema :: (HasFunctionObject r, Generic r, GSchema r (Rep r)) => Schema
  schema = gschema @r (from (undefined :: r))

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

--------------------------------
-- Custom Instruction Related --
--------------------------------

class CustomInstruction a where
  customHeader :: a -> Contents
  customFooter :: a -> Contents

data CustomInstructionProxy = forall t. (CustomInstruction t) => CustomInstructionProxy t

------------------
-- Hook Related --
------------------

class Hook a where
  preHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m ()
  postHook :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m ()

data HookProxy = forall t. (Hook t) => HookProxy t

-------------------
-- Proxy Related --
-------------------

data PersistProxy = forall t. (PersistentBackend t) => PersistProxy t

------------------
-- Tool Related --
------------------

class Tool a where
  data Output a :: Type

  toolFunctionName :: Text
  default toolFunctionName :: (HasFunctionObject a) => Text
  toolFunctionName = pack $ getFunctionName @a

  toolExec :: forall p m. (MonadIO m, MonadFail m, PersistentBackend p) => a -> Prompt m (Output a)

  toolHeader :: Contents
  toolHeader = []
  toolFooter :: Contents
  toolFooter = []

---------------------
-- GSchema Related --
---------------------

class GSchema s f where
  gschema :: forall a. f a -> Schema

instance (HasFunctionObject s) => GSchema s U1 where
  gschema _ = Null'

instance (HasFunctionObject s, JSONSchema c) => GSchema s (K1 i c) where
  gschema _ = schema @c

instance (HasFunctionObject s, GSchema s a, GSchema s b) => GSchema s (a :*: b) where
  gschema _ = gschema @s @a undefined <> gschema @s @b undefined

-- | Sum type instance - collects all constructors from both branches
instance (HasFunctionObject s, GSchema s a, GSchema s b) => GSchema s (a :+: b) where
  gschema _ =
    let leftSchema = gschema @s @a undefined
        rightSchema = gschema @s @b undefined
        leftConstructors = extractConstructors leftSchema
        rightConstructors = extractConstructors rightSchema
        allConstructors = leftConstructors ++ rightConstructors
    in chooseSumEncoding allConstructors

-- | Datatype - unwraps single-constructor types to their payload
instance (HasFunctionObject s, GSchema s f) => GSchema s (M1 D c f) where
  gschema _ =
    let innerSchema = gschema @s @f undefined
    in case innerSchema of
         -- Single constructor case: unwrap to just the payload
         Enum' [_] -> Null'  -- Single nullary constructor is just Null
         OneOfUntagged [ConstructorSchema _ payload _] -> payload
         -- Multiple constructors: keep as-is
         _ -> innerSchema

-- | Constructor Metadata - captures constructor name and wraps payload
instance (HasFunctionObject s, GSchema s f, Constructor c) => GSchema s (M1 C c f) where
  gschema _ =
    let name = pack $ conName (undefined :: M1 C c f p)
        normalizedName = normalizeConstructorName name
        payload = gschema @s @f undefined
        isNullary = isNullarySchema payload
    -- Return a single-constructor "sum type" that will be collected by :+:
    -- or used directly if there's only one constructor
    in case payload of
         Null' -> Enum' [normalizedName]  -- Nullary constructor
         _ -> OneOfUntagged [ConstructorSchema normalizedName payload isNullary]

-- | Selector Metadata
instance (HasFunctionObject s, GSchema s f, Selector c) => GSchema s (M1 S c f) where
  gschema a =
    let name = selName a
        desc = getFieldDescription @s name
     in Object' [(name, desc, (gschema @s @f undefined))]

-- | Extract constructors from a schema, wrapping non-sum schemas as single-constructor lists
extractConstructors :: Schema -> [ConstructorSchema]
extractConstructors (Enum' names) = map (\n -> ConstructorSchema n Null' True) names
extractConstructors (OneOfUntagged cs) = cs
extractConstructors (OneOfTagged cs) = cs
extractConstructors other = [ConstructorSchema "" other (isNullarySchema other)]

-- | Normalize constructor name (e.g., "Red" -> "red")
normalizeConstructorName :: Text -> Text
normalizeConstructorName = toLower

-- | Check if a schema represents a nullary constructor
isNullarySchema :: Schema -> Bool
isNullarySchema Null' = True
isNullarySchema _ = False

-- | Choose appropriate sum type encoding based on constructor analysis
chooseSumEncoding :: [ConstructorSchema] -> Schema
chooseSumEncoding constructors
  | null constructors = Null'  -- Shouldn't happen, but handle gracefully
  | isEnum constructors = Enum' (map csName constructors)
  | areShapesDistinguishable constructors = OneOfUntagged constructors
  | otherwise = OneOfTagged constructors

-- | Check if constructor shapes are mutually exclusive (distinguishable)
areShapesDistinguishable :: [ConstructorSchema] -> Bool
areShapesDistinguishable constructors =
  let shapes = map (schemaShape . csPayload) constructors
      uniqueShapes = nub shapes
  in length shapes == length uniqueShapes

-- | Check if all constructors are nullary (enum pattern)
isEnum :: [ConstructorSchema] -> Bool
isEnum = all csIsNullary

-------------------------
-- Persistence Related --
-------------------------

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
