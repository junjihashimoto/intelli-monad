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
import Control.Monad (when)
import Control.Monad.IO.Class
import Control.Monad.Trans.State (StateT)
import Data.Aeson (FromJSON, ToJSON, eitherDecode, encode, Value)
import Data.IORef (IORef, newIORef, readIORef, writeIORef, modifyIORef')
import Data.Map (Map)
import qualified Data.Aeson as A
import qualified Data.Aeson.Key as A
import qualified Data.Aeson.KeyMap as A
import qualified Data.Aeson.KeyMap as HM
import qualified Data.Aeson.Text as A
import Data.List (nub)
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
import qualified Louter.Client as Louter
import qualified Louter.Types.Request as Louter
import qualified Louter.Types.Response as Louter
import qualified Louter.Types.Streaming as Louter
import IntelliMonad.Config (readConfig)
import qualified IntelliMonad.Config as Config
import qualified Data.ByteString as BS
import qualified Data.Text.IO as T
import System.Environment (lookupEnv)
import Data.Aeson.Encode.Pretty (encodePretty)

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

class GSchema s f where
  gschema :: forall a. f a -> Schema

class JSONSchema r where
  schema :: Schema
  default schema :: (HasFunctionObject r, Generic r, GSchema r (Rep r)) => Schema
  schema = gschema @r (from (undefined :: r))

class ChatCompletion b where
  toRequest :: Louter.ChatRequest -> b -> Louter.ChatRequest
  fromResponse :: Text -> Louter.ChatResponse -> (b, FinishReason)

toPV :: (ToJSON a) => a -> PersistValue
toPV = toPersistValue . toStrict . encode

fromPV :: (FromJSON a) => PersistValue -> Either Text a
fromPV json = do
  json' <- fmap fromStrict $ fromPersistValue json
  case eitherDecode json' of
    Right v -> return v
    Left err -> Left $ "Decoding JSON fails : " <> T.pack err

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
  -- Sum type schemas
  Enum' constructors ->
    A.Object
      [ ("type", "string"),
        ("enum", A.Array $ V.fromList $ map A.String constructors)
      ]
  OneOfUntagged constructors ->
    A.Object
      [ ("oneOf", A.Array $ V.fromList $ map constructorToUntaggedSchema constructors)
      ]
  OneOfTagged constructors ->
    A.Object
      [ ("oneOf", A.Array $ V.fromList $ map constructorToTaggedSchema constructors)
      ]

-- | Convert a constructor schema to an untagged JSON schema
constructorToUntaggedSchema :: ConstructorSchema -> A.Value
constructorToUntaggedSchema (ConstructorSchema name payload _) =
  -- For untagged, just emit the payload schema
  -- The constructor name is used for documentation but not in the schema itself
  toAeson payload

-- | Convert a constructor schema to a tagged JSON schema with @tag/@value
constructorToTaggedSchema :: ConstructorSchema -> A.Value
constructorToTaggedSchema (ConstructorSchema name payload isNullary) =
  if isNullary
    then
      -- Nullary constructor: {"@tag": "ConstructorName"}
      A.Object
        [ ("type", "object"),
          ("properties", A.Object
            [ ("@tag", A.Object [("type", "string"), ("const", A.String name)])
            ]),
          ("required", A.Array $ V.fromList [A.String "@tag"]),
          ("additionalProperties", A.Bool False)
        ]
    else
      case payload of
        Object' _ ->
          -- Object payload: flat tagged format {"@tag": "Ctor", "field1": ..., "field2": ...}
          case toAeson payload of
            A.Object payloadObj ->
              let tagProp = ("@tag", A.Object [("type", "string"), ("const", A.String name)])
                  -- Extract properties from payload
                  props = case HM.lookup "properties" payloadObj of
                    Just (A.Object p) -> p
                    _ -> HM.empty
                  -- Extract required fields
                  req = case HM.lookup "required" payloadObj of
                    Just (A.Array r) -> V.toList r
                    _ -> []
                  -- Merge @tag with payload properties
                  allProps = HM.insert "@tag" (A.Object [("type", "string"), ("const", A.String name)]) props
                  allRequired = A.String "@tag" : req
              in A.Object
                  [ ("type", "object"),
                    ("properties", A.Object allProps),
                    ("required", A.Array $ V.fromList allRequired)
                  ]
            _ -> A.Null  -- Shouldn't happen
        _ ->
          -- Non-object payload: nested format {"@tag": "Ctor", "@value": ...}
          A.Object
            [ ("type", "object"),
              ("properties", A.Object
                [ ("@tag", A.Object [("type", "string"), ("const", A.String name)]),
                  ("@value", toAeson payload)
                ]),
              ("required", A.Array $ V.fromList [A.String "@tag", A.String "@value"]),
              ("additionalProperties", A.Bool False)
            ]

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

-- | Helper functions for sum type schema generation

-- | Internal wrapper to track constructor schemas during generic traversal
-- This allows us to distinguish between a single constructor and a sum type
data SchemaOrConstructors
  = SingleSchema Schema  -- ^ Not a sum type, just a regular schema
  | Constructors [ConstructorSchema]  -- ^ Sum type with multiple constructors
  deriving (Show, Eq)

-- | Extract constructors from a schema, wrapping non-sum schemas as single-constructor lists
extractConstructors :: Schema -> [ConstructorSchema]
extractConstructors (Enum' names) = map (\n -> ConstructorSchema n Null' True) names
extractConstructors (OneOfUntagged cs) = cs
extractConstructors (OneOfTagged cs) = cs
extractConstructors other = [ConstructorSchema "" other (isNullarySchema other)]

-- | Normalize constructor name (e.g., "Red" -> "red")
normalizeConstructorName :: Text -> Text
normalizeConstructorName = T.toLower

-- | Check if a schema represents a nullary constructor
isNullarySchema :: Schema -> Bool
isNullarySchema Null' = True
isNullarySchema _ = False

-- | Check if all constructors are nullary (enum pattern)
isEnum :: [ConstructorSchema] -> Bool
isEnum = all csIsNullary

-- | Extract schema shape for distinguishability check
-- Two schemas are distinguishable if they have different shapes
schemaShape :: Schema -> Text
schemaShape String' = "string"
schemaShape Number' = "number"
schemaShape Integer' = "integer"
schemaShape Boolean' = "boolean"
schemaShape Null' = "null"
schemaShape (Array' _) = "array"
schemaShape (Object' fields) = "object:" <> T.intercalate "," (map (\(n,_,_) -> T.pack n) fields)
schemaShape (Maybe' s) = "maybe:" <> schemaShape s
schemaShape (Enum' _) = "enum"
schemaShape (OneOfUntagged _) = "oneof-untagged"
schemaShape (OneOfTagged _) = "oneof-tagged"

-- | Check if constructor shapes are mutually exclusive (distinguishable)
areShapesDistinguishable :: [ConstructorSchema] -> Bool
areShapesDistinguishable constructors =
  let shapes = map (schemaShape . csPayload) constructors
      uniqueShapes = nub shapes
  in length shapes == length uniqueShapes

-- | Choose appropriate sum type encoding based on constructor analysis
chooseSumEncoding :: [ConstructorSchema] -> Schema
chooseSumEncoding constructors
  | null constructors = Null'  -- Shouldn't happen, but handle gracefully
  | isEnum constructors = Enum' (map csName constructors)
  | areShapesDistinguishable constructors = OneOfUntagged constructors
  | otherwise = OneOfTagged constructors

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
  gschema proxy =
    let name = T.pack $ conName (undefined :: M1 C c f p)
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

toolAdd :: forall a. (Tool a, HasFunctionObject a, JSONSchema a) => Louter.ChatRequest -> Louter.ChatRequest
toolAdd req =
  let prevTools = case toTools req of
        [] -> []
        v -> v
      newTools = prevTools ++ [newTool @a Proxy]
   in fromTools req newTools

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


-- FIXME: use mempty here, instead of providing a model name.
defaultRequest :: Louter.ChatRequest
defaultRequest =
  Louter.ChatRequest
    { Louter.reqModel = ""
    , Louter.reqMessages = []
    , Louter.reqTools = []
    , Louter.reqToolChoice = Louter.ToolChoiceAuto
    , Louter.reqTemperature = Nothing
    , Louter.reqMaxTokens = Nothing
    , Louter.reqStream = False
    }

newTool :: forall a. (HasFunctionObject a, JSONSchema a) => Proxy a -> Louter.Tool
newTool (Proxy :: Proxy a) =
  Louter.Tool
    { Louter.toolName = T.pack $ getFunctionName @a
    , Louter.toolDescription = Just (T.pack $ getFunctionDescription @a)
    , Louter.toolParameters = toAeson (schema @a)
    }

toTools :: Louter.ChatRequest -> [Louter.Tool]
toTools req = Louter.reqTools req

fromTools :: Louter.ChatRequest -> [Louter.Tool] -> Louter.ChatRequest
fromTools req tools = req { Louter.reqTools = tools }

fromModel_ :: Text -> Louter.ChatRequest
fromModel_ model =
  (defaultRequest :: Louter.ChatRequest)
    { Louter.reqModel = model
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
  toRequest orgRequest contents =
    let messages = flip map contents $ \case
          Content user (Message message) _ _ ->
            Louter.Message
              { Louter.msgRole = case user of
                  User -> Louter.RoleUser
                  System -> Louter.RoleSystem
                  Assistant -> Louter.RoleAssistant
                  Tool -> Louter.RoleTool
              , Louter.msgContent = [Louter.TextPart message]
              }
          Content user (Image type' img) _ _ ->
            Louter.Message
              { Louter.msgRole = case user of
                  User -> Louter.RoleUser
                  System -> Louter.RoleSystem
                  Assistant -> Louter.RoleAssistant
                  Tool -> Louter.RoleTool
              , Louter.msgContent = [Louter.ImagePart type' img]
              }
          Content user (ToolCall id' name' args') _ _ ->
            -- Tool calls need to be handled differently - for now, convert to text
            Louter.Message
              { Louter.msgRole = Louter.RoleAssistant
              , Louter.msgContent = [Louter.TextPart $ "Tool call: " <> name' <> " with args: " <> args']
              }
          Content user (ToolReturn id' name' ret') _ _ ->
            Louter.Message
              { Louter.msgRole = Louter.RoleTool
              , Louter.msgContent = [Louter.TextPart ret']
              }
     in orgRequest { Louter.reqMessages = messages }

  fromResponse sessionName response =
    let choice = head (Louter.respChoices response)
        message = Louter.choiceMessage choice
        toolCalls = Louter.choiceToolCalls choice
        finishReason = case Louter.choiceFinishReason choice of
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

    
updateRequest :: Louter.ChatRequest -> Contents -> Louter.ChatRequest
updateRequest = toRequest

addTools :: [ToolProxy] -> Louter.ChatRequest -> Louter.ChatRequest
addTools [] req' = req'
addTools (tool : tools') req' =
  case tool of
    (ToolProxy (_ :: Proxy a)) ->
      addTools tools' (toolAdd @a req')

fromModel :: Text -> Louter.ChatRequest
fromModel = fromModel_

runRequest :: forall a. (ChatCompletion a) => Text -> Louter.ChatRequest -> a -> IO ((a, FinishReason), Louter.ChatResponse)
runRequest sessionName defaultReq request = do
  config <- readConfig

  -- Determine backend type (default to OpenAI if not specified)
  let backendType = case Config.backend config of
        Just bt -> bt
        Nothing -> Config.OpenAI

  let louterBackend = case backendType of
        Config.OpenAI -> Louter.BackendOpenAI
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Anthropic -> Louter.BackendAnthropic
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Gemini -> Louter.BackendGemini
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }

  client <- Louter.newClient louterBackend
  let req = toRequest defaultReq request

  openai_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_DEBUG"
  when openai_debug $ do
      liftIO $ do
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  openai_http_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_HTTP_DEBUG"
  when openai_http_debug $ do
      liftIO $ do
        T.putStrLn "========== Backend ==========="
        print louterBackend
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  result <- Louter.chatCompletion client req
  case result of
    Left err -> error $ T.unpack $ "Louter error: " <> err
    Right res -> do
      when openai_http_debug $ do
        T.putStrLn "========== Response =========="
        BS.putStr $ BS.toStrict $ encodePretty res
        T.putStrLn ""
      return $ (fromResponse sessionName res, res)

runRequestStreaming :: forall a. (ChatCompletion a) => Text -> Louter.ChatRequest -> a -> (Text -> IO ()) -> IO ((a, FinishReason), Louter.ChatResponse)
runRequestStreaming sessionName defaultReq request contentCallback = do
  config <- readConfig

  -- Determine backend type (default to OpenAI if not specified)
  let backendType = case Config.backend config of
        Just bt -> bt
        Nothing -> Config.OpenAI

  let louterBackend = case backendType of
        Config.OpenAI -> Louter.BackendOpenAI
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Anthropic -> Louter.BackendAnthropic
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }
        Config.Gemini -> Louter.BackendGemini
          { Louter.backendApiKey = Config.apiKey config
          , Louter.backendBaseUrl = Just (Config.endpoint config)
          , Louter.backendRequiresAuth = not (T.null (Config.apiKey config))
          }

  client <- Louter.newClient louterBackend
  let req = toRequest defaultReq request

  openai_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_DEBUG"
  when openai_debug $ do
      liftIO $ do
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  openai_http_debug <- maybe False (== "1") <$> lookupEnv "OPENAI_HTTP_DEBUG"
  when openai_http_debug $ do
      liftIO $ do
        T.putStrLn "========== Backend ==========="
        print louterBackend
        T.putStrLn "========== Request ==========="
        BS.putStr $ BS.toStrict $ encodePretty req
        T.putStrLn ""

  -- Accumulate response while streaming
  contentRef <- newIORef []
  finishReasonRef <- newIORef "stop"

  -- Stream with callback
  Louter.streamChatWithCallback client req $ \event -> case event of
    Louter.StreamContent txt -> do
      contentCallback txt  -- Call user callback for incremental output
      modifyIORef' contentRef (++ [txt])
    Louter.StreamFinish reason -> do
      writeIORef finishReasonRef reason
    Louter.StreamError err -> do
      error $ T.unpack $ "Louter streaming error: " <> err
    _ -> pure ()  -- Ignore reasoning and tool calls for now

  -- Build response from accumulated content
  fullContent <- T.concat <$> readIORef contentRef
  finishReasonText <- readIORef finishReasonRef

  -- Convert text finish reason to FinishReason type
  let finishReason = case finishReasonText of
        "stop" -> Louter.FinishStop
        "length" -> Louter.FinishLength
        "tool_calls" -> Louter.FinishToolCalls
        "content_filter" -> Louter.FinishContentFilter
        _ -> Louter.FinishStop

  -- Create a minimal ChatResponse for compatibility
  let response = Louter.ChatResponse
        { Louter.respId = "stream-1"
        , Louter.respModel = Louter.reqModel req
        , Louter.respChoices =
            [ Louter.Choice
                { Louter.choiceIndex = 0
                , Louter.choiceMessage = fullContent
                , Louter.choiceToolCalls = []
                , Louter.choiceFinishReason = Just finishReason
                }
            ]
        , Louter.respUsage = Nothing
        }

  when openai_http_debug $ do
    T.putStrLn "==== Response (Accumulated) ===="
    BS.putStr $ BS.toStrict $ encodePretty response
    T.putStrLn ""

  return $ (fromResponse sessionName response, response)
