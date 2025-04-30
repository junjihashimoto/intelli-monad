{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings #-} -- Optional, but convenient for Text literals

module IntelliMonad.ExternalApis.Ollama where

import Data.Aeson hiding (Options) -- Avoid conflict with our Options type
import Data.Aeson.Types (Value, Options) -- For flexible fields like 'format' and 'options'
import Data.ByteString (ByteString)
import Data.Int (Int64) -- For nanosecond durations
import Data.Map (Map)
import Data.Text (Text)
import Data.Time (UTCTime)
import GHC.Generics (Generic)
import Servant.API
import Servant.API.Stream -- For streaming responses
import Control.Applicative ((<|>))

-- Common type alias for nanosecond durations
type Nanoseconds = Int64

-- | Options for model generation/chat (maps to Modelfile parameters)
-- Using Value for flexibility as the exact keys/types aren't fully specified
-- in the top-level API doc, but linked elsewhere.
type ModelOptions = Map Text Value

-- | Represents the format parameter which can be "json" or a JSON schema object
-- Using Value for flexibility. Could be refined with a specific schema type if needed.
type FormatOption = Maybe Value -- Using Maybe Value to represent optional string "json" or object

-- =============================================================================
--  API Type Definition
-- =============================================================================

type OllamaAPI =
       "api" :> "generate" :> ReqBody '[JSON] GenerateRequest :> Stream Post 200 NewlineFraming JSON (SourceIO GenerateResponse) -- Streaming endpoint
  :<|> "api" :> "chat" :> ReqBody '[JSON] ChatRequest :> Stream Post 200 NewlineFraming JSON (SourceIO ChatResponse) -- Streaming endpoint
  :<|> "api" :> "create" :> ReqBody '[JSON] CreateModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> "api" :> "blobs" :> Capture "digest" Text :> GetNoContent -- Check Blob Exists
  :<|> "api" :> "blobs" :> Capture "digest" Text :> ReqBody '[OctetStream] ByteString :> Verb 'POST 201 '[PlainText] NoContent -- Push Blob
  :<|> "api" :> "tags" :> Get '[JSON] ListModelsResponse -- List Local Models
  :<|> "api" :> "show" :> ReqBody '[JSON] ShowModelRequest :> Post '[JSON] ShowModelResponse -- Show Model Info
  :<|> "api" :> "copy" :> ReqBody '[JSON] CopyModelRequest :> PostNoContent -- Copy Model
  :<|> "api" :> "delete" :> ReqBody '[JSON] DeleteModelRequest :> DeleteNoContent -- Delete Model
  :<|> "api" :> "pull" :> ReqBody '[JSON] PullModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> "api" :> "push" :> ReqBody '[JSON] PushModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> "api" :> "embed" :> ReqBody '[JSON] GenerateEmbeddingsRequest :> Post '[JSON] GenerateEmbeddingsResponse -- Generate Embeddings (New)
  :<|> "api" :> "ps" :> Get '[JSON] ListRunningModelsResponse -- List Running Models
  :<|> "api" :> "embeddings" :> ReqBody '[JSON] GenerateEmbeddingsDeprecatedRequest :> Post '[JSON] GenerateEmbeddingsDeprecatedResponse -- Generate Embeddings (Deprecated)
  :<|> "api" :> "version" :> Get '[JSON] VersionResponse -- Version

-- =============================================================================
--  Data Types
-- =============================================================================

-- Common options for Aeson derivation to handle snake_case
aesonOptions :: Options
aesonOptions = defaultOptions { fieldLabelModifier = camelTo2 '_' }

-- -----------------------------------------------------------------------------
-- Generate Completion (/api/generate)
-- -----------------------------------------------------------------------------

data GenerateRequest = GenerateRequest
  { grModel      :: Text
  , grPrompt     :: Maybe Text -- Optional if just loading/unloading
  , grSuffix     :: Maybe Text
  , grImages     :: Maybe [Text] -- List of base64 encoded images
  , grFormat     :: FormatOption
  , grOptions    :: Maybe ModelOptions
  , grSystem     :: Maybe Text
  , grTemplate   :: Maybe Text
  , grStream     :: Maybe Bool -- True by default, affects response format
  , grRaw        :: Maybe Bool
  , grKeepAlive  :: Maybe Text -- e.g., "5m", "0"
  , grContext    :: Maybe [Int] -- Deprecated
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON GenerateRequest where
  toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance FromJSON GenerateRequest where
  parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }


-- Response type covers both streaming chunks and the final summary message.
-- The 'done' field distinguishes them.
data GenerateResponse = GenerateResponse
  { genRespModel             :: Text
  , genRespCreatedAt         :: UTCTime
  , genRespResponse          :: Text -- Partial response in stream, empty or full in final/non-stream
  , genRespDone              :: Bool
  , genRespContext           :: Maybe [Int] -- Only in final response if not raw
  , genRespTotalDuration     :: Maybe Nanoseconds -- Only in final response
  , genRespLoadDuration      :: Maybe Nanoseconds -- Only in final response
  , genRespPromptEvalCount   :: Maybe Int -- Only in final response
  , genRespPromptEvalDuration :: Maybe Nanoseconds -- Only in final response
  , genRespEvalCount         :: Maybe Int -- Only in final response
  , genRespEvalDuration      :: Maybe Nanoseconds -- Only in final response
  , genRespDoneReason        :: Maybe Text -- e.g., "stop", "unload", only in final response
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON GenerateResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 7 n of { "" -> ""; x -> x } }
instance ToJSON GenerateResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 7 n of { "" -> ""; x -> x } }

-- -----------------------------------------------------------------------------
-- Generate Chat Completion (/api/chat)
-- -----------------------------------------------------------------------------

data ChatMessage = ChatMessage
  { msgRole      :: Text -- "system", "user", "assistant", or "tool"
  , msgContent   :: Text
  , msgImages    :: Maybe [Text] -- List of base64 encoded images
  , msgToolCalls :: Maybe [ToolCall] -- Optional list of tool calls
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON ChatMessage where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON ChatMessage where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


data ToolCall = ToolCall
  { tcFunction :: ToolFunction
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON ToolCall where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance FromJSON ToolCall where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }

data ToolFunction = ToolFunction
  { tfName      :: Text
  , tfArguments :: Map Text Value -- Arguments for the function call
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON ToolFunction where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance FromJSON ToolFunction where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }

-- Represents the 'tools' parameter in the chat request
data Tool = Tool
    { toolType     :: Text -- Currently only "function"
    , toolFunction :: ToolFunctionDefinition
    } deriving (Show, Generic, Eq, Ord)

instance ToJSON Tool where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }
instance FromJSON Tool where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }


data ToolFunctionDefinition = ToolFunctionDefinition
    { tfdName        :: Text
    , tfdDescription :: Maybe Text
    , tfdParameters  :: Value -- JSON Schema Object for parameters
    } deriving (Show, Generic, Eq, Ord)

instance ToJSON ToolFunctionDefinition where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON ToolFunctionDefinition where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }

data ChatRequest = ChatRequest
  { crModel     :: Text
  , crMessages  :: [ChatMessage] -- Can be empty to load/unload
  , crTools     :: Maybe [Tool]
  , crFormat    :: FormatOption
  , crOptions   :: Maybe ModelOptions
  , crStream    :: Maybe Bool
  , crKeepAlive :: Maybe Text
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON ChatRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance FromJSON ChatRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }

-- Covers streaming chunks and the final summary message.
data ChatResponse = ChatResponse
  { chatRespModel             :: Text
  , chatRespCreatedAt         :: UTCTime
  , chatRespMessage           :: ChatMessage -- Partial message in stream, potentially empty in final
  , chatRespDone              :: Bool
  , chatRespTotalDuration     :: Maybe Nanoseconds -- Only in final response
  , chatRespLoadDuration      :: Maybe Nanoseconds -- Only in final response
  , chatRespPromptEvalCount   :: Maybe Int -- Only in final response
  , chatRespPromptEvalDuration :: Maybe Nanoseconds -- Only in final response
  , chatRespEvalCount         :: Maybe Int -- Only in final response
  , chatRespEvalDuration      :: Maybe Nanoseconds -- Only in final response
  , chatRespDoneReason        :: Maybe Text -- e.g., "stop", "load", "unload", only in final response
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ChatResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 8 n of { "" -> ""; x -> x } }
instance ToJSON ChatResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 8 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Create Model (/api/create)
-- -----------------------------------------------------------------------------

-- Represents the license field which can be a string or list of strings
newtype License = License Value deriving (Show, Generic, FromJSON, ToJSON, Eq, Ord)

data CreateModelRequest = CreateModelRequest
  { cmrModel      :: Text
  , cmrFrom       :: Maybe Text
  , cmrFiles      :: Maybe (Map Text Text) -- filename -> sha256 digest
  , cmrAdapters   :: Maybe (Map Text Text) -- filename -> sha256 digest
  , cmrTemplate   :: Maybe Text
  , cmrLicense    :: Maybe License -- String or [String]
  , cmrSystem     :: Maybe Text
  , cmrParameters :: Maybe (Map Text Value) -- Modelfile parameters
  , cmrMessages   :: Maybe [ChatMessage]
  , cmrStream     :: Maybe Bool
  , cmrQuantize   :: Maybe Text -- e.g., "q4_K_M"
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON CreateModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON CreateModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }

-- -----------------------------------------------------------------------------
-- Status Response (for /api/create, /api/pull, /api/push)
-- -----------------------------------------------------------------------------

data StatusResponse = StatusResponse
  { statusRespStatus    :: Text
  , statusRespDigest    :: Maybe Text -- Only for push/pull download/upload phases
  , statusRespTotal     :: Maybe Int64 -- Only for push/pull download/upload phases
  , statusRespCompleted :: Maybe Int64 -- Only for push/pull download/upload phases
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON StatusResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 10 n of { "" -> ""; x -> x } }
instance ToJSON StatusResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 10 n of { "" -> ""; x -> x } }

-- -----------------------------------------------------------------------------
-- List Local Models (/api/tags)
-- -----------------------------------------------------------------------------

data ListModelsResponse = ListModelsResponse
  { lmrModels :: [ModelEntry]
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ListModelsResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance ToJSON ListModelsResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


data ModelEntry = ModelEntry
  { meName         :: Text
  , meModifiedAt   :: UTCTime
  , meSize         :: Int64 -- Size in bytes
  , meDigest       :: Text
  , meDetails      :: ModelDetails
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ModelEntry where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance ToJSON ModelEntry where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }


data ModelDetails = ModelDetails
  { mdFormat            :: Text
  , mdFamily            :: Text
  , mdFamilies          :: Maybe [Text]
  , mdParameterSize     :: Text -- e.g., "7B", "13B"
  , mdQuantizationLevel :: Text -- e.g., "Q4_0"
  , mdParentModel       :: Maybe Text -- Added based on /api/ps example, seems relevant here too
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ModelDetails where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance ToJSON ModelDetails where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Show Model Information (/api/show)
-- -----------------------------------------------------------------------------

data ShowModelRequest = ShowModelRequest
  { smrModel   :: Text
  , smrVerbose :: Maybe Bool
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON ShowModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON ShowModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }

data ShowModelResponse = ShowModelResponse
  { showRespModelfile   :: Text
  , showRespParameters  :: Text
  , showRespTemplate    :: Text
  , showRespDetails     :: ShowModelDetails
  , showRespModelInfo   :: Map Text Value -- Flexible map for various model info keys
  , showRespCapabilities :: Maybe [Text]
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ShowModelResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 8 n of { "" -> ""; x -> x } }
instance ToJSON ShowModelResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 8 n of { "" -> ""; x -> x } }

-- Reusing ModelDetails but adding parent_model explicitly if needed
-- Using a specific type here for clarity, though it overlaps with ModelDetails
data ShowModelDetails = ShowModelDetails
  { smdParentModel       :: Text
  , smdFormat            :: Text
  , smdFamily            :: Text
  , smdFamilies          :: Maybe [Text]
  , smdParameterSize     :: Text
  , smdQuantizationLevel :: Text
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ShowModelDetails where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance ToJSON ShowModelDetails where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Copy Model (/api/copy)
-- -----------------------------------------------------------------------------

data CopyModelRequest = CopyModelRequest
  { cprSource      :: Text
  , cprDestination :: Text
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON CopyModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON CopyModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Delete Model (/api/delete)
-- -----------------------------------------------------------------------------

data DeleteModelRequest = DeleteModelRequest
  { dmrModel :: Text
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON DeleteModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON DeleteModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Pull Model (/api/pull)
-- -----------------------------------------------------------------------------

data PullModelRequest = PullModelRequest
  { pmrModel    :: Text
  , pmrInsecure :: Maybe Bool
  , pmrStream   :: Maybe Bool
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON PullModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON PullModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Push Model (/api/push)
-- -----------------------------------------------------------------------------

data PushModelRequest = PushModelRequest
  { psrModel    :: Text -- Must be in <namespace>/<model>:<tag> format
  , psrInsecure :: Maybe Bool
  , psrStream   :: Maybe Bool
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON PushModelRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON PushModelRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Generate Embeddings (/api/embed) - New Endpoint
-- -----------------------------------------------------------------------------

-- Using Either Text [Text] to represent "text or list of text"
data EmbeddingInput = SingleInput Text | MultipleInputs [Text]
  deriving (Show, Generic, Eq, Ord)

-- Custom JSON instances to handle the "text or list of text" format
instance ToJSON EmbeddingInput where
  toJSON (SingleInput t)   = toJSON t
  toJSON (MultipleInputs ts) = toJSON ts

instance FromJSON EmbeddingInput where
  parseJSON v = (SingleInput <$> parseJSON v) <|> (MultipleInputs <$> parseJSON v)


data GenerateEmbeddingsRequest = GenerateEmbeddingsRequest
  { gerModel     :: Text
  , gerInput     :: EmbeddingInput
  , gerTruncate  :: Maybe Bool
  , gerOptions   :: Maybe ModelOptions
  , gerKeepAlive :: Maybe Text
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON GenerateEmbeddingsRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance FromJSON GenerateEmbeddingsRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }

data GenerateEmbeddingsResponse = GenerateEmbeddingsResponse
  { genEmbRespModel :: Text -- Added field based on example response
  , genEmbRespEmbeddings :: [[Double]]
  , genEmbRespTotalDuration :: Maybe Nanoseconds -- Added field based on example response
  , genEmbRespLoadDuration :: Maybe Nanoseconds -- Added field based on example response
  , genEmbRespPromptEvalCount :: Maybe Int -- Added field based on example response
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON GenerateEmbeddingsResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 10 n of { "" -> ""; x -> x } }
instance ToJSON GenerateEmbeddingsResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 10 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- List Running Models (/api/ps)
-- -----------------------------------------------------------------------------

data ListRunningModelsResponse = ListRunningModelsResponse
  { lrmrModels :: [RunningModelEntry]
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON ListRunningModelsResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }
instance ToJSON ListRunningModelsResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }

data RunningModelEntry = RunningModelEntry
  { rmeName      :: Text
  , rmeModel     :: Text -- Redundant? Included based on example
  , rmeSize      :: Int64
  , rmeDigest    :: Text
  , rmeDetails   :: ModelDetails -- Reusing ModelDetails type
  , rmeExpiresAt :: UTCTime
  , rmeSizeVram  :: Int64
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON RunningModelEntry where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }
instance ToJSON RunningModelEntry where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 3 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Generate Embeddings (/api/embeddings) - Deprecated Endpoint
-- -----------------------------------------------------------------------------

data GenerateEmbeddingsDeprecatedRequest = GenerateEmbeddingsDeprecatedRequest
  { gedrModel     :: Text
  , gedrPrompt    :: Text
  , gedrOptions   :: Maybe ModelOptions
  , gedrKeepAlive :: Maybe Text
  } deriving (Show, Generic, Eq, Ord)

instance ToJSON GenerateEmbeddingsDeprecatedRequest where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }
instance FromJSON GenerateEmbeddingsDeprecatedRequest where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 4 n of { "" -> ""; x -> x } }

data GenerateEmbeddingsDeprecatedResponse = GenerateEmbeddingsDeprecatedResponse
  { gedRespEmbedding :: [Double]
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON GenerateEmbeddingsDeprecatedResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 7 n of { "" -> ""; x -> x } }
instance ToJSON GenerateEmbeddingsDeprecatedResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 7 n of { "" -> ""; x -> x } }


-- -----------------------------------------------------------------------------
-- Version (/api/version)
-- -----------------------------------------------------------------------------

data VersionResponse = VersionResponse
  { vrVersion :: Text
  } deriving (Show, Generic, Eq, Ord)

instance FromJSON VersionResponse where
 parseJSON = genericParseJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
instance ToJSON VersionResponse where
 toJSON = genericToJSON aesonOptions { fieldLabelModifier = \n -> case drop 2 n of { "" -> ""; x -> x } }
