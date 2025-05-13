{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveDataTypeable         #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE DeriveTraversable          #-}
{-# LANGUAGE DerivingStrategies         #-}
{-# LANGUAGE DuplicateRecordFields      #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses      #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE OverloadedRecordDot        #-}
{-# LANGUAGE RecordWildCards            #-}
{-# LANGUAGE TypeFamilies               #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE TypeApplications           #-}
{-# LANGUAGE ViewPatterns               #-}

module IntelliMonad.ExternalApis.Ollama where

import           Network.HTTP.Types                 (Status, statusIsSuccessful)
import           Control.Applicative ((<|>))
import           Control.Monad.Catch                (Exception, MonadThrow, throwM)
import           Control.Monad.Except               (ExceptT, runExceptT)
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Reader         (ReaderT (..))
import           Control.Monad.Reader               (MonadReader, ReaderT, ask, runReaderT)
import           Control.Monad                      (unless)
import           Control.Exception                  (evaluate, throwIO)
import           Data.Aeson hiding (Options) -- Avoid conflict with our Options type
import           Data.Aeson.Types (Value, Options) -- For flexible fields like 'format' and 'options'
import           Data.ByteString                    (ByteString, fromStrict, toStrict)
import           Data.Coerce                        (coerce)
import           Data.Function                      ((&))
import           Data.Int (Int64) -- For nanosecond durations
import           Data.Map (Map)
import           Data.Proxy                         (Proxy (..))
import           Data.Text (Text)
import           Data.Time (UTCTime)
import           GHC.Exts                           (IsString (..))
import           GHC.Generics (Generic)
import           Network.HTTP.Client                (Manager, newManager)
import           Network.HTTP.Client.TLS            (tlsManagerSettings)
import           Network.Wai                        (Middleware, Request, requestHeaders)
import           Network.Wai.Middleware.HttpAuth    (extractBearerAuth)
import           Servant                            (ServerError, serveWithContextT, throwError)
import           Servant.API.Verbs
import           Servant.API                        hiding (addHeader)
import           Servant.API.Stream -- For streaming responses
import           Servant.Client                     (ClientEnv, Scheme (Http), ClientError, client, mkClientEnv, parseBaseUrl)
import           Servant.Client.Core                (baseUrlPort, baseUrlHost, AuthClientData, AuthenticatedRequest, addHeader, mkAuthenticatedRequest, StreamingResponse)
import           Servant.Client.Core.RunClient
import qualified Servant.Client.Core.Request      as Core
import           Servant.Client.Internal.HttpClient
import           Servant.Client.Internal.HttpClient.Streaming
import           Servant.Server                     (Handler (..), Application, Context ((:.), EmptyContext))
import           Servant.Server.Experimental.Auth   (AuthHandler, AuthServerData, mkAuthHandler)
import           Servant.Server.StaticFiles         (serveDirectoryFileServer)
import qualified Network.Wai.Handler.Warp           as Warp
import qualified Network.HTTP.Client                as Client
import qualified Data.ByteString.Lazy               as BSL
import qualified Servant.Types.SourceT              as S
import qualified Data.ByteString                    as BS
import           Data.Time.Clock                    (getCurrentTime)
import           Control.Monad.STM                  (atomically)
import           Control.Concurrent.STM.TVar
import Control.Monad.Trans.Class (MonadTrans(..))
import           Control.Monad.Codensity
                 (Codensity (..))

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

instance ReflectMethod (Verb 'POST 200) where
    reflectMethod _ = reflectMethod @POST Proxy

type OllamaAPI =
       Protected :> "api" :> "generate" :> ReqBody '[JSON] GenerateRequest :> Stream Post 200 NewlineFraming JSON (SourceIO GenerateResponse) -- Streaming endpoint
  :<|> Protected :> "api" :> "chat" :> ReqBody '[JSON] ChatRequest :> Stream Post 200 NewlineFraming JSON (SourceIO ChatResponse) -- Streaming endpoint
  :<|> Protected :> "api" :> "create" :> ReqBody '[JSON] CreateModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> Protected :> "api" :> "blobs" :> Capture "digest" Text :> GetNoContent -- Check Blob Exists
  :<|> Protected :> "api" :> "blobs" :> Capture "digest" Text :> ReqBody '[OctetStream] ByteString :> Verb 'POST 201 '[PlainText] NoContent -- Push Blob
  :<|> Protected :> "api" :> "tags" :> Get '[JSON] ListModelsResponse -- List Local Models
  :<|> Protected :> "api" :> "show" :> ReqBody '[JSON] ShowModelRequest :> Post '[JSON] ShowModelResponse -- Show Model Info
  :<|> Protected :> "api" :> "copy" :> ReqBody '[JSON] CopyModelRequest :> PostNoContent -- Copy Model
  :<|> Protected :> "api" :> "delete" :> ReqBody '[JSON] DeleteModelRequest :> DeleteNoContent -- Delete Model
  :<|> Protected :> "api" :> "pull" :> ReqBody '[JSON] PullModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> Protected :> "api" :> "push" :> ReqBody '[JSON] PushModelRequest :> Stream Post 200 NewlineFraming JSON (SourceIO StatusResponse) -- Streaming endpoint
  :<|> Protected :> "api" :> "embed" :> ReqBody '[JSON] GenerateEmbeddingsRequest :> Post '[JSON] GenerateEmbeddingsResponse -- Generate Embeddings (New)
  :<|> Protected :> "api" :> "ps" :> Get '[JSON] ListRunningModelsResponse -- List Running Models
  :<|> Protected :> "api" :> "embeddings" :> ReqBody '[JSON] GenerateEmbeddingsDeprecatedRequest :> Post '[JSON] GenerateEmbeddingsDeprecatedResponse -- Generate Embeddings (Deprecated)
  :<|> Protected :> "api" :> "version" :> Get '[JSON] VersionResponse -- Version
  :<|> Raw

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
  , msgName :: Maybe Text -- The function name
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

type ChatResponses = [ChatResponse]

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

-- Define a proxy for your API
ollamaAPI :: Proxy OllamaAPI
ollamaAPI = Proxy

data OllamaBackend a m = OllamaBackend
  { generateClient :: a -> GenerateRequest -> m (SourceIO GenerateResponse)
  , chatClient :: a -> ChatRequest -> m (SourceIO ChatResponse)
  , createClient :: a -> CreateModelRequest -> m (SourceIO StatusResponse)
  , checkBlobClient :: a -> Text -> m NoContent
  , pushBlobClient :: a -> Text -> ByteString -> m NoContent
  , listTagsClient :: a -> m ListModelsResponse
  , showClient :: a -> ShowModelRequest -> m ShowModelResponse
  , copyClient :: a -> CopyModelRequest -> m NoContent
  , deleteClient :: a -> DeleteModelRequest -> m NoContent
  , pullClient :: a -> PullModelRequest -> m (SourceIO StatusResponse)
  , pushClient :: a -> PushModelRequest -> m (SourceIO StatusResponse)
  , embedClient :: a -> GenerateEmbeddingsRequest -> m GenerateEmbeddingsResponse
  , psClient :: a -> m ListRunningModelsResponse
  , embeddingsDeprecatedClient :: a -> GenerateEmbeddingsDeprecatedRequest -> m GenerateEmbeddingsDeprecatedResponse
  , versionClient :: a -> m VersionResponse
  }

-- | TODO: support UVerb ('acceptStatus' argument, like in 'performRequest' above).
performWithStreamingRequest :: Core.Request -> (StreamingResponse -> IO a) -> Servant.Client.Internal.HttpClient.ClientM a
performWithStreamingRequest req k = do
  ClientEnv m burl cookieJar' createClientRequest _ <- ask
  clientRequest <- liftIO $ createClientRequest burl req
  request <- case cookieJar' of
    Nothing -> pure clientRequest
    Just cj -> liftIO $ do
      now <- getCurrentTime
      atomically $ do
        oldCookieJar <- readTVar cj
        let (newRequest, newCookieJar) =
              Client.insertCookiesIntoRequest
                clientRequest
                oldCookieJar
                now
        writeTVar cj newCookieJar
        pure newRequest
  Servant.Client.Internal.HttpClient.ClientM $ lift $ lift $
      Client.withResponse request m $ \res -> do
          let status = Client.responseStatus res

          -- we throw FailureResponse in IO :(
          unless (statusIsSuccessful status) $ do
              b <- BSL.fromChunks <$> Client.brConsume (Client.responseBody res)
              throwIO $ mkFailureResponse burl req (clientResponseToResponse (const b) res)

          k (clientResponseToResponse (S.fromAction BS.null) res)

instance RunStreamingClient Servant.Client.Internal.HttpClient.ClientM where
  withStreamingRequest = IntelliMonad.ExternalApis.Ollama.performWithStreamingRequest
 
data OllamaAuth = OllamaAuth
  { lookupUser :: ByteString -> Handler AuthServer
  , authError :: Request -> ServerError
  }

newtype OllamaClient a = OllamaClient
  { runClient :: ClientEnv -> ExceptT ClientError IO a
  } deriving Functor

instance Applicative OllamaClient where
  pure x = OllamaClient (\_ -> pure x)
  (OllamaClient f) <*> (OllamaClient x) =
    OllamaClient (\env -> f env <*> x env)

instance Monad OllamaClient where
  (OllamaClient a) >>= f =
    OllamaClient (\env -> do
      value <- a env
      runClient (f value) env)

instance MonadIO OllamaClient where
  liftIO io = OllamaClient (\_ -> liftIO io)

createOllamaClient :: OllamaBackend AuthClient OllamaClient
createOllamaClient = OllamaBackend{..}
  where
    ((coerce -> generateClient) :<|>
     (coerce -> chatClient) :<|>
     (coerce -> createClient) :<|>
     (coerce -> checkBlobClient) :<|>
     (coerce -> pushBlobClient) :<|>
     (coerce -> listTagsClient) :<|>
     (coerce -> showClient) :<|>
     (coerce -> copyClient) :<|>
     (coerce -> deleteClient) :<|>
     (coerce -> pullClient) :<|>
     (coerce -> pushClient) :<|>
     (coerce -> embedClient) :<|>
     (coerce -> psClient) :<|>
     (coerce -> embeddingsDeprecatedClient) :<|>
     (coerce -> versionClient) :<|>
     _) = Servant.Client.client (Proxy :: Proxy OllamaAPI)

-- | Server or client configuration, specifying the host and port to query or serve on.
data OllamaConfig = OllamaConfig
  { configUrl :: String  -- ^ scheme://hostname:port/path, e.g. "http://localhost:8080/"
  } deriving (Eq, Ord, Show, Read)

-- | Custom exception type for our errors.
newtype OllamaClientError = OllamaClientError ClientError
  deriving (Show, Exception)
-- | Configuration, specifying the full url of the service.


-- | Run requests in the OllamaClient monad.
runOllamaClient :: OllamaConfig -> OllamaClient a -> ExceptT ClientError IO a
runOllamaClient clientConfig cl = do
  manager <- liftIO $ newManager tlsManagerSettings
  runOllamaClientWithManager manager clientConfig cl

-- | Run requests in the OllamaClient monad using a custom manager.
runOllamaClientWithManager :: Manager -> OllamaConfig -> OllamaClient a -> ExceptT ClientError IO a
runOllamaClientWithManager manager OllamaConfig{..} cl = do
  url <- parseBaseUrl configUrl
  runClient cl $ mkClientEnv manager url

-- | Like @runClient@, but returns the response or throws
--   a OllamaClientError
callOllama
  :: (MonadIO m, MonadThrow m)
  => ClientEnv -> OllamaClient a -> m a
callOllama env f = do
  res <- liftIO $ runExceptT $ runClient f env
  case res of
    Left err       -> throwM (OllamaClientError err)
    Right response -> pure response


requestMiddlewareId :: Application -> Application
requestMiddlewareId a = a

-- | Run the Ollama server at the provided host and port.
runOllamaServer
  :: (MonadIO m, MonadThrow m)
  => OllamaConfig -> OllamaAuth -> OllamaBackend AuthServer (ExceptT ServerError IO) -> m ()
runOllamaServer config auth backend = runOllamaMiddlewareServer config requestMiddlewareId auth backend

-- | Run the Ollama server at the provided host and port.
runOllamaMiddlewareServer
  :: (MonadIO m, MonadThrow m)
  => OllamaConfig -> Middleware -> OllamaAuth -> OllamaBackend AuthServer (ExceptT ServerError IO) -> m ()
runOllamaMiddlewareServer OllamaConfig{..} middleware auth backend = do
  url <- parseBaseUrl configUrl
  let warpSettings = Warp.defaultSettings
        & Warp.setPort (baseUrlPort url)
        & Warp.setHost (fromString $ baseUrlHost url)
  liftIO $ Warp.runSettings warpSettings $ middleware $ serverWaiApplicationOllama auth backend

serverWaiApplicationOllama :: OllamaAuth -> OllamaBackend AuthServer (ExceptT ServerError IO) -> Application
serverWaiApplicationOllama auth backend = serveWithContextT (Proxy :: Proxy OllamaAPI) context id (serverFromBackend backend)
  where
    context = serverContext auth
    serverFromBackend OllamaBackend{..} =
      (coerce generateClient :<|>
        coerce chatClient :<|>
        coerce createClient :<|>
        coerce checkBlobClient :<|>
        coerce pushBlobClient :<|>
        coerce listTagsClient :<|>
        coerce showClient :<|>
        coerce copyClient :<|>
        coerce deleteClient :<|>
        coerce pullClient :<|>
        coerce pushClient :<|>
        coerce embedClient :<|>
        coerce psClient :<|>
        coerce embeddingsDeprecatedClient :<|>
        coerce versionClient :<|>
        serveDirectoryFileServer "static")

-- Authentication is implemented with servants generalized authentication:
-- https://docs.servant.dev/en/stable/tutorial/Authentication.html#generalized-authentication

authHandler :: OllamaAuth -> AuthHandler Request AuthServer
authHandler OllamaAuth{..} = mkAuthHandler handler
  where
    handler req = case lookup "Authorization" (requestHeaders req) of
      Just header -> case extractBearerAuth header of
        Just key -> lookupUser key
        Nothing -> throwError (authError req)
      Nothing -> throwError (authError req)

type Protected = AuthProtect "bearer"
type AuthServer = AuthServerData Protected
type AuthClient = AuthenticatedRequest Protected
type instance AuthClientData Protected = Text

clientAuth :: Text -> AuthClient
clientAuth key = mkAuthenticatedRequest ("Bearer " <> key) (addHeader "Authorization")

serverContext :: OllamaAuth -> Context (AuthHandler Request AuthServer ': '[])
serverContext auth = authHandler auth :. EmptyContext

-- concatChatResponse :: [ChatResponse] -> ChatResponse
-- concatChatResponse = foldl1 (\a b -> ChatResponse {
--                                chatRespModel = a.chatRespModel <> b.chatRespModel,
--                                chatRespCreatedAt = b.chatRespCreatedAt,
--                                chatRespMessage = a.chatRespMessage <> b.chatRespMessage,
--                                chatRespDone = b.chatRespDone,
--                                chatRespTotalDuration = a.chatRespTotalDuration <> b.chatRespTotalDuration,
--                                chatRespLoadDuration = a.chatRespLoadDuration <> b.chatRespLoadDuration,
--                                chatRespPromptEvalCount = a.chatRespPromptEvalCount <> b.chatRespPromptEvalCount,
--                                chatRespPromptEvalDuration = a.chatRespPromptEvalDuration <> b.chatRespPromptEvalDuration,
--                                chatRespEvalCount = a.chatRespEvalCount <> b.chatRespEvalCount,
--                                chatRespEvalDuration = a.chatRespEvalDuration <> b.chatRespEvalDuration,
--                                chatRespDoneReason = b.chatRespDoneReason
--                                                 })
