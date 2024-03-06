{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveDataTypeable         #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE DeriveTraversable          #-}
{-# LANGUAGE DuplicateRecordFields      #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE RecordWildCards            #-}
{-# LANGUAGE TypeFamilies               #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE ViewPatterns               #-}
{-# LANGUAGE MultiParamTypeClasses      #-}

{-# OPTIONS_GHC
-fno-warn-unused-binds -fno-warn-unused-imports -freduction-depth=328 #-}

module OpenAI.API
  ( -- * Client and Server
    Config(..)
  , OpenAIBackend(..)
  , createOpenAIClient
  , runOpenAIServer
  , runOpenAIMiddlewareServer
  , runOpenAIClient
  , runOpenAIClientWithManager
  , callOpenAI
  , OpenAIClient
  , OpenAIClientError(..)
  -- ** Servant
  , OpenAIAPI
  -- ** Plain WAI Application
  , serverWaiApplicationOpenAI
  -- ** Authentication
  , OpenAIAuth(..)
  , clientAuth
  , Protected
  ) where

import           OpenAI.Types

import           Control.Monad.Catch                (Exception, MonadThrow, throwM)
import           Control.Monad.Except               (ExceptT, runExceptT)
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Reader         (ReaderT (..))
import           Data.Aeson                         (Value)
import           Data.ByteString                    (ByteString, fromStrict, toStrict)
import           Data.Coerce                        (coerce)
import           Data.Data                          (Data)
import           Data.Function                      ((&))
import qualified Data.Map                           as Map
import           Data.Monoid                        ((<>))
import           Data.Proxy                         (Proxy (..))
import           Data.Set                           (Set)
import           Data.Text                          (Text)
import qualified Data.Text                          as T
import           Data.Time
import           Data.UUID                          (UUID)
import           GHC.Exts                           (IsString (..))
import           GHC.Generics                       (Generic)
import           Network.HTTP.Client                (Manager, newManager)
import           Network.HTTP.Client.TLS            (tlsManagerSettings)
import           Network.HTTP.Types.Method          (methodOptions)
import           Network.Wai                        (Middleware, Request, requestHeaders)
import qualified Network.Wai.Handler.Warp           as Warp
import           Network.Wai.Middleware.HttpAuth    (extractBearerAuth)
import           Servant                            (ServerError, serveWithContextT, throwError)
import           Servant.API                        hiding (addHeader)
import           Servant.API.Verbs                  (StdMethod (..), Verb)
import           Servant.API.Experimental.Auth      (AuthProtect)
import           Servant.Client                     (ClientEnv, Scheme (Http), ClientError, client,
                                                     mkClientEnv, parseBaseUrl)
import           Servant.Client.Core                (baseUrlPort, baseUrlHost, AuthClientData, AuthenticatedRequest, addHeader, mkAuthenticatedRequest)
import           Servant.Client.Internal.HttpClient (ClientM (..))
import           Servant.Server                     (Handler (..), Application, Context ((:.), EmptyContext))
import           Servant.Server.Experimental.Auth   (AuthHandler, AuthServerData, mkAuthHandler)
import           Servant.Server.StaticFiles         (serveDirectoryFileServer)
import           Web.FormUrlEncoded
import           Web.HttpApiData

import qualified Network.HTTP.Media               as M
import           Data.Data

data FormCreateTranscription = FormCreateTranscription
  { createTranscriptionFile :: FilePath
  , createTranslationModel :: String
  , createTranscriptionLanguage :: Text
  , createTranscriptionPrompt :: Text
  , createTranscriptionResponseFormat :: Text
  , createTranscriptionTemperature :: Double
  , createTranscriptionTimestampGranularities :: [Text]
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateTranscription
instance ToForm FormCreateTranscription

data FormCreateTranslation = FormCreateTranslation
  { createTranslationFile :: FilePath
  , createTranslationModel :: String
  , createTranslationPrompt :: Text
  , createTranslationResponseFormat :: Text
  , createTranslationTemperature :: Double
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateTranslation
instance ToForm FormCreateTranslation

data FormCreateFile = FormCreateFile
  { createFileFile :: FilePath
  , createFilePurpose :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateFile
instance ToForm FormCreateFile

data FormCreateImageEdit = FormCreateImageEdit
  { createImageEditImage :: FilePath
  , createImageEditPrompt :: Text
  , createImageEditMask :: FilePath
--  , createImageVariationModel :: CreateImageEditRequestModel
  , createImageEditN :: Int
  , createImageEditSize :: Text
  , createImageEditResponseFormat :: Text
  , createImageEditUser :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateImageEdit
instance ToForm FormCreateImageEdit

data FormCreateImageVariation = FormCreateImageVariation
  { createImageVariationImage :: FilePath
--  , createImageVariationModel :: CreateImageEditRequestModel
  , createImageVariationN :: Int
  , createImageVariationResponseFormat :: Text
  , createImageVariationSize :: Text
  , createImageVariationUser :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateImageVariation
instance ToForm FormCreateImageVariation


-- | List of elements parsed from a query.
newtype QueryList (p :: CollectionFormat) a = QueryList
  { fromQueryList :: [a]
  } deriving (Functor, Applicative, Monad, Foldable, Traversable)

-- | Formats in which a list can be encoded into a HTTP path.
data CollectionFormat
  = CommaSeparated -- ^ CSV format for multiple parameters.
  | SpaceSeparated -- ^ Also called "SSV"
  | TabSeparated -- ^ Also called "TSV"
  | PipeSeparated -- ^ `value1|value2|value2`
  | MultiParamArray -- ^ Using multiple GET parameters, e.g. `foo=bar&foo=baz`. Only for GET params.

instance FromHttpApiData a => FromHttpApiData (QueryList 'CommaSeparated a) where
  parseQueryParam = parseSeparatedQueryList ','

instance FromHttpApiData a => FromHttpApiData (QueryList 'TabSeparated a) where
  parseQueryParam = parseSeparatedQueryList '\t'

instance FromHttpApiData a => FromHttpApiData (QueryList 'SpaceSeparated a) where
  parseQueryParam = parseSeparatedQueryList ' '

instance FromHttpApiData a => FromHttpApiData (QueryList 'PipeSeparated a) where
  parseQueryParam = parseSeparatedQueryList '|'

instance FromHttpApiData a => FromHttpApiData (QueryList 'MultiParamArray a) where
  parseQueryParam = error "unimplemented FromHttpApiData for MultiParamArray collection format"

parseSeparatedQueryList :: FromHttpApiData a => Char -> Text -> Either Text (QueryList p a)
parseSeparatedQueryList char = fmap QueryList . mapM parseQueryParam . T.split (== char)

instance ToHttpApiData a => ToHttpApiData (QueryList 'CommaSeparated a) where
  toQueryParam = formatSeparatedQueryList ','

instance ToHttpApiData a => ToHttpApiData (QueryList 'TabSeparated a) where
  toQueryParam = formatSeparatedQueryList '\t'

instance ToHttpApiData a => ToHttpApiData (QueryList 'SpaceSeparated a) where
  toQueryParam = formatSeparatedQueryList ' '

instance ToHttpApiData a => ToHttpApiData (QueryList 'PipeSeparated a) where
  toQueryParam = formatSeparatedQueryList '|'

instance ToHttpApiData a => ToHttpApiData (QueryList 'MultiParamArray a) where
  toQueryParam = error "unimplemented ToHttpApiData for MultiParamArray collection format"

formatSeparatedQueryList :: ToHttpApiData a => Char ->  QueryList p a -> Text
formatSeparatedQueryList char = T.intercalate (T.singleton char) . map toQueryParam . fromQueryList

data AudioMpeg deriving Typeable
instance Accept AudioMpeg where
    contentType _ = "audio" M.// "mpeg"

instance MimeRender AudioMpeg ByteString where
    mimeRender _ = fromStrict

instance MimeUnrender AudioMpeg ByteString where
    mimeUnrender _ = Right . toStrict

-- | Servant type-level API, generated from the OpenAPI spec for OpenAI.
type OpenAIAPI
    =    Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> "cancel" :> Verb 'POST 200 '[JSON] RunObject -- 'cancelRun' route
    :<|> Protected :> "assistants" :> ReqBody '[JSON] CreateAssistantRequest :> Verb 'POST 200 '[JSON] AssistantObject -- 'createAssistant' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> "files" :> ReqBody '[JSON] CreateAssistantFileRequest :> Verb 'POST 200 '[JSON] AssistantFileObject -- 'createAssistantFile' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> ReqBody '[JSON] CreateMessageRequest :> Verb 'POST 200 '[JSON] MessageObject -- 'createMessage' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> ReqBody '[JSON] CreateRunRequest :> Verb 'POST 200 '[JSON] RunObject -- 'createRun' route
    :<|> Protected :> "threads" :> ReqBody '[JSON] CreateThreadRequest :> Verb 'POST 200 '[JSON] ThreadObject -- 'createThread' route
    :<|> Protected :> "threads" :> "runs" :> ReqBody '[JSON] CreateThreadAndRunRequest :> Verb 'POST 200 '[JSON] RunObject -- 'createThreadAndRun' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> Verb 'DELETE 200 '[JSON] DeleteAssistantResponse -- 'deleteAssistant' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> "files" :> Capture "file_id" Text :> Verb 'DELETE 200 '[JSON] DeleteAssistantFileResponse -- 'deleteAssistantFile' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> Verb 'DELETE 200 '[JSON] DeleteThreadResponse -- 'deleteThread' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> Verb 'GET 200 '[JSON] AssistantObject -- 'getAssistant' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> "files" :> Capture "file_id" Text :> Verb 'GET 200 '[JSON] AssistantFileObject -- 'getAssistantFile' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> Capture "message_id" Text :> Verb 'GET 200 '[JSON] MessageObject -- 'getMessage' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> Capture "message_id" Text :> "files" :> Capture "file_id" Text :> Verb 'GET 200 '[JSON] MessageFileObject -- 'getMessageFile' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> Verb 'GET 200 '[JSON] RunObject -- 'getRun' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> "steps" :> Capture "step_id" Text :> Verb 'GET 200 '[JSON] RunStepObject -- 'getRunStep' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> Verb 'GET 200 '[JSON] ThreadObject -- 'getThread' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> "files" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListAssistantFilesResponse -- 'listAssistantFiles' route
    :<|> Protected :> "assistants" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListAssistantsResponse -- 'listAssistants' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> Capture "message_id" Text :> "files" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListMessageFilesResponse -- 'listMessageFiles' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListMessagesResponse -- 'listMessages' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> "steps" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListRunStepsResponse -- 'listRunSteps' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> QueryParam "limit" Int :> QueryParam "order" Text :> QueryParam "after" Text :> QueryParam "before" Text :> Verb 'GET 200 '[JSON] ListRunsResponse -- 'listRuns' route
    :<|> Protected :> "assistants" :> Capture "assistant_id" Text :> ReqBody '[JSON] ModifyAssistantRequest :> Verb 'POST 200 '[JSON] AssistantObject -- 'modifyAssistant' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "messages" :> Capture "message_id" Text :> ReqBody '[JSON] ModifyMessageRequest :> Verb 'POST 200 '[JSON] MessageObject -- 'modifyMessage' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> ReqBody '[JSON] ModifyRunRequest :> Verb 'POST 200 '[JSON] RunObject -- 'modifyRun' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> ReqBody '[JSON] ModifyThreadRequest :> Verb 'POST 200 '[JSON] ThreadObject -- 'modifyThread' route
    :<|> Protected :> "threads" :> Capture "thread_id" Text :> "runs" :> Capture "run_id" Text :> "submit_tool_outputs" :> ReqBody '[JSON] SubmitToolOutputsRunRequest :> Verb 'POST 200 '[JSON] RunObject -- 'submitToolOuputsToRun' route
    :<|> Protected :> "audio" :> "speech" :> ReqBody '[JSON] CreateSpeechRequest :> Verb 'POST 200 '[AudioMpeg] ByteString -- 'createSpeech' route
    :<|> Protected :> "audio" :> "transcriptions" :> ReqBody '[FormUrlEncoded] FormCreateTranscription :> Verb 'POST 200 '[JSON] CreateTranscription200Response -- 'createTranscription' route
    :<|> Protected :> "audio" :> "translations" :> ReqBody '[FormUrlEncoded] FormCreateTranslation :> Verb 'POST 200 '[JSON] CreateTranslation200Response -- 'createTranslation' route
    :<|> Protected :> "chat" :> "completions" :> ReqBody '[JSON] CreateChatCompletionRequest :> Verb 'POST 200 '[JSON] CreateChatCompletionResponse -- 'createChatCompletion' route
    :<|> Protected :> "completions" :> ReqBody '[JSON] CreateCompletionRequest :> Verb 'POST 200 '[JSON] CreateCompletionResponse -- 'createCompletion' route
    :<|> Protected :> "embeddings" :> ReqBody '[JSON] CreateEmbeddingRequest :> Verb 'POST 200 '[JSON] CreateEmbeddingResponse -- 'createEmbedding' route
    :<|> Protected :> "files" :> ReqBody '[FormUrlEncoded] FormCreateFile :> Verb 'POST 200 '[JSON] OpenAIFile -- 'createFile' route
    :<|> Protected :> "files" :> Capture "file_id" Text :> Verb 'DELETE 200 '[JSON] DeleteFileResponse -- 'deleteFile' route
    :<|> Protected :> "files" :> Capture "file_id" Text :> "content" :> Verb 'GET 200 '[JSON] Text -- 'downloadFile' route
    :<|> Protected :> "files" :> QueryParam "purpose" Text :> Verb 'GET 200 '[JSON] ListFilesResponse -- 'listFiles' route
    :<|> Protected :> "files" :> Capture "file_id" Text :> Verb 'GET 200 '[JSON] OpenAIFile -- 'retrieveFile' route
    :<|> Protected :> "fine_tuning" :> "jobs" :> Capture "fine_tuning_job_id" Text :> "cancel" :> Verb 'POST 200 '[JSON] FineTuningJob -- 'cancelFineTuningJob' route
    :<|> Protected :> "fine_tuning" :> "jobs" :> ReqBody '[JSON] CreateFineTuningJobRequest :> Verb 'POST 200 '[JSON] FineTuningJob -- 'createFineTuningJob' route
    :<|> Protected :> "fine_tuning" :> "jobs" :> Capture "fine_tuning_job_id" Text :> "events" :> QueryParam "after" Text :> QueryParam "limit" Int :> Verb 'GET 200 '[JSON] ListFineTuningJobEventsResponse -- 'listFineTuningEvents' route
    :<|> Protected :> "fine_tuning" :> "jobs" :> QueryParam "after" Text :> QueryParam "limit" Int :> Verb 'GET 200 '[JSON] ListPaginatedFineTuningJobsResponse -- 'listPaginatedFineTuningJobs' route
    :<|> Protected :> "fine_tuning" :> "jobs" :> Capture "fine_tuning_job_id" Text :> Verb 'GET 200 '[JSON] FineTuningJob -- 'retrieveFineTuningJob' route
    :<|> Protected :> "images" :> "generations" :> ReqBody '[JSON] CreateImageRequest :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImage' route
    :<|> Protected :> "images" :> "edits" :> ReqBody '[FormUrlEncoded] FormCreateImageEdit :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImageEdit' route
    :<|> Protected :> "images" :> "variations" :> ReqBody '[FormUrlEncoded] FormCreateImageVariation :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImageVariation' route
    :<|> Protected :> "models" :> Capture "model" Text :> Verb 'DELETE 200 '[JSON] DeleteModelResponse -- 'deleteModel' route
    :<|> Protected :> "models" :> Verb 'GET 200 '[JSON] ListModelsResponse -- 'listModels' route
    :<|> Protected :> "models" :> Capture "model" Text :> Verb 'GET 200 '[JSON] Model -- 'retrieveModel' route
    :<|> Protected :> "moderations" :> ReqBody '[JSON] CreateModerationRequest :> Verb 'POST 200 '[JSON] CreateModerationResponse -- 'createModeration' route
    :<|> Raw


-- | Server or client configuration, specifying the host and port to query or serve on.
data Config = Config
  { configUrl :: String  -- ^ scheme://hostname:port/path, e.g. "http://localhost:8080/"
  } deriving (Eq, Ord, Show, Read)


-- | Custom exception type for our errors.
newtype OpenAIClientError = OpenAIClientError ClientError
  deriving (Show, Exception)
-- | Configuration, specifying the full url of the service.


-- | Backend for OpenAI.
-- The backend can be used both for the client and the server. The client generated from the OpenAI OpenAPI spec
-- is a backend that executes actions by sending HTTP requests (see @createOpenAIClient@). Alternatively, provided
-- a backend, the API can be served using @runOpenAIMiddlewareServer@.
data OpenAIBackend a m = OpenAIBackend
  { cancelRun :: a -> Text -> Text -> m RunObject{- ^  -}
  , createAssistant :: a -> CreateAssistantRequest -> m AssistantObject{- ^  -}
  , createAssistantFile :: a -> Text -> CreateAssistantFileRequest -> m AssistantFileObject{- ^  -}
  , createMessage :: a -> Text -> CreateMessageRequest -> m MessageObject{- ^  -}
  , createRun :: a -> Text -> CreateRunRequest -> m RunObject{- ^  -}
  , createThread :: a -> CreateThreadRequest -> m ThreadObject{- ^  -}
  , createThreadAndRun :: a -> CreateThreadAndRunRequest -> m RunObject{- ^  -}
  , deleteAssistant :: a -> Text -> m DeleteAssistantResponse{- ^  -}
  , deleteAssistantFile :: a -> Text -> Text -> m DeleteAssistantFileResponse{- ^  -}
  , deleteThread :: a -> Text -> m DeleteThreadResponse{- ^  -}
  , getAssistant :: a -> Text -> m AssistantObject{- ^  -}
  , getAssistantFile :: a -> Text -> Text -> m AssistantFileObject{- ^  -}
  , getMessage :: a -> Text -> Text -> m MessageObject{- ^  -}
  , getMessageFile :: a -> Text -> Text -> Text -> m MessageFileObject{- ^  -}
  , getRun :: a -> Text -> Text -> m RunObject{- ^  -}
  , getRunStep :: a -> Text -> Text -> Text -> m RunStepObject{- ^  -}
  , getThread :: a -> Text -> m ThreadObject{- ^  -}
  , listAssistantFiles :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListAssistantFilesResponse{- ^  -}
  , listAssistants :: a -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListAssistantsResponse{- ^  -}
  , listMessageFiles :: a -> Text -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListMessageFilesResponse{- ^  -}
  , listMessages :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListMessagesResponse{- ^  -}
  , listRunSteps :: a -> Text -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListRunStepsResponse{- ^  -}
  , listRuns :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListRunsResponse{- ^  -}
  , modifyAssistant :: a -> Text -> ModifyAssistantRequest -> m AssistantObject{- ^  -}
  , modifyMessage :: a -> Text -> Text -> ModifyMessageRequest -> m MessageObject{- ^  -}
  , modifyRun :: a -> Text -> Text -> ModifyRunRequest -> m RunObject{- ^  -}
  , modifyThread :: a -> Text -> ModifyThreadRequest -> m ThreadObject{- ^  -}
  , submitToolOuputsToRun :: a -> Text -> Text -> SubmitToolOutputsRunRequest -> m RunObject{- ^  -}
  , createSpeech :: a -> CreateSpeechRequest -> m ByteString{- ^  -}
  , createTranscription :: a -> FormCreateTranscription -> m CreateTranscription200Response{- ^  -}
  , createTranslation :: a -> FormCreateTranslation -> m CreateTranslation200Response{- ^  -}
  , createChatCompletion :: a -> CreateChatCompletionRequest -> m CreateChatCompletionResponse{- ^  -}
  , createCompletion :: a -> CreateCompletionRequest -> m CreateCompletionResponse{- ^  -}
  , createEmbedding :: a -> CreateEmbeddingRequest -> m CreateEmbeddingResponse{- ^  -}
  , createFile :: a -> FormCreateFile -> m OpenAIFile{- ^  -}
  , deleteFile :: a -> Text -> m DeleteFileResponse{- ^  -}
  , downloadFile :: a -> Text -> m Text{- ^  -}
  , listFiles :: a -> Maybe Text -> m ListFilesResponse{- ^  -}
  , retrieveFile :: a -> Text -> m OpenAIFile{- ^  -}
  , cancelFineTuningJob :: a -> Text -> m FineTuningJob{- ^  -}
  , createFineTuningJob :: a -> CreateFineTuningJobRequest -> m FineTuningJob{- ^  -}
  , listFineTuningEvents :: a -> Text -> Maybe Text -> Maybe Int -> m ListFineTuningJobEventsResponse{- ^  -}
  , listPaginatedFineTuningJobs :: a -> Maybe Text -> Maybe Int -> m ListPaginatedFineTuningJobsResponse{- ^  -}
  , retrieveFineTuningJob :: a -> Text -> m FineTuningJob{- ^  -}
  , createImage :: a -> CreateImageRequest -> m ImagesResponse{- ^  -}
  , createImageEdit :: a -> FormCreateImageEdit -> m ImagesResponse{- ^  -}
  , createImageVariation :: a -> FormCreateImageVariation -> m ImagesResponse{- ^  -}
  , deleteModel :: a -> Text -> m DeleteModelResponse{- ^  -}
  , listModels :: a -> m ListModelsResponse{- ^  -}
  , retrieveModel :: a -> Text -> m Model{- ^  -}
  , createModeration :: a -> CreateModerationRequest -> m CreateModerationResponse{- ^  -}
  }

-- | Authentication settings for OpenAI.
-- lookupUser is used to retrieve a user given a header value. The data type can be specified by providing an
-- type instance for AuthServerData. authError is a function that given a request returns a custom error that
-- is returned when the header is not found.
data OpenAIAuth = OpenAIAuth
  { lookupUser :: ByteString -> Handler AuthServer
  , authError :: Request -> ServerError
  }

newtype OpenAIClient a = OpenAIClient
  { runClient :: ClientEnv -> ExceptT ClientError IO a
  } deriving Functor

instance Applicative OpenAIClient where
  pure x = OpenAIClient (\_ -> pure x)
  (OpenAIClient f) <*> (OpenAIClient x) =
    OpenAIClient (\env -> f env <*> x env)

instance Monad OpenAIClient where
  (OpenAIClient a) >>= f =
    OpenAIClient (\env -> do
      value <- a env
      runClient (f value) env)

instance MonadIO OpenAIClient where
  liftIO io = OpenAIClient (\_ -> liftIO io)

createOpenAIClient :: OpenAIBackend AuthClient OpenAIClient
createOpenAIClient = OpenAIBackend{..}
  where
    ((coerce -> cancelRun) :<|>
     (coerce -> createAssistant) :<|>
     (coerce -> createAssistantFile) :<|>
     (coerce -> createMessage) :<|>
     (coerce -> createRun) :<|>
     (coerce -> createThread) :<|>
     (coerce -> createThreadAndRun) :<|>
     (coerce -> deleteAssistant) :<|>
     (coerce -> deleteAssistantFile) :<|>
     (coerce -> deleteThread) :<|>
     (coerce -> getAssistant) :<|>
     (coerce -> getAssistantFile) :<|>
     (coerce -> getMessage) :<|>
     (coerce -> getMessageFile) :<|>
     (coerce -> getRun) :<|>
     (coerce -> getRunStep) :<|>
     (coerce -> getThread) :<|>
     (coerce -> listAssistantFiles) :<|>
     (coerce -> listAssistants) :<|>
     (coerce -> listMessageFiles) :<|>
     (coerce -> listMessages) :<|>
     (coerce -> listRunSteps) :<|>
     (coerce -> listRuns) :<|>
     (coerce -> modifyAssistant) :<|>
     (coerce -> modifyMessage) :<|>
     (coerce -> modifyRun) :<|>
     (coerce -> modifyThread) :<|>
     (coerce -> submitToolOuputsToRun) :<|>
     (coerce -> createSpeech) :<|>
     (coerce -> createTranscription) :<|>
     (coerce -> createTranslation) :<|>
     (coerce -> createChatCompletion) :<|>
     (coerce -> createCompletion) :<|>
     (coerce -> createEmbedding) :<|>
     (coerce -> createFile) :<|>
     (coerce -> deleteFile) :<|>
     (coerce -> downloadFile) :<|>
     (coerce -> listFiles) :<|>
     (coerce -> retrieveFile) :<|>
     (coerce -> cancelFineTuningJob) :<|>
     (coerce -> createFineTuningJob) :<|>
     (coerce -> listFineTuningEvents) :<|>
     (coerce -> listPaginatedFineTuningJobs) :<|>
     (coerce -> retrieveFineTuningJob) :<|>
     (coerce -> createImage) :<|>
     (coerce -> createImageEdit) :<|>
     (coerce -> createImageVariation) :<|>
     (coerce -> deleteModel) :<|>
     (coerce -> listModels) :<|>
     (coerce -> retrieveModel) :<|>
     (coerce -> createModeration) :<|>
     _) = client (Proxy :: Proxy OpenAIAPI)

-- | Run requests in the OpenAIClient monad.
runOpenAIClient :: Config -> OpenAIClient a -> ExceptT ClientError IO a
runOpenAIClient clientConfig cl = do
  manager <- liftIO $ newManager tlsManagerSettings
  runOpenAIClientWithManager manager clientConfig cl

-- | Run requests in the OpenAIClient monad using a custom manager.
runOpenAIClientWithManager :: Manager -> Config -> OpenAIClient a -> ExceptT ClientError IO a
runOpenAIClientWithManager manager Config{..} cl = do
  url <- parseBaseUrl configUrl
  runClient cl $ mkClientEnv manager url

-- | Like @runClient@, but returns the response or throws
--   a OpenAIClientError
callOpenAI
  :: (MonadIO m, MonadThrow m)
  => ClientEnv -> OpenAIClient a -> m a
callOpenAI env f = do
  res <- liftIO $ runExceptT $ runClient f env
  case res of
    Left err       -> throwM (OpenAIClientError err)
    Right response -> pure response


requestMiddlewareId :: Application -> Application
requestMiddlewareId a = a

-- | Run the OpenAI server at the provided host and port.
runOpenAIServer
  :: (MonadIO m, MonadThrow m)
  => Config -> OpenAIAuth -> OpenAIBackend AuthServer (ExceptT ServerError IO) -> m ()
runOpenAIServer config auth backend = runOpenAIMiddlewareServer config requestMiddlewareId auth backend

-- | Run the OpenAI server at the provided host and port.
runOpenAIMiddlewareServer
  :: (MonadIO m, MonadThrow m)
  => Config -> Middleware -> OpenAIAuth -> OpenAIBackend AuthServer (ExceptT ServerError IO) -> m ()
runOpenAIMiddlewareServer Config{..} middleware auth backend = do
  url <- parseBaseUrl configUrl
  let warpSettings = Warp.defaultSettings
        & Warp.setPort (baseUrlPort url)
        & Warp.setHost (fromString $ baseUrlHost url)
  liftIO $ Warp.runSettings warpSettings $ middleware $ serverWaiApplicationOpenAI auth backend

-- | Plain "Network.Wai" Application for the OpenAI server.
--
-- Can be used to implement e.g. tests that call the API without a full webserver.
serverWaiApplicationOpenAI :: OpenAIAuth -> OpenAIBackend AuthServer (ExceptT ServerError IO) -> Application
serverWaiApplicationOpenAI auth backend = serveWithContextT (Proxy :: Proxy OpenAIAPI) context id (serverFromBackend backend)
  where
    context = serverContext auth
    serverFromBackend OpenAIBackend{..} =
      (coerce cancelRun :<|>
       coerce createAssistant :<|>
       coerce createAssistantFile :<|>
       coerce createMessage :<|>
       coerce createRun :<|>
       coerce createThread :<|>
       coerce createThreadAndRun :<|>
       coerce deleteAssistant :<|>
       coerce deleteAssistantFile :<|>
       coerce deleteThread :<|>
       coerce getAssistant :<|>
       coerce getAssistantFile :<|>
       coerce getMessage :<|>
       coerce getMessageFile :<|>
       coerce getRun :<|>
       coerce getRunStep :<|>
       coerce getThread :<|>
       coerce listAssistantFiles :<|>
       coerce listAssistants :<|>
       coerce listMessageFiles :<|>
       coerce listMessages :<|>
       coerce listRunSteps :<|>
       coerce listRuns :<|>
       coerce modifyAssistant :<|>
       coerce modifyMessage :<|>
       coerce modifyRun :<|>
       coerce modifyThread :<|>
       coerce submitToolOuputsToRun :<|>
       coerce createSpeech :<|>
       coerce createTranscription :<|>
       coerce createTranslation :<|>
       coerce createChatCompletion :<|>
       coerce createCompletion :<|>
       coerce createEmbedding :<|>
       coerce createFile :<|>
       coerce deleteFile :<|>
       coerce downloadFile :<|>
       coerce listFiles :<|>
       coerce retrieveFile :<|>
       coerce cancelFineTuningJob :<|>
       coerce createFineTuningJob :<|>
       coerce listFineTuningEvents :<|>
       coerce listPaginatedFineTuningJobs :<|>
       coerce retrieveFineTuningJob :<|>
       coerce createImage :<|>
       coerce createImageEdit :<|>
       coerce createImageVariation :<|>
       coerce deleteModel :<|>
       coerce listModels :<|>
       coerce retrieveModel :<|>
       coerce createModeration :<|>
       serveDirectoryFileServer "static")

-- Authentication is implemented with servants generalized authentication:
-- https://docs.servant.dev/en/stable/tutorial/Authentication.html#generalized-authentication

authHandler :: OpenAIAuth -> AuthHandler Request AuthServer
authHandler OpenAIAuth{..} = mkAuthHandler handler
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

serverContext :: OpenAIAuth -> Context (AuthHandler Request AuthServer ': '[])
serverContext auth = authHandler auth :. EmptyContext
