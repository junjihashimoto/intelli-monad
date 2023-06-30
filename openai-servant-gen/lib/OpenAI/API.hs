{-# LANGUAGE DataKinds                  #-}
{-# LANGUAGE DeriveDataTypeable         #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# LANGUAGE DeriveTraversable          #-}
{-# LANGUAGE FlexibleContexts           #-}
{-# LANGUAGE FlexibleInstances          #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE OverloadedStrings          #-}
{-# LANGUAGE RecordWildCards            #-}
{-# LANGUAGE TypeFamilies               #-}
{-# LANGUAGE TypeOperators              #-}
{-# LANGUAGE ViewPatterns               #-}
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
  ) where

import           OpenAI.Types

import           Control.Monad.Catch                (Exception, MonadThrow, throwM)
import           Control.Monad.Except               (ExceptT, runExceptT)
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Reader         (ReaderT (..))
import           Data.Aeson                         (Value)
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
import           Network.Wai                        (Middleware)
import qualified Network.Wai.Handler.Warp           as Warp
import           Servant                            (ServerError, serveWithContextT)
import           Servant.API                        hiding (addHeader)
import           Servant.API.Verbs                  (StdMethod (..), Verb)
import           Servant.Client                     (ClientEnv, Scheme (Http), ClientError, client,
                                                     mkClientEnv, parseBaseUrl)
import           Servant.Client.Core                (baseUrlPort, baseUrlHost)
import           Servant.Client.Internal.HttpClient (ClientM (..))
import           Servant.Server                     (Handler (..), Application, Context (EmptyContext))
import           Servant.Server.StaticFiles         (serveDirectoryFileServer)
import           Web.FormUrlEncoded
import           Web.HttpApiData



data FormCreateFile = FormCreateFile
  { createFileFile :: FilePath
  , createFilePurpose :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateFile
instance ToForm FormCreateFile

data FormCreateImageEdit = FormCreateImageEdit
  { createImageEditImage :: FilePath
  , createImageEditMask :: FilePath
  , createImageEditPrompt :: Text
  , createImageEditN :: Int
  , createImageEditSize :: Text
  , createImageEditResponseFormat :: Text
  , createImageEditUser :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateImageEdit
instance ToForm FormCreateImageEdit

data FormCreateImageVariation = FormCreateImageVariation
  { createImageVariationImage :: FilePath
  , createImageVariationN :: Int
  , createImageVariationSize :: Text
  , createImageVariationResponseFormat :: Text
  , createImageVariationUser :: Text
  } deriving (Show, Eq, Generic, Data)

instance FromForm FormCreateImageVariation
instance ToForm FormCreateImageVariation

data FormCreateTranscription = FormCreateTranscription
  { createTranscriptionFile :: FilePath
  , createTranscriptionModel :: String
  , createTranscriptionPrompt :: Text
  , createTranscriptionResponseFormat :: Text
  , createTranscriptionTemperature :: Double
  , createTranscriptionLanguage :: Text
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


-- | Servant type-level API, generated from the OpenAPI spec for OpenAI.
type OpenAIAPI
    =    "fine-tunes" :> Capture "fine_tune_id" Text :> "cancel" :> Verb 'POST 200 '[JSON] FineTune -- 'cancelFineTune' route
    :<|> "chat" :> "completions" :> ReqBody '[JSON] CreateChatCompletionRequest :> Verb 'POST 200 '[JSON] CreateChatCompletionResponse -- 'createChatCompletion' route
    :<|> "completions" :> ReqBody '[JSON] CreateCompletionRequest :> Verb 'POST 200 '[JSON] CreateCompletionResponse -- 'createCompletion' route
    :<|> "edits" :> ReqBody '[JSON] CreateEditRequest :> Verb 'POST 200 '[JSON] CreateEditResponse -- 'createEdit' route
    :<|> "embeddings" :> ReqBody '[JSON] CreateEmbeddingRequest :> Verb 'POST 200 '[JSON] CreateEmbeddingResponse -- 'createEmbedding' route
    :<|> "files" :> ReqBody '[FormUrlEncoded] FormCreateFile :> Verb 'POST 200 '[JSON] OpenAIFile -- 'createFile' route
    :<|> "fine-tunes" :> ReqBody '[JSON] CreateFineTuneRequest :> Verb 'POST 200 '[JSON] FineTune -- 'createFineTune' route
    :<|> "images" :> "generations" :> ReqBody '[JSON] CreateImageRequest :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImage' route
    :<|> "images" :> "edits" :> ReqBody '[FormUrlEncoded] FormCreateImageEdit :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImageEdit' route
    :<|> "images" :> "variations" :> ReqBody '[FormUrlEncoded] FormCreateImageVariation :> Verb 'POST 200 '[JSON] ImagesResponse -- 'createImageVariation' route
    :<|> "moderations" :> ReqBody '[JSON] CreateModerationRequest :> Verb 'POST 200 '[JSON] CreateModerationResponse -- 'createModeration' route
    :<|> "audio" :> "transcriptions" :> ReqBody '[FormUrlEncoded] FormCreateTranscription :> Verb 'POST 200 '[JSON] CreateTranscriptionResponse -- 'createTranscription' route
    :<|> "audio" :> "translations" :> ReqBody '[FormUrlEncoded] FormCreateTranslation :> Verb 'POST 200 '[JSON] CreateTranslationResponse -- 'createTranslation' route
    :<|> "files" :> Capture "file_id" Text :> Verb 'DELETE 200 '[JSON] DeleteFileResponse -- 'deleteFile' route
    :<|> "models" :> Capture "model" Text :> Verb 'DELETE 200 '[JSON] DeleteModelResponse -- 'deleteModel' route
    :<|> "files" :> Capture "file_id" Text :> "content" :> Verb 'GET 200 '[JSON] Text -- 'downloadFile' route
    :<|> "files" :> Verb 'GET 200 '[JSON] ListFilesResponse -- 'listFiles' route
    :<|> "fine-tunes" :> Capture "fine_tune_id" Text :> "events" :> QueryParam "stream" Bool :> Verb 'GET 200 '[JSON] ListFineTuneEventsResponse -- 'listFineTuneEvents' route
    :<|> "fine-tunes" :> Verb 'GET 200 '[JSON] ListFineTunesResponse -- 'listFineTunes' route
    :<|> "models" :> Verb 'GET 200 '[JSON] ListModelsResponse -- 'listModels' route
    :<|> "files" :> Capture "file_id" Text :> Verb 'GET 200 '[JSON] OpenAIFile -- 'retrieveFile' route
    :<|> "fine-tunes" :> Capture "fine_tune_id" Text :> Verb 'GET 200 '[JSON] FineTune -- 'retrieveFineTune' route
    :<|> "models" :> Capture "model" Text :> Verb 'GET 200 '[JSON] Model -- 'retrieveModel' route
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
data OpenAIBackend m = OpenAIBackend
  { cancelFineTune :: Text -> m FineTune{- ^  -}
  , createChatCompletion :: CreateChatCompletionRequest -> m CreateChatCompletionResponse{- ^  -}
  , createCompletion :: CreateCompletionRequest -> m CreateCompletionResponse{- ^  -}
  , createEdit :: CreateEditRequest -> m CreateEditResponse{- ^  -}
  , createEmbedding :: CreateEmbeddingRequest -> m CreateEmbeddingResponse{- ^  -}
  , createFile :: FormCreateFile -> m OpenAIFile{- ^  -}
  , createFineTune :: CreateFineTuneRequest -> m FineTune{- ^  -}
  , createImage :: CreateImageRequest -> m ImagesResponse{- ^  -}
  , createImageEdit :: FormCreateImageEdit -> m ImagesResponse{- ^  -}
  , createImageVariation :: FormCreateImageVariation -> m ImagesResponse{- ^  -}
  , createModeration :: CreateModerationRequest -> m CreateModerationResponse{- ^  -}
  , createTranscription :: FormCreateTranscription -> m CreateTranscriptionResponse{- ^  -}
  , createTranslation :: FormCreateTranslation -> m CreateTranslationResponse{- ^  -}
  , deleteFile :: Text -> m DeleteFileResponse{- ^  -}
  , deleteModel :: Text -> m DeleteModelResponse{- ^  -}
  , downloadFile :: Text -> m Text{- ^  -}
  , listFiles :: m ListFilesResponse{- ^  -}
  , listFineTuneEvents :: Text -> Maybe Bool -> m ListFineTuneEventsResponse{- ^  -}
  , listFineTunes :: m ListFineTunesResponse{- ^  -}
  , listModels :: m ListModelsResponse{- ^  -}
  , retrieveFile :: Text -> m OpenAIFile{- ^  -}
  , retrieveFineTune :: Text -> m FineTune{- ^  -}
  , retrieveModel :: Text -> m Model{- ^  -}
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

createOpenAIClient :: OpenAIBackend OpenAIClient
createOpenAIClient = OpenAIBackend{..}
  where
    ((coerce -> cancelFineTune) :<|>
     (coerce -> createChatCompletion) :<|>
     (coerce -> createCompletion) :<|>
     (coerce -> createEdit) :<|>
     (coerce -> createEmbedding) :<|>
     (coerce -> createFile) :<|>
     (coerce -> createFineTune) :<|>
     (coerce -> createImage) :<|>
     (coerce -> createImageEdit) :<|>
     (coerce -> createImageVariation) :<|>
     (coerce -> createModeration) :<|>
     (coerce -> createTranscription) :<|>
     (coerce -> createTranslation) :<|>
     (coerce -> deleteFile) :<|>
     (coerce -> deleteModel) :<|>
     (coerce -> downloadFile) :<|>
     (coerce -> listFiles) :<|>
     (coerce -> listFineTuneEvents) :<|>
     (coerce -> listFineTunes) :<|>
     (coerce -> listModels) :<|>
     (coerce -> retrieveFile) :<|>
     (coerce -> retrieveFineTune) :<|>
     (coerce -> retrieveModel) :<|>
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
  => Config -> OpenAIBackend (ExceptT ServerError IO) -> m ()
runOpenAIServer config backend = runOpenAIMiddlewareServer config requestMiddlewareId backend

-- | Run the OpenAI server at the provided host and port.
runOpenAIMiddlewareServer
  :: (MonadIO m, MonadThrow m)
  => Config -> Middleware -> OpenAIBackend (ExceptT ServerError IO) -> m ()
runOpenAIMiddlewareServer Config{..} middleware backend = do
  url <- parseBaseUrl configUrl
  let warpSettings = Warp.defaultSettings
        & Warp.setPort (baseUrlPort url)
        & Warp.setHost (fromString $ baseUrlHost url)
  liftIO $ Warp.runSettings warpSettings $ middleware $ serverWaiApplicationOpenAI backend

-- | Plain "Network.Wai" Application for the OpenAI server.
--
-- Can be used to implement e.g. tests that call the API without a full webserver.
serverWaiApplicationOpenAI :: OpenAIBackend (ExceptT ServerError IO) -> Application
serverWaiApplicationOpenAI backend = serveWithContextT (Proxy :: Proxy OpenAIAPI) context id (serverFromBackend backend)
  where
    context = serverContext
    serverFromBackend OpenAIBackend{..} =
      (coerce cancelFineTune :<|>
       coerce createChatCompletion :<|>
       coerce createCompletion :<|>
       coerce createEdit :<|>
       coerce createEmbedding :<|>
       coerce createFile :<|>
       coerce createFineTune :<|>
       coerce createImage :<|>
       coerce createImageEdit :<|>
       coerce createImageVariation :<|>
       coerce createModeration :<|>
       coerce createTranscription :<|>
       coerce createTranslation :<|>
       coerce deleteFile :<|>
       coerce deleteModel :<|>
       coerce downloadFile :<|>
       coerce listFiles :<|>
       coerce listFineTuneEvents :<|>
       coerce listFineTunes :<|>
       coerce listModels :<|>
       coerce retrieveFile :<|>
       coerce retrieveFineTune :<|>
       coerce retrieveModel :<|>
       serveDirectoryFileServer "static")


serverContext :: Context ('[])
serverContext = EmptyContext
