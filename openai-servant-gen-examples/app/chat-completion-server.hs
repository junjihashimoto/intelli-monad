{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeFamilies #-}

import Control.Monad.IO.Class (liftIO)
import Control.Monad.Trans.Except (ExceptT)
import OpenAI.API
import OpenAI.Types
import Servant (ServerError, err404)
import Servant.API.Experimental.Auth (AuthProtect)
import Servant.Server.Experimental.Auth (AuthServerData)

config :: Config
config =
  Config
    { --  { configUrl :: String  -- ^ scheme://hostname:port/path, e.g. "http://localhost:8080/"
      configUrl = "http://localhost:8080/"
    }

data AuthData = AuthData
  {
  }
  deriving (Eq, Show)

type instance AuthServerData (AuthProtect "bearer") = AuthData

auth :: OpenAIAuth
auth =
  OpenAIAuth
    { --  { lookupUser :: ByteString -> Handler AuthServer
      lookupUser = \_ -> return AuthData,
      --  , authError :: Request -> ServerError
      authError = \_ -> err404
    }

backend :: OpenAIBackend a (ExceptT ServerError IO)
backend =
  OpenAIBackend
    { --  { cancelRun :: a -> Text -> Text -> m RunObject{- ^  -}
      cancelRun = undefined,
      --  , createAssistant :: a -> CreateAssistantRequest -> m AssistantObject{- ^  -}
      createAssistant = undefined,
      --  , createAssistantFile :: a -> Text -> CreateAssistantFileRequest -> m AssistantFileObject{- ^  -}
      createAssistantFile = undefined,
      --  , createMessage :: a -> Text -> CreateMessageRequest -> m MessageObject{- ^  -}
      createMessage = undefined,
      --  , createRun :: a -> Text -> CreateRunRequest -> m RunObject{- ^  -}
      createRun = undefined,
      --  , createThread :: a -> CreateThreadRequest -> m ThreadObject{- ^  -}
      createThread = undefined,
      --  , createThreadAndRun :: a -> CreateThreadAndRunRequest -> m RunObject{- ^  -}
      createThreadAndRun = undefined,
      --  , deleteAssistant :: a -> Text -> m DeleteAssistantResponse{- ^  -}
      deleteAssistant = undefined,
      --  , deleteAssistantFile :: a -> Text -> Text -> m DeleteAssistantFileResponse{- ^  -}
      deleteAssistantFile = undefined,
      --  , deleteThread :: a -> Text -> m DeleteThreadResponse{- ^  -}
      deleteThread = undefined,
      --  , getAssistant :: a -> Text -> m AssistantObject{- ^  -}
      getAssistant = undefined,
      --  , getAssistantFile :: a -> Text -> Text -> m AssistantFileObject{- ^  -}
      getAssistantFile = undefined,
      --  , getMessage :: a -> Text -> Text -> m MessageObject{- ^  -}
      getMessage = undefined,
      --  , getMessageFile :: a -> Text -> Text -> Text -> m MessageFileObject{- ^  -}
      getMessageFile = undefined,
      --  , getRun :: a -> Text -> Text -> m RunObject{- ^  -}
      getRun = undefined,
      --  , getRunStep :: a -> Text -> Text -> Text -> m RunStepObject{- ^  -}
      getRunStep = undefined,
      --  , getThread :: a -> Text -> m ThreadObject{- ^  -}
      getThread = undefined,
      --  , listAssistantFiles :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListAssistantFilesResponse{- ^  -}
      listAssistantFiles = undefined,
      --  , listAssistants :: a -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListAssistantsResponse{- ^  -}
      listAssistants = undefined,
      --  , listMessageFiles :: a -> Text -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListMessageFilesResponse{- ^  -}
      listMessageFiles = undefined,
      --  , listMessages :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListMessagesResponse{- ^  -}
      listMessages = undefined,
      --  , listRunSteps :: a -> Text -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListRunStepsResponse{- ^  -}
      listRunSteps = undefined,
      --  , listRuns :: a -> Text -> Maybe Int -> Maybe Text -> Maybe Text -> Maybe Text -> m ListRunsResponse{- ^  -}
      listRuns = undefined,
      --  , modifyAssistant :: a -> Text -> ModifyAssistantRequest -> m AssistantObject{- ^  -}
      modifyAssistant = undefined,
      --  , modifyMessage :: a -> Text -> Text -> ModifyMessageRequest -> m MessageObject{- ^  -}
      modifyMessage = undefined,
      --  , modifyRun :: a -> Text -> Text -> ModifyRunRequest -> m RunObject{- ^  -}
      modifyRun = undefined,
      --  , modifyThread :: a -> Text -> ModifyThreadRequest -> m ThreadObject{- ^  -}
      modifyThread = undefined,
      --  , submitToolOuputsToRun :: a -> Text -> Text -> SubmitToolOutputsRunRequest -> m RunObject{- ^  -}
      submitToolOuputsToRun = undefined,
      --  , createSpeech :: a -> CreateSpeechRequest -> m (Headers '[Header "Transfer-Encoding" Text] FilePath){- ^  -}
      createSpeech = undefined,
      --  , createTranscription :: a -> FormCreateTranscription -> m CreateTranscription200Response{- ^  -}
      createTranscription = undefined,
      --  , createTranslation :: a -> FormCreateTranslation -> m CreateTranslation200Response{- ^  -}
      createTranslation = undefined,
      --  , createChatCompletion :: a -> CreateChatCompletionRequest -> m CreateChatCompletionResponse{- ^  -}
      createChatCompletion = \_ req -> do
        liftIO $ print req
        return
          CreateChatCompletionResponse
            { createChatCompletionResponseId = "chatcmpl-8zKzKZVra3d7t082fFoB8ZAFxLSWf",
              createChatCompletionResponseObject = "chat.completion",
              createChatCompletionResponseCreated = 1709629378,
              createChatCompletionResponseModel = "gpt-3.5-turbo-0125",
              createChatCompletionResponseChoices =
                [ CreateChatCompletionResponseChoicesInner
                    { createChatCompletionResponseChoicesInnerIndex = 0,
                      createChatCompletionResponseChoicesInnerMessage =
                        ChatCompletionResponseMessage
                          { chatCompletionResponseMessageRole = "assistant",
                            chatCompletionResponseMessageContent = Just "Hello! How can I assist you today?",
                            chatCompletionResponseMessageToolUnderscorecalls = Nothing,
                            chatCompletionResponseMessageFunctionUnderscorecall = Nothing
                          },
                      createChatCompletionResponseChoicesInnerLogprobs = Nothing,
                      createChatCompletionResponseChoicesInnerFinishUnderscorereason = "stop"
                    }
                ],
              createChatCompletionResponseUsage =
                Just $
                  CompletionUsage
                    { completionUsagePromptUnderscoretokens = 8,
                      completionUsageCompletionUnderscoretokens = 9,
                      completionUsageTotalUnderscoretokens = 17
                    },
              createChatCompletionResponseSystemUnderscorefingerprint = Just "fp_b9d4cef803"
            },
      --  , createCompletion :: a -> CreateCompletionRequest -> m CreateCompletionResponse{- ^  -}
      createCompletion = undefined,
      --  , createEmbedding :: a -> CreateEmbeddingRequest -> m CreateEmbeddingResponse{- ^  -}
      createEmbedding = undefined,
      --  , createFile :: a -> FormCreateFile -> m OpenAIFile{- ^  -}
      createFile = undefined,
      --  , deleteFile :: a -> Text -> m DeleteFileResponse{- ^  -}
      deleteFile = undefined,
      --  , downloadFile :: a -> Text -> m Text{- ^  -}
      downloadFile = undefined,
      --  , listFiles :: a -> Maybe Text -> m ListFilesResponse{- ^  -}
      listFiles = undefined,
      --  , retrieveFile :: a -> Text -> m OpenAIFile{- ^  -}
      retrieveFile = undefined,
      --  , cancelFineTuningJob :: a -> Text -> m FineTuningJob{- ^  -}
      cancelFineTuningJob = undefined,
      --  , createFineTuningJob :: a -> CreateFineTuningJobRequest -> m FineTuningJob{- ^  -}
      createFineTuningJob = undefined,
      --  , listFineTuningEvents :: a -> Text -> Maybe Text -> Maybe Int -> m ListFineTuningJobEventsResponse{- ^  -}
      listFineTuningEvents = undefined,
      --  , listPaginatedFineTuningJobs :: a -> Maybe Text -> Maybe Int -> m ListPaginatedFineTuningJobsResponse{- ^  -}
      listPaginatedFineTuningJobs = undefined,
      --  , retrieveFineTuningJob :: a -> Text -> m FineTuningJob{- ^  -}
      retrieveFineTuningJob = undefined,
      --  , createImage :: a -> CreateImageRequest -> m ImagesResponse{- ^  -}
      createImage = undefined,
      --  , createImageEdit :: a -> FormCreateImageEdit -> m ImagesResponse{- ^  -}
      createImageEdit = undefined,
      --  , createImageVariation :: a -> FormCreateImageVariation -> m ImagesResponse{- ^  -}
      createImageVariation = undefined,
      --  , deleteModel :: a -> Text -> m DeleteModelResponse{- ^  -}
      deleteModel = undefined,
      --  , listModels :: a -> m ListModelsResponse{- ^  -}
      listModels = undefined,
      --  , retrieveModel :: a -> Text -> m Model{- ^  -}
      retrieveModel = undefined,
      --  , createModeration :: a -> CreateModerationRequest -> m CreateModerationResponse{- ^  -}
      createModeration = undefined
    }

main :: IO ()
main =
  -- runOpenAIServer :: (MonadIO m, MonadThrow m) => Config -> OpenAIAuth -> OpenAIBackend AuthServer (ExceptT ServerError IO) -> m ()
  runOpenAIServer config auth backend
