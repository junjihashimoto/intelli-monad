{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveDataTypeable         #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# OPTIONS_GHC -fno-warn-unused-binds -fno-warn-unused-imports #-}

module OpenAI.Types (
  AssistantFileObject (..),
  AssistantObject (..),
  AssistantObjectToolsInner (..),
  AssistantToolsCode (..),
  AssistantToolsFunction (..),
  AssistantToolsRetrieval (..),
  ChatCompletionFunctionCallOption (..),
  ChatCompletionFunctions (..),
  ChatCompletionMessageToolCall (..),
  ChatCompletionMessageToolCallChunk (..),
  ChatCompletionMessageToolCallChunkFunction (..),
  ChatCompletionMessageToolCallFunction (..),
  ChatCompletionNamedToolChoice (..),
  ChatCompletionNamedToolChoiceFunction (..),
  ChatCompletionRequestAssistantMessage (..),
  ChatCompletionRequestAssistantMessageFunctionCall (..),
  ChatCompletionRequestFunctionMessage (..),
  ChatCompletionRequestMessage (..),
  ChatCompletionRequestMessageContentPart (..),
  ChatCompletionRequestMessageContentPartImage (..),
  ChatCompletionRequestMessageContentPartImageImageUrl (..),
  ChatCompletionRequestMessageContentPartText (..),
  ChatCompletionRequestSystemMessage (..),
  ChatCompletionRequestToolMessage (..),
  ChatCompletionRequestUserMessage (..),
  ChatCompletionRequestUserMessageContent (..),
  ChatCompletionResponseMessage (..),
  ChatCompletionRole (..),
  ChatCompletionStreamResponseDelta (..),
  ChatCompletionStreamResponseDeltaFunctionCall (..),
  ChatCompletionTokenLogprob (..),
  ChatCompletionTokenLogprobTopLogprobsInner (..),
  ChatCompletionTool (..),
  ChatCompletionToolChoiceOption (..),
  CompletionUsage (..),
  CreateAssistantFileRequest (..),
  CreateAssistantRequest (..),
  CreateAssistantRequestModel (..),
  CreateChatCompletionFunctionResponse (..),
  CreateChatCompletionFunctionResponseChoicesInner (..),
  CreateChatCompletionRequest (..),
  CreateChatCompletionRequestFunctionCall (..),
  CreateChatCompletionRequestModel (..),
  CreateChatCompletionRequestResponseFormat (..),
  CreateChatCompletionRequestStop (..),
  CreateChatCompletionResponse (..),
  CreateChatCompletionResponseChoicesInner (..),
  CreateChatCompletionResponseChoicesInnerLogprobs (..),
  CreateChatCompletionStreamResponse (..),
  CreateChatCompletionStreamResponseChoicesInner (..),
  CreateCompletionRequest (..),
  CreateCompletionRequestModel (..),
  CreateCompletionRequestPrompt (..),
  CreateCompletionRequestStop (..),
  CreateCompletionResponse (..),
  CreateCompletionResponseChoicesInner (..),
  CreateCompletionResponseChoicesInnerLogprobs (..),
  CreateEmbeddingRequest (..),
  CreateEmbeddingRequestInput (..),
  CreateEmbeddingRequestModel (..),
  CreateEmbeddingResponse (..),
  CreateEmbeddingResponseUsage (..),
  CreateFineTuningJobRequest (..),
  CreateFineTuningJobRequestHyperparameters (..),
  CreateFineTuningJobRequestHyperparametersBatchSize (..),
  CreateFineTuningJobRequestHyperparametersLearningRateMultiplier (..),
  CreateFineTuningJobRequestHyperparametersNEpochs (..),
  CreateFineTuningJobRequestModel (..),
  CreateImageRequest (..),
  CreateImageRequestModel (..),
  CreateMessageRequest (..),
  CreateModerationRequest (..),
  CreateModerationRequestInput (..),
  CreateModerationRequestModel (..),
  CreateModerationResponse (..),
  CreateModerationResponseResultsInner (..),
  CreateModerationResponseResultsInnerCategories (..),
  CreateModerationResponseResultsInnerCategoryScores (..),
  CreateRunRequest (..),
  CreateSpeechRequest (..),
  CreateSpeechRequestModel (..),
  CreateThreadAndRunRequest (..),
  CreateThreadAndRunRequestToolsInner (..),
  CreateThreadRequest (..),
  CreateTranscription200Response (..),
  CreateTranscriptionResponseJson (..),
  CreateTranscriptionResponseVerboseJson (..),
  CreateTranslation200Response (..),
  CreateTranslationResponseJson (..),
  CreateTranslationResponseVerboseJson (..),
  DeleteAssistantFileResponse (..),
  DeleteAssistantResponse (..),
  DeleteFileResponse (..),
  DeleteMessageResponse (..),
  DeleteModelResponse (..),
  DeleteThreadResponse (..),
  Embedding (..),
  Error (..),
  ErrorResponse (..),
  FineTuningJob (..),
  FineTuningJobError (..),
  FineTuningJobEvent (..),
  FineTuningJobHyperparameters (..),
  FineTuningJobHyperparametersNEpochs (..),
  FunctionObject (..),
  Image (..),
  ImagesResponse (..),
  ListAssistantFilesResponse (..),
  ListAssistantsResponse (..),
  ListFilesResponse (..),
  ListFineTuningJobEventsResponse (..),
  ListMessageFilesResponse (..),
  ListMessagesResponse (..),
  ListModelsResponse (..),
  ListPaginatedFineTuningJobsResponse (..),
  ListRunStepsResponse (..),
  ListRunsResponse (..),
  ListThreadsResponse (..),
  MessageContentImageFileObject (..),
  MessageContentImageFileObjectImageFile (..),
  MessageContentTextAnnotationsFileCitationObject (..),
  MessageContentTextAnnotationsFileCitationObjectFileCitation (..),
  MessageContentTextAnnotationsFilePathObject (..),
  MessageContentTextAnnotationsFilePathObjectFilePath (..),
  MessageContentTextObject (..),
  MessageContentTextObjectText (..),
  MessageContentTextObjectTextAnnotationsInner (..),
  MessageFileObject (..),
  MessageObject (..),
  MessageObjectContentInner (..),
  Model (..),
  ModifyAssistantRequest (..),
  ModifyMessageRequest (..),
  ModifyRunRequest (..),
  ModifyThreadRequest (..),
  OpenAIFile (..),
  RunCompletionUsage (..),
  RunObject (..),
  RunObjectLastError (..),
  RunObjectRequiredAction (..),
  RunObjectRequiredActionSubmitToolOutputs (..),
  RunStepCompletionUsage (..),
  RunStepDetailsMessageCreationObject (..),
  RunStepDetailsMessageCreationObjectMessageCreation (..),
  RunStepDetailsToolCallsCodeObject (..),
  RunStepDetailsToolCallsCodeObjectCodeInterpreter (..),
  RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner (..),
  RunStepDetailsToolCallsCodeOutputImageObject (..),
  RunStepDetailsToolCallsCodeOutputImageObjectImage (..),
  RunStepDetailsToolCallsCodeOutputLogsObject (..),
  RunStepDetailsToolCallsFunctionObject (..),
  RunStepDetailsToolCallsFunctionObjectFunction (..),
  RunStepDetailsToolCallsObject (..),
  RunStepDetailsToolCallsObjectToolCallsInner (..),
  RunStepDetailsToolCallsRetrievalObject (..),
  RunStepObject (..),
  RunStepObjectLastError (..),
  RunStepObjectStepDetails (..),
  RunToolCallObject (..),
  RunToolCallObjectFunction (..),
  SubmitToolOutputsRunRequest (..),
  SubmitToolOutputsRunRequestToolOutputsInner (..),
  ThreadObject (..),
  TranscriptionSegment (..),
  TranscriptionWord (..),
  ) where

import Data.Data (Data)
import Data.UUID (UUID)
import Data.List (stripPrefix)
import Data.Maybe (fromMaybe)
import Data.Aeson (Value, FromJSON(..), ToJSON(..), genericToJSON, genericParseJSON)
import Data.Aeson.Types (Options(..), defaultOptions)
import Data.Set (Set)
import Data.Text (Text)
import Data.Time
import Data.Swagger (ToSchema, declareNamedSchema)
import qualified Data.Swagger as Swagger
import qualified Data.Char as Char
import qualified Data.Text as T
import qualified Data.Map as Map
import GHC.Generics (Generic)
import Data.Function ((&))


-- | A list of [Files](/docs/api-reference/files) attached to an &#x60;assistant&#x60;.
data AssistantFileObject = AssistantFileObject
  { assistantFileObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , assistantFileObjectObject :: Text -- ^ The object type, which is always `assistant.file`.
  , assistantFileObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the assistant file was created.
  , assistantFileObjectAssistantUnderscoreid :: Text -- ^ The assistant ID that the file is attached to.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantFileObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantFileObject")
instance ToJSON AssistantFileObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantFileObject")


-- | Represents an &#x60;assistant&#x60; that can call the model and use tools.
data AssistantObject = AssistantObject
  { assistantObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , assistantObjectObject :: Text -- ^ The object type, which is always `assistant`.
  , assistantObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the assistant was created.
  , assistantObjectName :: Text -- ^ The name of the assistant. The maximum length is 256 characters. 
  , assistantObjectDescription :: Text -- ^ The description of the assistant. The maximum length is 512 characters. 
  , assistantObjectModel :: Text -- ^ ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them. 
  , assistantObjectInstructions :: Text -- ^ The system instructions that the assistant uses. The maximum length is 32768 characters. 
  , assistantObjectTools :: [AssistantObjectToolsInner] -- ^ A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`. 
  , assistantObjectFileUnderscoreids :: [Text] -- ^ A list of [file](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. 
  , assistantObjectMetadata :: Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantObject")
instance ToJSON AssistantObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantObject")


-- | 
data AssistantObjectToolsInner = AssistantObjectToolsInner
  { assistantObjectToolsInnerType :: Text -- ^ The type of tool being defined: `function`
  , assistantObjectToolsInnerFunction :: FunctionObject -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantObjectToolsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantObjectToolsInner")
instance ToJSON AssistantObjectToolsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantObjectToolsInner")


-- | 
data AssistantToolsCode = AssistantToolsCode
  { assistantToolsCodeType :: Text -- ^ The type of tool being defined: `code_interpreter`
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantToolsCode where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantToolsCode")
instance ToJSON AssistantToolsCode where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantToolsCode")


-- | 
data AssistantToolsFunction = AssistantToolsFunction
  { assistantToolsFunctionType :: Text -- ^ The type of tool being defined: `function`
  , assistantToolsFunctionFunction :: FunctionObject -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantToolsFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantToolsFunction")
instance ToJSON AssistantToolsFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantToolsFunction")


-- | 
data AssistantToolsRetrieval = AssistantToolsRetrieval
  { assistantToolsRetrievalType :: Text -- ^ The type of tool being defined: `retrieval`
  } deriving (Show, Eq, Generic, Data)

instance FromJSON AssistantToolsRetrieval where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "assistantToolsRetrieval")
instance ToJSON AssistantToolsRetrieval where
  toJSON = genericToJSON (removeFieldLabelPrefix False "assistantToolsRetrieval")


-- | Specifying a particular function via &#x60;{\&quot;name\&quot;: \&quot;my_function\&quot;}&#x60; forces the model to call that function. 
data ChatCompletionFunctionCallOption = ChatCompletionFunctionCallOption
  { chatCompletionFunctionCallOptionName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionFunctionCallOption where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionFunctionCallOption")
instance ToJSON ChatCompletionFunctionCallOption where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionFunctionCallOption")


-- | 
data ChatCompletionFunctions = ChatCompletionFunctions
  { chatCompletionFunctionsDescription :: Maybe Text -- ^ A description of what the function does, used by the model to choose when and how to call the function.
  , chatCompletionFunctionsName :: Text -- ^ The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
  , chatCompletionFunctionsParameters :: Maybe (Map.Map String Value) -- ^ The parameters the functions accepts, described as a JSON Schema object. See the [guide](/docs/guides/text-generation/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.   Omitting `parameters` defines a function with an empty parameter list.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionFunctions where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionFunctions")
instance ToJSON ChatCompletionFunctions where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionFunctions")


-- | 
data ChatCompletionMessageToolCall = ChatCompletionMessageToolCall
  { chatCompletionMessageToolCallId :: Text -- ^ The ID of the tool call.
  , chatCompletionMessageToolCallType :: Text -- ^ The type of the tool. Currently, only `function` is supported.
  , chatCompletionMessageToolCallFunction :: ChatCompletionMessageToolCallFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionMessageToolCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionMessageToolCall")
instance ToJSON ChatCompletionMessageToolCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionMessageToolCall")


-- | 
data ChatCompletionMessageToolCallChunk = ChatCompletionMessageToolCallChunk
  { chatCompletionMessageToolCallChunkIndex :: Int -- ^ 
  , chatCompletionMessageToolCallChunkId :: Maybe Text -- ^ The ID of the tool call.
  , chatCompletionMessageToolCallChunkType :: Maybe Text -- ^ The type of the tool. Currently, only `function` is supported.
  , chatCompletionMessageToolCallChunkFunction :: Maybe ChatCompletionMessageToolCallChunkFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionMessageToolCallChunk where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionMessageToolCallChunk")
instance ToJSON ChatCompletionMessageToolCallChunk where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionMessageToolCallChunk")


-- | 
data ChatCompletionMessageToolCallChunkFunction = ChatCompletionMessageToolCallChunkFunction
  { chatCompletionMessageToolCallChunkFunctionName :: Maybe Text -- ^ The name of the function to call.
  , chatCompletionMessageToolCallChunkFunctionArguments :: Maybe Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionMessageToolCallChunkFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionMessageToolCallChunkFunction")
instance ToJSON ChatCompletionMessageToolCallChunkFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionMessageToolCallChunkFunction")


-- | The function that the model called.
data ChatCompletionMessageToolCallFunction = ChatCompletionMessageToolCallFunction
  { chatCompletionMessageToolCallFunctionName :: Text -- ^ The name of the function to call.
  , chatCompletionMessageToolCallFunctionArguments :: Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionMessageToolCallFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionMessageToolCallFunction")
instance ToJSON ChatCompletionMessageToolCallFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionMessageToolCallFunction")


-- | Specifies a tool the model should use. Use to force the model to call a specific function.
data ChatCompletionNamedToolChoice = ChatCompletionNamedToolChoice
  { chatCompletionNamedToolChoiceType :: Text -- ^ The type of the tool. Currently, only `function` is supported.
  , chatCompletionNamedToolChoiceFunction :: ChatCompletionNamedToolChoiceFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionNamedToolChoice where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionNamedToolChoice")
instance ToJSON ChatCompletionNamedToolChoice where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionNamedToolChoice")


-- | 
data ChatCompletionNamedToolChoiceFunction = ChatCompletionNamedToolChoiceFunction
  { chatCompletionNamedToolChoiceFunctionName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionNamedToolChoiceFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionNamedToolChoiceFunction")
instance ToJSON ChatCompletionNamedToolChoiceFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionNamedToolChoiceFunction")


-- | 
data ChatCompletionRequestAssistantMessage = ChatCompletionRequestAssistantMessage
  { chatCompletionRequestAssistantMessageContent :: Maybe Text -- ^ The contents of the assistant message. Required unless `tool_calls` or `function_call` is specified. 
  , chatCompletionRequestAssistantMessageRole :: Text -- ^ The role of the messages author, in this case `assistant`.
  , chatCompletionRequestAssistantMessageName :: Maybe Text -- ^ An optional name for the participant. Provides the model information to differentiate between participants of the same role.
  , chatCompletionRequestAssistantMessageToolUnderscorecalls :: Maybe [ChatCompletionMessageToolCall] -- ^ The tool calls generated by the model, such as function calls.
  , chatCompletionRequestAssistantMessageFunctionUnderscorecall :: Maybe ChatCompletionRequestAssistantMessageFunctionCall -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestAssistantMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestAssistantMessage")
instance ToJSON ChatCompletionRequestAssistantMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestAssistantMessage")


-- | Deprecated and replaced by &#x60;tool_calls&#x60;. The name and arguments of a function that should be called, as generated by the model.
data ChatCompletionRequestAssistantMessageFunctionCall = ChatCompletionRequestAssistantMessageFunctionCall
  { chatCompletionRequestAssistantMessageFunctionCallArguments :: Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  , chatCompletionRequestAssistantMessageFunctionCallName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestAssistantMessageFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestAssistantMessageFunctionCall")
instance ToJSON ChatCompletionRequestAssistantMessageFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestAssistantMessageFunctionCall")


-- | 
data ChatCompletionRequestFunctionMessage = ChatCompletionRequestFunctionMessage
  { chatCompletionRequestFunctionMessageRole :: Text -- ^ The role of the messages author, in this case `function`.
  , chatCompletionRequestFunctionMessageContent :: Text -- ^ The contents of the function message.
  , chatCompletionRequestFunctionMessageName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestFunctionMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestFunctionMessage")
instance ToJSON ChatCompletionRequestFunctionMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestFunctionMessage")


-- | 
data ChatCompletionRequestMessage = ChatCompletionRequestMessage
  { chatCompletionRequestMessageContent :: Text -- ^ The contents of the function message.
  , chatCompletionRequestMessageRole :: Text -- ^ The role of the messages author, in this case `function`.
--  , chatCompletionRequestMessageName :: Text -- ^ The name of the function to call.
--  , chatCompletionRequestMessageToolUnderscorecalls :: Maybe [ChatCompletionMessageToolCall] -- ^ The tool calls generated by the model, such as function calls.
--  , chatCompletionRequestMessageFunctionUnderscorecall :: Maybe ChatCompletionRequestAssistantMessageFunctionCall -- ^ 
--  , chatCompletionRequestMessageToolUnderscorecallUnderscoreid :: Text -- ^ Tool call that this message is responding to.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessage")
instance ToJSON ChatCompletionRequestMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessage")


-- | 
data ChatCompletionRequestMessageContentPart = ChatCompletionRequestMessageContentPart
  { chatCompletionRequestMessageContentPartType :: Text -- ^ The type of the content part.
  , chatCompletionRequestMessageContentPartText :: Text -- ^ The text content.
  , chatCompletionRequestMessageContentPartImageUnderscoreurl :: ChatCompletionRequestMessageContentPartImageImageUrl -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessageContentPart where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessageContentPart")
instance ToJSON ChatCompletionRequestMessageContentPart where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessageContentPart")


-- | 
data ChatCompletionRequestMessageContentPartImage = ChatCompletionRequestMessageContentPartImage
  { chatCompletionRequestMessageContentPartImageType :: Text -- ^ The type of the content part.
  , chatCompletionRequestMessageContentPartImageImageUnderscoreurl :: ChatCompletionRequestMessageContentPartImageImageUrl -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessageContentPartImage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessageContentPartImage")
instance ToJSON ChatCompletionRequestMessageContentPartImage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessageContentPartImage")


-- | 
data ChatCompletionRequestMessageContentPartImageImageUrl = ChatCompletionRequestMessageContentPartImageImageUrl
  { chatCompletionRequestMessageContentPartImageImageUrlUrl :: Text -- ^ Either a URL of the image or the base64 encoded image data.
  , chatCompletionRequestMessageContentPartImageImageUrlDetail :: Maybe Text -- ^ Specifies the detail level of the image. Learn more in the [Vision guide](/docs/guides/vision/low-or-high-fidelity-image-understanding).
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessageContentPartImageImageUrl where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessageContentPartImageImageUrl")
instance ToJSON ChatCompletionRequestMessageContentPartImageImageUrl where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessageContentPartImageImageUrl")


-- | 
data ChatCompletionRequestMessageContentPartText = ChatCompletionRequestMessageContentPartText
  { chatCompletionRequestMessageContentPartTextType :: Text -- ^ The type of the content part.
  , chatCompletionRequestMessageContentPartTextText :: Text -- ^ The text content.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessageContentPartText where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessageContentPartText")
instance ToJSON ChatCompletionRequestMessageContentPartText where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessageContentPartText")


-- | 
data ChatCompletionRequestSystemMessage = ChatCompletionRequestSystemMessage
  { chatCompletionRequestSystemMessageContent :: Text -- ^ The contents of the system message.
  , chatCompletionRequestSystemMessageRole :: Text -- ^ The role of the messages author, in this case `system`.
  , chatCompletionRequestSystemMessageName :: Maybe Text -- ^ An optional name for the participant. Provides the model information to differentiate between participants of the same role.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestSystemMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestSystemMessage")
instance ToJSON ChatCompletionRequestSystemMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestSystemMessage")


-- | 
data ChatCompletionRequestToolMessage = ChatCompletionRequestToolMessage
  { chatCompletionRequestToolMessageRole :: Text -- ^ The role of the messages author, in this case `tool`.
  , chatCompletionRequestToolMessageContent :: Text -- ^ The contents of the tool message.
  , chatCompletionRequestToolMessageToolUnderscorecallUnderscoreid :: Text -- ^ Tool call that this message is responding to.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestToolMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestToolMessage")
instance ToJSON ChatCompletionRequestToolMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestToolMessage")


-- | 
data ChatCompletionRequestUserMessage = ChatCompletionRequestUserMessage
  { chatCompletionRequestUserMessageContent :: ChatCompletionRequestUserMessageContent -- ^ 
  , chatCompletionRequestUserMessageRole :: Text -- ^ The role of the messages author, in this case `user`.
  , chatCompletionRequestUserMessageName :: Maybe Text -- ^ An optional name for the participant. Provides the model information to differentiate between participants of the same role.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestUserMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestUserMessage")
instance ToJSON ChatCompletionRequestUserMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestUserMessage")


-- | The contents of the user message. 
data ChatCompletionRequestUserMessageContent = ChatCompletionRequestUserMessageContent
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestUserMessageContent where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestUserMessageContent")
instance ToJSON ChatCompletionRequestUserMessageContent where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestUserMessageContent")


-- | A chat completion message generated by the model.
data ChatCompletionResponseMessage = ChatCompletionResponseMessage
  { chatCompletionResponseMessageContent :: Text -- ^ The contents of the message.
  , chatCompletionResponseMessageToolUnderscorecalls :: Maybe [ChatCompletionMessageToolCall] -- ^ The tool calls generated by the model, such as function calls.
  , chatCompletionResponseMessageRole :: Text -- ^ The role of the author of this message.
  , chatCompletionResponseMessageFunctionUnderscorecall :: Maybe ChatCompletionRequestAssistantMessageFunctionCall -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionResponseMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionResponseMessage")
instance ToJSON ChatCompletionResponseMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionResponseMessage")


-- | The role of the author of a message
data ChatCompletionRole = ChatCompletionRole
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRole where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRole")
instance ToJSON ChatCompletionRole where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRole")


-- | A chat completion delta generated by streamed model responses.
data ChatCompletionStreamResponseDelta = ChatCompletionStreamResponseDelta
  { chatCompletionStreamResponseDeltaContent :: Maybe Text -- ^ The contents of the chunk message.
  , chatCompletionStreamResponseDeltaFunctionUnderscorecall :: Maybe ChatCompletionStreamResponseDeltaFunctionCall -- ^ 
  , chatCompletionStreamResponseDeltaToolUnderscorecalls :: Maybe [ChatCompletionMessageToolCallChunk] -- ^ 
  , chatCompletionStreamResponseDeltaRole :: Maybe Text -- ^ The role of the author of this message.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionStreamResponseDelta where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionStreamResponseDelta")
instance ToJSON ChatCompletionStreamResponseDelta where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionStreamResponseDelta")


-- | Deprecated and replaced by &#x60;tool_calls&#x60;. The name and arguments of a function that should be called, as generated by the model.
data ChatCompletionStreamResponseDeltaFunctionCall = ChatCompletionStreamResponseDeltaFunctionCall
  { chatCompletionStreamResponseDeltaFunctionCallArguments :: Maybe Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  , chatCompletionStreamResponseDeltaFunctionCallName :: Maybe Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionStreamResponseDeltaFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionStreamResponseDeltaFunctionCall")
instance ToJSON ChatCompletionStreamResponseDeltaFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionStreamResponseDeltaFunctionCall")


-- | 
data ChatCompletionTokenLogprob = ChatCompletionTokenLogprob
  { chatCompletionTokenLogprobToken :: Text -- ^ The token.
  , chatCompletionTokenLogprobLogprob :: Double -- ^ The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely.
  , chatCompletionTokenLogprobBytes :: [Int] -- ^ A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.
  , chatCompletionTokenLogprobTopUnderscorelogprobs :: [ChatCompletionTokenLogprobTopLogprobsInner] -- ^ List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionTokenLogprob where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionTokenLogprob")
instance ToJSON ChatCompletionTokenLogprob where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionTokenLogprob")


-- | 
data ChatCompletionTokenLogprobTopLogprobsInner = ChatCompletionTokenLogprobTopLogprobsInner
  { chatCompletionTokenLogprobTopLogprobsInnerToken :: Text -- ^ The token.
  , chatCompletionTokenLogprobTopLogprobsInnerLogprob :: Double -- ^ The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely.
  , chatCompletionTokenLogprobTopLogprobsInnerBytes :: [Int] -- ^ A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionTokenLogprobTopLogprobsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionTokenLogprobTopLogprobsInner")
instance ToJSON ChatCompletionTokenLogprobTopLogprobsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionTokenLogprobTopLogprobsInner")


-- | 
data ChatCompletionTool = ChatCompletionTool
  { chatCompletionToolType :: Text -- ^ The type of the tool. Currently, only `function` is supported.
  , chatCompletionToolFunction :: FunctionObject -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionTool where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionTool")
instance ToJSON ChatCompletionTool where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionTool")


-- | Controls which (if any) function is called by the model. &#x60;none&#x60; means the model will not call a function and instead generates a message. &#x60;auto&#x60; means the model can pick between generating a message or calling a function. Specifying a particular function via &#x60;{\&quot;type\&quot;: \&quot;function\&quot;, \&quot;function\&quot;: {\&quot;name\&quot;: \&quot;my_function\&quot;}}&#x60; forces the model to call that function.  &#x60;none&#x60; is the default when no functions are present. &#x60;auto&#x60; is the default if functions are present. 
data ChatCompletionToolChoiceOption = ChatCompletionToolChoiceOption
  { chatCompletionToolChoiceOptionType :: Text -- ^ The type of the tool. Currently, only `function` is supported.
  , chatCompletionToolChoiceOptionFunction :: ChatCompletionNamedToolChoiceFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionToolChoiceOption where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionToolChoiceOption")
instance ToJSON ChatCompletionToolChoiceOption where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionToolChoiceOption")


-- | Usage statistics for the completion request.
data CompletionUsage = CompletionUsage
  { completionUsageCompletionUnderscoretokens :: Int -- ^ Number of tokens in the generated completion.
  , completionUsagePromptUnderscoretokens :: Int -- ^ Number of tokens in the prompt.
  , completionUsageTotalUnderscoretokens :: Int -- ^ Total number of tokens used in the request (prompt + completion).
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CompletionUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "completionUsage")
instance ToJSON CompletionUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "completionUsage")


-- | 
data CreateAssistantFileRequest = CreateAssistantFileRequest
  { createAssistantFileRequestFileUnderscoreid :: Text -- ^ A [File](/docs/api-reference/files) ID (with `purpose=\"assistants\"`) that the assistant should use. Useful for tools like `retrieval` and `code_interpreter` that can access files.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateAssistantFileRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createAssistantFileRequest")
instance ToJSON CreateAssistantFileRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createAssistantFileRequest")


-- | 
data CreateAssistantRequest = CreateAssistantRequest
  { createAssistantRequestModel :: CreateAssistantRequestModel -- ^ 
  , createAssistantRequestName :: Maybe Text -- ^ The name of the assistant. The maximum length is 256 characters. 
  , createAssistantRequestDescription :: Maybe Text -- ^ The description of the assistant. The maximum length is 512 characters. 
  , createAssistantRequestInstructions :: Maybe Text -- ^ The system instructions that the assistant uses. The maximum length is 32768 characters. 
  , createAssistantRequestTools :: Maybe [AssistantObjectToolsInner] -- ^ A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`. 
  , createAssistantRequestFileUnderscoreids :: Maybe [Text] -- ^ A list of [file](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. 
  , createAssistantRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateAssistantRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createAssistantRequest")
instance ToJSON CreateAssistantRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createAssistantRequest")


-- | ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them. 
data CreateAssistantRequestModel = CreateAssistantRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateAssistantRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createAssistantRequestModel")
instance ToJSON CreateAssistantRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createAssistantRequestModel")


-- | Represents a chat completion response returned by model, based on the provided input.
data CreateChatCompletionFunctionResponse = CreateChatCompletionFunctionResponse
  { createChatCompletionFunctionResponseId :: Text -- ^ A unique identifier for the chat completion.
  , createChatCompletionFunctionResponseChoices :: [CreateChatCompletionFunctionResponseChoicesInner] -- ^ A list of chat completion choices. Can be more than one if `n` is greater than 1.
  , createChatCompletionFunctionResponseCreated :: Int -- ^ The Unix timestamp (in seconds) of when the chat completion was created.
  , createChatCompletionFunctionResponseModel :: Text -- ^ The model used for the chat completion.
  , createChatCompletionFunctionResponseSystemUnderscorefingerprint :: Maybe Text -- ^ This fingerprint represents the backend configuration that the model runs with.  Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism. 
  , createChatCompletionFunctionResponseObject :: Text -- ^ The object type, which is always `chat.completion`.
  , createChatCompletionFunctionResponseUsage :: Maybe CompletionUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionFunctionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionFunctionResponse")
instance ToJSON CreateChatCompletionFunctionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionFunctionResponse")


-- | 
data CreateChatCompletionFunctionResponseChoicesInner = CreateChatCompletionFunctionResponseChoicesInner
  { createChatCompletionFunctionResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, or `function_call` if the model called a function. 
  , createChatCompletionFunctionResponseChoicesInnerIndex :: Int -- ^ The index of the choice in the list of choices.
  , createChatCompletionFunctionResponseChoicesInnerMessage :: ChatCompletionResponseMessage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionFunctionResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionFunctionResponseChoicesInner")
instance ToJSON CreateChatCompletionFunctionResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionFunctionResponseChoicesInner")


-- | 
data CreateChatCompletionRequest = CreateChatCompletionRequest
  { createChatCompletionRequestMessages :: [ChatCompletionRequestMessage] -- ^ A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
  , createChatCompletionRequestModel :: CreateChatCompletionRequestModel -- ^ 
  , createChatCompletionRequestFrequencyUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/guides/text-generation/parameter-details) 
  , createChatCompletionRequestLogitUnderscorebias :: Maybe (Map.Map String Int) -- ^ Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token. 
  , createChatCompletionRequestLogprobs :: Maybe Bool -- ^ Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`. This option is currently not available on the `gpt-4-vision-preview` model.
  , createChatCompletionRequestTopUnderscorelogprobs :: Maybe Int -- ^ An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.
  , createChatCompletionRequestMaxUnderscoretokens :: Maybe Int -- ^ The maximum number of [tokens](/tokenizer) that can be generated in the chat completion.  The total length of input tokens and generated tokens is limited by the model's context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens. 
  , createChatCompletionRequestN :: Maybe Int -- ^ How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep `n` as `1` to minimize costs.
  , createChatCompletionRequestPresenceUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/guides/text-generation/parameter-details) 
  , createChatCompletionRequestResponseUnderscoreformat :: Maybe CreateChatCompletionRequestResponseFormat -- ^ 
  , createChatCompletionRequestSeed :: Maybe Int -- ^ This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result. Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend. 
  , createChatCompletionRequestStop :: Maybe CreateChatCompletionRequestStop -- ^ 
  , createChatCompletionRequestStream :: Maybe Bool -- ^ If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions). 
  , createChatCompletionRequestTemperature :: Maybe Double -- ^ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. 
  , createChatCompletionRequestTopUnderscorep :: Maybe Double -- ^ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. 
  , createChatCompletionRequestTools :: Maybe [ChatCompletionTool] -- ^ A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. 
  , createChatCompletionRequestToolUnderscorechoice :: Maybe ChatCompletionToolChoiceOption -- ^ 
  , createChatCompletionRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  , createChatCompletionRequestFunctionUnderscorecall :: Maybe CreateChatCompletionRequestFunctionCall -- ^ 
  , createChatCompletionRequestFunctions :: Maybe [ChatCompletionFunctions] -- ^ Deprecated in favor of `tools`.  A list of functions the model may generate JSON inputs for. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequest")
instance ToJSON CreateChatCompletionRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequest")


-- | Deprecated in favor of &#x60;tool_choice&#x60;.  Controls which (if any) function is called by the model. &#x60;none&#x60; means the model will not call a function and instead generates a message. &#x60;auto&#x60; means the model can pick between generating a message or calling a function. Specifying a particular function via &#x60;{\&quot;name\&quot;: \&quot;my_function\&quot;}&#x60; forces the model to call that function.  &#x60;none&#x60; is the default when no functions are present. &#x60;auto&#x60; is the default if functions are present. 
data CreateChatCompletionRequestFunctionCall = CreateChatCompletionRequestFunctionCall
  { createChatCompletionRequestFunctionCallName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestFunctionCall")
instance ToJSON CreateChatCompletionRequestFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestFunctionCall")


-- | ID of the model to use. See the [model endpoint compatibility](/docs/models/model-endpoint-compatibility) table for details on which models work with the Chat API.
newtype CreateChatCompletionRequestModel = CreateChatCompletionRequestModel Text deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestModel")
instance ToJSON CreateChatCompletionRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestModel")


-- | An object specifying the format that the model must output. Compatible with [GPT-4 Turbo](/docs/models/gpt-4-and-gpt-4-turbo) and all GPT-3.5 Turbo models newer than &#x60;gpt-3.5-turbo-1106&#x60;.  Setting to &#x60;{ \&quot;type\&quot;: \&quot;json_object\&quot; }&#x60; enables JSON mode, which guarantees the message the model generates is valid JSON.  **Important:** when using JSON mode, you **must** also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly \&quot;stuck\&quot; request. Also note that the message content may be partially cut off if &#x60;finish_reason&#x3D;\&quot;length\&quot;&#x60;, which indicates the generation exceeded &#x60;max_tokens&#x60; or the conversation exceeded the max context length. 
data CreateChatCompletionRequestResponseFormat = CreateChatCompletionRequestResponseFormat
  { createChatCompletionRequestResponseFormatType :: Maybe Text -- ^ Must be one of `text` or `json_object`.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestResponseFormat where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestResponseFormat")
instance ToJSON CreateChatCompletionRequestResponseFormat where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestResponseFormat")


-- | Up to 4 sequences where the API will stop generating further tokens. 
data CreateChatCompletionRequestStop = CreateChatCompletionRequestStop
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestStop where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestStop")
instance ToJSON CreateChatCompletionRequestStop where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestStop")


-- | Represents a chat completion response returned by model, based on the provided input.
data CreateChatCompletionResponse = CreateChatCompletionResponse
  { createChatCompletionResponseId :: Text -- ^ A unique identifier for the chat completion.
  , createChatCompletionResponseChoices :: [CreateChatCompletionResponseChoicesInner] -- ^ A list of chat completion choices. Can be more than one if `n` is greater than 1.
  , createChatCompletionResponseCreated :: Int -- ^ The Unix timestamp (in seconds) of when the chat completion was created.
  , createChatCompletionResponseModel :: Text -- ^ The model used for the chat completion.
  , createChatCompletionResponseSystemUnderscorefingerprint :: Maybe Text -- ^ This fingerprint represents the backend configuration that the model runs with.  Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism. 
  , createChatCompletionResponseObject :: Text -- ^ The object type, which is always `chat.completion`.
  , createChatCompletionResponseUsage :: Maybe CompletionUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionResponse")
instance ToJSON CreateChatCompletionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionResponse")


-- | 
data CreateChatCompletionResponseChoicesInner = CreateChatCompletionResponseChoicesInner
  { createChatCompletionResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function. 
  , createChatCompletionResponseChoicesInnerIndex :: Int -- ^ The index of the choice in the list of choices.
  , createChatCompletionResponseChoicesInnerMessage :: ChatCompletionResponseMessage -- ^ 
  , createChatCompletionResponseChoicesInnerLogprobs :: Maybe CreateChatCompletionResponseChoicesInnerLogprobs -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionResponseChoicesInner")
instance ToJSON CreateChatCompletionResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionResponseChoicesInner")


-- | Log probability information for the choice.
data CreateChatCompletionResponseChoicesInnerLogprobs = CreateChatCompletionResponseChoicesInnerLogprobs
  { createChatCompletionResponseChoicesInnerLogprobsContent :: [ChatCompletionTokenLogprob] -- ^ A list of message content tokens with log probability information.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionResponseChoicesInnerLogprobs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionResponseChoicesInnerLogprobs")
instance ToJSON CreateChatCompletionResponseChoicesInnerLogprobs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionResponseChoicesInnerLogprobs")


-- | Represents a streamed chunk of a chat completion response returned by model, based on the provided input.
data CreateChatCompletionStreamResponse = CreateChatCompletionStreamResponse
  { createChatCompletionStreamResponseId :: Text -- ^ A unique identifier for the chat completion. Each chunk has the same ID.
  , createChatCompletionStreamResponseChoices :: [CreateChatCompletionStreamResponseChoicesInner] -- ^ A list of chat completion choices. Can be more than one if `n` is greater than 1.
  , createChatCompletionStreamResponseCreated :: Int -- ^ The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same timestamp.
  , createChatCompletionStreamResponseModel :: Text -- ^ The model to generate the completion.
  , createChatCompletionStreamResponseSystemUnderscorefingerprint :: Maybe Text -- ^ This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism. 
  , createChatCompletionStreamResponseObject :: Text -- ^ The object type, which is always `chat.completion.chunk`.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionStreamResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionStreamResponse")
instance ToJSON CreateChatCompletionStreamResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionStreamResponse")


-- | 
data CreateChatCompletionStreamResponseChoicesInner = CreateChatCompletionStreamResponseChoicesInner
  { createChatCompletionStreamResponseChoicesInnerDelta :: ChatCompletionStreamResponseDelta -- ^ 
  , createChatCompletionStreamResponseChoicesInnerLogprobs :: Maybe CreateChatCompletionResponseChoicesInnerLogprobs -- ^ 
  , createChatCompletionStreamResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, `tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function. 
  , createChatCompletionStreamResponseChoicesInnerIndex :: Int -- ^ The index of the choice in the list of choices.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionStreamResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionStreamResponseChoicesInner")
instance ToJSON CreateChatCompletionStreamResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionStreamResponseChoicesInner")


-- | 
data CreateCompletionRequest = CreateCompletionRequest
  { createCompletionRequestModel :: CreateCompletionRequestModel -- ^ 
  , createCompletionRequestPrompt :: CreateCompletionRequestPrompt -- ^ 
  , createCompletionRequestBestUnderscoreof :: Maybe Int -- ^ Generates `best_of` completions server-side and returns the \"best\" (the one with the highest log probability per token). Results cannot be streamed.  When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. 
  , createCompletionRequestEcho :: Maybe Bool -- ^ Echo back the prompt in addition to the completion 
  , createCompletionRequestFrequencyUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/guides/text-generation/parameter-details) 
  , createCompletionRequestLogitUnderscorebias :: Maybe (Map.Map String Int) -- ^ Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view=bpe) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass `{\"50256\": -100}` to prevent the <|endoftext|> token from being generated. 
  , createCompletionRequestLogprobs :: Maybe Int -- ^ Include the log probabilities on the `logprobs` most likely output tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.  The maximum value for `logprobs` is 5. 
  , createCompletionRequestMaxUnderscoretokens :: Maybe Int -- ^ The maximum number of [tokens](/tokenizer) that can be generated in the completion.  The token count of your prompt plus `max_tokens` cannot exceed the model's context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens. 
  , createCompletionRequestN :: Maybe Int -- ^ How many completions to generate for each prompt.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. 
  , createCompletionRequestPresenceUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/guides/text-generation/parameter-details) 
  , createCompletionRequestSeed :: Maybe Int -- ^ If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result.  Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend. 
  , createCompletionRequestStop :: Maybe CreateCompletionRequestStop -- ^ 
  , createCompletionRequestStream :: Maybe Bool -- ^ Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions). 
  , createCompletionRequestSuffix :: Maybe Text -- ^ The suffix that comes after a completion of inserted text.
  , createCompletionRequestTemperature :: Maybe Double -- ^ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. 
  , createCompletionRequestTopUnderscorep :: Maybe Double -- ^ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. 
  , createCompletionRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionRequest")
instance ToJSON CreateCompletionRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionRequest")


-- | ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them. 
data CreateCompletionRequestModel = CreateCompletionRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionRequestModel")
instance ToJSON CreateCompletionRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionRequestModel")


-- | The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.  Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document. 
data CreateCompletionRequestPrompt = CreateCompletionRequestPrompt
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionRequestPrompt where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionRequestPrompt")
instance ToJSON CreateCompletionRequestPrompt where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionRequestPrompt")


-- | Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence. 
data CreateCompletionRequestStop = CreateCompletionRequestStop
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionRequestStop where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionRequestStop")
instance ToJSON CreateCompletionRequestStop where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionRequestStop")


-- | Represents a completion response from the API. Note: both the streamed and non-streamed response objects share the same shape (unlike the chat endpoint). 
data CreateCompletionResponse = CreateCompletionResponse
  { createCompletionResponseId :: Text -- ^ A unique identifier for the completion.
  , createCompletionResponseChoices :: [CreateCompletionResponseChoicesInner] -- ^ The list of completion choices the model generated for the input prompt.
  , createCompletionResponseCreated :: Int -- ^ The Unix timestamp (in seconds) of when the completion was created.
  , createCompletionResponseModel :: Text -- ^ The model used for completion.
  , createCompletionResponseSystemUnderscorefingerprint :: Maybe Text -- ^ This fingerprint represents the backend configuration that the model runs with.  Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism. 
  , createCompletionResponseObject :: Text -- ^ The object type, which is always \"text_completion\"
  , createCompletionResponseUsage :: Maybe CompletionUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponse")
instance ToJSON CreateCompletionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponse")


-- | 
data CreateCompletionResponseChoicesInner = CreateCompletionResponseChoicesInner
  { createCompletionResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, or `content_filter` if content was omitted due to a flag from our content filters. 
  , createCompletionResponseChoicesInnerIndex :: Int -- ^ 
  , createCompletionResponseChoicesInnerLogprobs :: Maybe CreateCompletionResponseChoicesInnerLogprobs -- ^ 
  , createCompletionResponseChoicesInnerText :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponseChoicesInner")
instance ToJSON CreateCompletionResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponseChoicesInner")


-- | 
data CreateCompletionResponseChoicesInnerLogprobs = CreateCompletionResponseChoicesInnerLogprobs
  { createCompletionResponseChoicesInnerLogprobsTextUnderscoreoffset :: Maybe [Int] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTokenUnderscorelogprobs :: Maybe [Double] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTokens :: Maybe [Text] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTopUnderscorelogprobs :: Maybe [(Map.Map String Double)] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponseChoicesInnerLogprobs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponseChoicesInnerLogprobs")
instance ToJSON CreateCompletionResponseChoicesInnerLogprobs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponseChoicesInnerLogprobs")


-- | 
data CreateEmbeddingRequest = CreateEmbeddingRequest
  { createEmbeddingRequestInput :: CreateEmbeddingRequestInput -- ^ 
  , createEmbeddingRequestModel :: CreateEmbeddingRequestModel -- ^ 
  , createEmbeddingRequestEncodingUnderscoreformat :: Maybe Text -- ^ The format to return the embeddings in. Can be either `float` or [`base64`](https://pypi.org/project/pybase64/).
  , createEmbeddingRequestDimensions :: Maybe Int -- ^ The number of dimensions the resulting output embeddings should have. Only supported in `text-embedding-3` and later models. 
  , createEmbeddingRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingRequest")
instance ToJSON CreateEmbeddingRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingRequest")


-- | Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for &#x60;text-embedding-ada-002&#x60;), cannot be an empty string, and any array must be 2048 dimensions or less. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens. 
data CreateEmbeddingRequestInput = CreateEmbeddingRequestInput
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingRequestInput where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingRequestInput")
instance ToJSON CreateEmbeddingRequestInput where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingRequestInput")


-- | ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them. 
data CreateEmbeddingRequestModel = CreateEmbeddingRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingRequestModel")
instance ToJSON CreateEmbeddingRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingRequestModel")


-- | 
data CreateEmbeddingResponse = CreateEmbeddingResponse
  { createEmbeddingResponseData :: [Embedding] -- ^ The list of embeddings generated by the model.
  , createEmbeddingResponseModel :: Text -- ^ The name of the model used to generate the embedding.
  , createEmbeddingResponseObject :: Text -- ^ The object type, which is always \"list\".
  , createEmbeddingResponseUsage :: CreateEmbeddingResponseUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingResponse")
instance ToJSON CreateEmbeddingResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingResponse")


-- | The usage information for the request.
data CreateEmbeddingResponseUsage = CreateEmbeddingResponseUsage
  { createEmbeddingResponseUsagePromptUnderscoretokens :: Int -- ^ The number of tokens used by the prompt.
  , createEmbeddingResponseUsageTotalUnderscoretokens :: Int -- ^ The total number of tokens used by the request.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingResponseUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingResponseUsage")
instance ToJSON CreateEmbeddingResponseUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingResponseUsage")


-- | 
data CreateFineTuningJobRequest = CreateFineTuningJobRequest
  { createFineTuningJobRequestModel :: CreateFineTuningJobRequestModel -- ^ 
  , createFineTuningJobRequestTrainingUnderscorefile :: Text -- ^ The ID of an uploaded file that contains training data.  See [upload file](/docs/api-reference/files/upload) for how to upload a file.  Your dataset must be formatted as a JSONL file. Additionally, you must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/fine-tuning) for more details. 
  , createFineTuningJobRequestHyperparameters :: Maybe CreateFineTuningJobRequestHyperparameters -- ^ 
  , createFineTuningJobRequestSuffix :: Maybe Text -- ^ A string of up to 18 characters that will be added to your fine-tuned model name.  For example, a `suffix` of \"custom-model-name\" would produce a model name like `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`. 
  , createFineTuningJobRequestValidationUnderscorefile :: Maybe Text -- ^ The ID of an uploaded file that contains validation data.  If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the fine-tuning results file. The same data should not be present in both train and validation files.  Your dataset must be formatted as a JSONL file. You must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/fine-tuning) for more details. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequest")
instance ToJSON CreateFineTuningJobRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequest")


-- | The hyperparameters used for the fine-tuning job.
data CreateFineTuningJobRequestHyperparameters = CreateFineTuningJobRequestHyperparameters
  { createFineTuningJobRequestHyperparametersBatchUnderscoresize :: Maybe CreateFineTuningJobRequestHyperparametersBatchSize -- ^ 
  , createFineTuningJobRequestHyperparametersLearningUnderscorerateUnderscoremultiplier :: Maybe CreateFineTuningJobRequestHyperparametersLearningRateMultiplier -- ^ 
  , createFineTuningJobRequestHyperparametersNUnderscoreepochs :: Maybe CreateFineTuningJobRequestHyperparametersNEpochs -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequestHyperparameters where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequestHyperparameters")
instance ToJSON CreateFineTuningJobRequestHyperparameters where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequestHyperparameters")


-- | Number of examples in each batch. A larger batch size means that model parameters are updated less frequently, but with lower variance. 
data CreateFineTuningJobRequestHyperparametersBatchSize = CreateFineTuningJobRequestHyperparametersBatchSize
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequestHyperparametersBatchSize where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequestHyperparametersBatchSize")
instance ToJSON CreateFineTuningJobRequestHyperparametersBatchSize where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequestHyperparametersBatchSize")


-- | Scaling factor for the learning rate. A smaller learning rate may be useful to avoid overfitting. 
data CreateFineTuningJobRequestHyperparametersLearningRateMultiplier = CreateFineTuningJobRequestHyperparametersLearningRateMultiplier
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequestHyperparametersLearningRateMultiplier where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequestHyperparametersLearningRateMultiplier")
instance ToJSON CreateFineTuningJobRequestHyperparametersLearningRateMultiplier where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequestHyperparametersLearningRateMultiplier")


-- | The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset. 
data CreateFineTuningJobRequestHyperparametersNEpochs = CreateFineTuningJobRequestHyperparametersNEpochs
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequestHyperparametersNEpochs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequestHyperparametersNEpochs")
instance ToJSON CreateFineTuningJobRequestHyperparametersNEpochs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequestHyperparametersNEpochs")


-- | The name of the model to fine-tune. You can select one of the [supported models](/docs/guides/fine-tuning/what-models-can-be-fine-tuned). 
data CreateFineTuningJobRequestModel = CreateFineTuningJobRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuningJobRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuningJobRequestModel")
instance ToJSON CreateFineTuningJobRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuningJobRequestModel")


-- | 
data CreateImageRequest = CreateImageRequest
  { createImageRequestPrompt :: Text -- ^ A text description of the desired image(s). The maximum length is 1000 characters for `dall-e-2` and 4000 characters for `dall-e-3`.
  , createImageRequestModel :: Maybe CreateImageRequestModel -- ^ 
  , createImageRequestN :: Maybe Int -- ^ The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.
  , createImageRequestQuality :: Maybe Text -- ^ The quality of the image that will be generated. `hd` creates images with finer details and greater consistency across the image. This param is only supported for `dall-e-3`.
  , createImageRequestResponseUnderscoreformat :: Maybe Text -- ^ The format in which the generated images are returned. Must be one of `url` or `b64_json`. URLs are only valid for 60 minutes after the image has been generated.
  , createImageRequestSize :: Maybe Text -- ^ The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models.
  , createImageRequestStyle :: Maybe Text -- ^ The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`.
  , createImageRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateImageRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createImageRequest")
instance ToJSON CreateImageRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createImageRequest")


-- | The model to use for image generation.
data CreateImageRequestModel = CreateImageRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateImageRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createImageRequestModel")
instance ToJSON CreateImageRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createImageRequestModel")


-- | 
data CreateMessageRequest = CreateMessageRequest
  { createMessageRequestRole :: Text -- ^ The role of the entity that is creating the message. Currently only `user` is supported.
  , createMessageRequestContent :: Text -- ^ The content of the message.
  , createMessageRequestFileUnderscoreids :: Maybe [Text] -- ^ A list of [File](/docs/api-reference/files) IDs that the message should use. There can be a maximum of 10 files attached to a message. Useful for tools like `retrieval` and `code_interpreter` that can access and use files.
  , createMessageRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateMessageRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createMessageRequest")
instance ToJSON CreateMessageRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createMessageRequest")


-- | 
data CreateModerationRequest = CreateModerationRequest
  { createModerationRequestInput :: CreateModerationRequestInput -- ^ 
  , createModerationRequestModel :: Maybe CreateModerationRequestModel -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationRequest")
instance ToJSON CreateModerationRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationRequest")


-- | The input text to classify
data CreateModerationRequestInput = CreateModerationRequestInput
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationRequestInput where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationRequestInput")
instance ToJSON CreateModerationRequestInput where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationRequestInput")


-- | Two content moderations models are available: &#x60;text-moderation-stable&#x60; and &#x60;text-moderation-latest&#x60;.  The default is &#x60;text-moderation-latest&#x60; which will be automatically upgraded over time. This ensures you are always using our most accurate model. If you use &#x60;text-moderation-stable&#x60;, we will provide advanced notice before updating the model. Accuracy of &#x60;text-moderation-stable&#x60; may be slightly lower than for &#x60;text-moderation-latest&#x60;. 
data CreateModerationRequestModel = CreateModerationRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationRequestModel")
instance ToJSON CreateModerationRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationRequestModel")


-- | Represents if a given text input is potentially harmful.
data CreateModerationResponse = CreateModerationResponse
  { createModerationResponseId :: Text -- ^ The unique identifier for the moderation request.
  , createModerationResponseModel :: Text -- ^ The model used to generate the moderation results.
  , createModerationResponseResults :: [CreateModerationResponseResultsInner] -- ^ A list of moderation objects.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponse")
instance ToJSON CreateModerationResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponse")


-- | 
data CreateModerationResponseResultsInner = CreateModerationResponseResultsInner
  { createModerationResponseResultsInnerFlagged :: Bool -- ^ Whether any of the below categories are flagged.
  , createModerationResponseResultsInnerCategories :: CreateModerationResponseResultsInnerCategories -- ^ 
  , createModerationResponseResultsInnerCategoryUnderscorescores :: CreateModerationResponseResultsInnerCategoryScores -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInner")
instance ToJSON CreateModerationResponseResultsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInner")


-- | A list of the categories, and whether they are flagged or not.
data CreateModerationResponseResultsInnerCategories = CreateModerationResponseResultsInnerCategories
  { createModerationResponseResultsInnerCategoriesHate :: Bool -- ^ Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harassment.
  , createModerationResponseResultsInnerCategoriesHateSlashthreatening :: Bool -- ^ Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.
  , createModerationResponseResultsInnerCategoriesHarassment :: Bool -- ^ Content that expresses, incites, or promotes harassing language towards any target.
  , createModerationResponseResultsInnerCategoriesHarassmentSlashthreatening :: Bool -- ^ Harassment content that also includes violence or serious harm towards any target.
  , createModerationResponseResultsInnerCategoriesSelfDashharm :: Bool -- ^ Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
  , createModerationResponseResultsInnerCategoriesSelfDashharmSlashintent :: Bool -- ^ Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.
  , createModerationResponseResultsInnerCategoriesSelfDashharmSlashinstructions :: Bool -- ^ Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.
  , createModerationResponseResultsInnerCategoriesSexual :: Bool -- ^ Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).
  , createModerationResponseResultsInnerCategoriesSexualSlashminors :: Bool -- ^ Sexual content that includes an individual who is under 18 years old.
  , createModerationResponseResultsInnerCategoriesViolence :: Bool -- ^ Content that depicts death, violence, or physical injury.
  , createModerationResponseResultsInnerCategoriesViolenceSlashgraphic :: Bool -- ^ Content that depicts death, violence, or physical injury in graphic detail.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInnerCategories where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInnerCategories")
instance ToJSON CreateModerationResponseResultsInnerCategories where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInnerCategories")


-- | A list of the categories along with their scores as predicted by model.
data CreateModerationResponseResultsInnerCategoryScores = CreateModerationResponseResultsInnerCategoryScores
  { createModerationResponseResultsInnerCategoryScoresHate :: Double -- ^ The score for the category 'hate'.
  , createModerationResponseResultsInnerCategoryScoresHateSlashthreatening :: Double -- ^ The score for the category 'hate/threatening'.
  , createModerationResponseResultsInnerCategoryScoresHarassment :: Double -- ^ The score for the category 'harassment'.
  , createModerationResponseResultsInnerCategoryScoresHarassmentSlashthreatening :: Double -- ^ The score for the category 'harassment/threatening'.
  , createModerationResponseResultsInnerCategoryScoresSelfDashharm :: Double -- ^ The score for the category 'self-harm'.
  , createModerationResponseResultsInnerCategoryScoresSelfDashharmSlashintent :: Double -- ^ The score for the category 'self-harm/intent'.
  , createModerationResponseResultsInnerCategoryScoresSelfDashharmSlashinstructions :: Double -- ^ The score for the category 'self-harm/instructions'.
  , createModerationResponseResultsInnerCategoryScoresSexual :: Double -- ^ The score for the category 'sexual'.
  , createModerationResponseResultsInnerCategoryScoresSexualSlashminors :: Double -- ^ The score for the category 'sexual/minors'.
  , createModerationResponseResultsInnerCategoryScoresViolence :: Double -- ^ The score for the category 'violence'.
  , createModerationResponseResultsInnerCategoryScoresViolenceSlashgraphic :: Double -- ^ The score for the category 'violence/graphic'.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInnerCategoryScores where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInnerCategoryScores")
instance ToJSON CreateModerationResponseResultsInnerCategoryScores where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInnerCategoryScores")


-- | 
data CreateRunRequest = CreateRunRequest
  { createRunRequestAssistantUnderscoreid :: Text -- ^ The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.
  , createRunRequestModel :: Maybe Text -- ^ The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
  , createRunRequestInstructions :: Maybe Text -- ^ Overrides the [instructions](/docs/api-reference/assistants/createAssistant) of the assistant. This is useful for modifying the behavior on a per-run basis.
  , createRunRequestAdditionalUnderscoreinstructions :: Maybe Text -- ^ Appends additional instructions at the end of the instructions for the run. This is useful for modifying the behavior on a per-run basis without overriding other instructions.
  , createRunRequestTools :: Maybe [AssistantObjectToolsInner] -- ^ Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
  , createRunRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateRunRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createRunRequest")
instance ToJSON CreateRunRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createRunRequest")


-- | 
data CreateSpeechRequest = CreateSpeechRequest
  { createSpeechRequestModel :: CreateSpeechRequestModel -- ^ 
  , createSpeechRequestInput :: Text -- ^ The text to generate audio for. The maximum length is 4096 characters.
  , createSpeechRequestVoice :: Text -- ^ The voice to use when generating the audio. Supported voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`. Previews of the voices are available in the [Text to speech guide](/docs/guides/text-to-speech/voice-options).
  , createSpeechRequestResponseUnderscoreformat :: Maybe Text -- ^ The format to audio in. Supported formats are `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`.
  , createSpeechRequestSpeed :: Maybe Double -- ^ The speed of the generated audio. Select a value from `0.25` to `4.0`. `1.0` is the default.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateSpeechRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createSpeechRequest")
instance ToJSON CreateSpeechRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createSpeechRequest")


-- | One of the available [TTS models](/docs/models/tts): &#x60;tts-1&#x60; or &#x60;tts-1-hd&#x60; 
data CreateSpeechRequestModel = CreateSpeechRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateSpeechRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createSpeechRequestModel")
instance ToJSON CreateSpeechRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createSpeechRequestModel")


-- | 
data CreateThreadAndRunRequest = CreateThreadAndRunRequest
  { createThreadAndRunRequestAssistantUnderscoreid :: Text -- ^ The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.
  , createThreadAndRunRequestThread :: Maybe CreateThreadRequest -- ^ 
  , createThreadAndRunRequestModel :: Maybe Text -- ^ The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
  , createThreadAndRunRequestInstructions :: Maybe Text -- ^ Override the default system message of the assistant. This is useful for modifying the behavior on a per-run basis.
  , createThreadAndRunRequestTools :: Maybe [CreateThreadAndRunRequestToolsInner] -- ^ Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
  , createThreadAndRunRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateThreadAndRunRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createThreadAndRunRequest")
instance ToJSON CreateThreadAndRunRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createThreadAndRunRequest")


-- | 
data CreateThreadAndRunRequestToolsInner = CreateThreadAndRunRequestToolsInner
  { createThreadAndRunRequestToolsInnerType :: Text -- ^ The type of tool being defined: `function`
  , createThreadAndRunRequestToolsInnerFunction :: FunctionObject -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateThreadAndRunRequestToolsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createThreadAndRunRequestToolsInner")
instance ToJSON CreateThreadAndRunRequestToolsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createThreadAndRunRequestToolsInner")


-- | 
data CreateThreadRequest = CreateThreadRequest
  { createThreadRequestMessages :: Maybe [CreateMessageRequest] -- ^ A list of [messages](/docs/api-reference/messages) to start the thread with.
  , createThreadRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateThreadRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createThreadRequest")
instance ToJSON CreateThreadRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createThreadRequest")


-- | 
data CreateTranscription200Response = CreateTranscription200Response
  { createTranscription200ResponseText :: Text -- ^ The transcribed text.
  , createTranscription200ResponseLanguage :: Text -- ^ The language of the input audio.
  , createTranscription200ResponseDuration :: Text -- ^ The duration of the input audio.
  , createTranscription200ResponseWords :: Maybe [TranscriptionWord] -- ^ Extracted words and their corresponding timestamps.
  , createTranscription200ResponseSegments :: Maybe [TranscriptionSegment] -- ^ Segments of the transcribed text and their corresponding details.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranscription200Response where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranscription200Response")
instance ToJSON CreateTranscription200Response where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranscription200Response")


-- | Represents a transcription response returned by model, based on the provided input.
data CreateTranscriptionResponseJson = CreateTranscriptionResponseJson
  { createTranscriptionResponseJsonText :: Text -- ^ The transcribed text.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranscriptionResponseJson where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranscriptionResponseJson")
instance ToJSON CreateTranscriptionResponseJson where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranscriptionResponseJson")


-- | Represents a verbose json transcription response returned by model, based on the provided input.
data CreateTranscriptionResponseVerboseJson = CreateTranscriptionResponseVerboseJson
  { createTranscriptionResponseVerboseJsonLanguage :: Text -- ^ The language of the input audio.
  , createTranscriptionResponseVerboseJsonDuration :: Text -- ^ The duration of the input audio.
  , createTranscriptionResponseVerboseJsonText :: Text -- ^ The transcribed text.
  , createTranscriptionResponseVerboseJsonWords :: Maybe [TranscriptionWord] -- ^ Extracted words and their corresponding timestamps.
  , createTranscriptionResponseVerboseJsonSegments :: Maybe [TranscriptionSegment] -- ^ Segments of the transcribed text and their corresponding details.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranscriptionResponseVerboseJson where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranscriptionResponseVerboseJson")
instance ToJSON CreateTranscriptionResponseVerboseJson where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranscriptionResponseVerboseJson")


-- | 
data CreateTranslation200Response = CreateTranslation200Response
  { createTranslation200ResponseText :: Text -- ^ The translated text.
  , createTranslation200ResponseLanguage :: Text -- ^ The language of the output translation (always `english`).
  , createTranslation200ResponseDuration :: Text -- ^ The duration of the input audio.
  , createTranslation200ResponseSegments :: Maybe [TranscriptionSegment] -- ^ Segments of the translated text and their corresponding details.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranslation200Response where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranslation200Response")
instance ToJSON CreateTranslation200Response where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranslation200Response")


-- | 
data CreateTranslationResponseJson = CreateTranslationResponseJson
  { createTranslationResponseJsonText :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranslationResponseJson where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranslationResponseJson")
instance ToJSON CreateTranslationResponseJson where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranslationResponseJson")


-- | 
data CreateTranslationResponseVerboseJson = CreateTranslationResponseVerboseJson
  { createTranslationResponseVerboseJsonLanguage :: Text -- ^ The language of the output translation (always `english`).
  , createTranslationResponseVerboseJsonDuration :: Text -- ^ The duration of the input audio.
  , createTranslationResponseVerboseJsonText :: Text -- ^ The translated text.
  , createTranslationResponseVerboseJsonSegments :: Maybe [TranscriptionSegment] -- ^ Segments of the translated text and their corresponding details.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranslationResponseVerboseJson where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranslationResponseVerboseJson")
instance ToJSON CreateTranslationResponseVerboseJson where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranslationResponseVerboseJson")


-- | Deletes the association between the assistant and the file, but does not delete the [File](/docs/api-reference/files) object itself.
data DeleteAssistantFileResponse = DeleteAssistantFileResponse
  { deleteAssistantFileResponseId :: Text -- ^ 
  , deleteAssistantFileResponseDeleted :: Bool -- ^ 
  , deleteAssistantFileResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteAssistantFileResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteAssistantFileResponse")
instance ToJSON DeleteAssistantFileResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteAssistantFileResponse")


-- | 
data DeleteAssistantResponse = DeleteAssistantResponse
  { deleteAssistantResponseId :: Text -- ^ 
  , deleteAssistantResponseDeleted :: Bool -- ^ 
  , deleteAssistantResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteAssistantResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteAssistantResponse")
instance ToJSON DeleteAssistantResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteAssistantResponse")


-- | 
data DeleteFileResponse = DeleteFileResponse
  { deleteFileResponseId :: Text -- ^ 
  , deleteFileResponseObject :: Text -- ^ 
  , deleteFileResponseDeleted :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteFileResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteFileResponse")
instance ToJSON DeleteFileResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteFileResponse")


-- | 
data DeleteMessageResponse = DeleteMessageResponse
  { deleteMessageResponseId :: Text -- ^ 
  , deleteMessageResponseDeleted :: Bool -- ^ 
  , deleteMessageResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteMessageResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteMessageResponse")
instance ToJSON DeleteMessageResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteMessageResponse")


-- | 
data DeleteModelResponse = DeleteModelResponse
  { deleteModelResponseId :: Text -- ^ 
  , deleteModelResponseDeleted :: Bool -- ^ 
  , deleteModelResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteModelResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteModelResponse")
instance ToJSON DeleteModelResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteModelResponse")


-- | 
data DeleteThreadResponse = DeleteThreadResponse
  { deleteThreadResponseId :: Text -- ^ 
  , deleteThreadResponseDeleted :: Bool -- ^ 
  , deleteThreadResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteThreadResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteThreadResponse")
instance ToJSON DeleteThreadResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteThreadResponse")


-- | Represents an embedding vector returned by embedding endpoint. 
data Embedding = Embedding
  { embeddingIndex :: Int -- ^ The index of the embedding in the list of embeddings.
  , embeddingEmbedding :: [Double] -- ^ The embedding vector, which is a list of floats. The length of vector depends on the model as listed in the [embedding guide](/docs/guides/embeddings). 
  , embeddingObject :: Text -- ^ The object type, which is always \"embedding\".
  } deriving (Show, Eq, Generic, Data)

instance FromJSON Embedding where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "embedding")
instance ToJSON Embedding where
  toJSON = genericToJSON (removeFieldLabelPrefix False "embedding")


-- | 
data Error = Error
  { errorCode :: Text -- ^ 
  , errorMessage :: Text -- ^ 
  , errorParam :: Text -- ^ 
  , errorType :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON Error where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "error")
instance ToJSON Error where
  toJSON = genericToJSON (removeFieldLabelPrefix False "error")


-- | 
data ErrorResponse = ErrorResponse
  { errorResponseError :: Error -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ErrorResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "errorResponse")
instance ToJSON ErrorResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "errorResponse")


-- | The &#x60;fine_tuning.job&#x60; object represents a fine-tuning job that has been created through the API. 
data FineTuningJob = FineTuningJob
  { fineTuningJobId :: Text -- ^ The object identifier, which can be referenced in the API endpoints.
  , fineTuningJobCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the fine-tuning job was created.
  , fineTuningJobError :: FineTuningJobError -- ^ 
  , fineTuningJobFineUnderscoretunedUnderscoremodel :: Text -- ^ The name of the fine-tuned model that is being created. The value will be null if the fine-tuning job is still running.
  , fineTuningJobFinishedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the fine-tuning job was finished. The value will be null if the fine-tuning job is still running.
  , fineTuningJobHyperparameters :: FineTuningJobHyperparameters -- ^ 
  , fineTuningJobModel :: Text -- ^ The base model that is being fine-tuned.
  , fineTuningJobObject :: Text -- ^ The object type, which is always \"fine_tuning.job\".
  , fineTuningJobOrganizationUnderscoreid :: Text -- ^ The organization that owns the fine-tuning job.
  , fineTuningJobResultUnderscorefiles :: [Text] -- ^ The compiled results file ID(s) for the fine-tuning job. You can retrieve the results with the [Files API](/docs/api-reference/files/retrieve-contents).
  , fineTuningJobStatus :: Text -- ^ The current status of the fine-tuning job, which can be either `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
  , fineTuningJobTrainedUnderscoretokens :: Int -- ^ The total number of billable tokens processed by this fine-tuning job. The value will be null if the fine-tuning job is still running.
  , fineTuningJobTrainingUnderscorefile :: Text -- ^ The file ID used for training. You can retrieve the training data with the [Files API](/docs/api-reference/files/retrieve-contents).
  , fineTuningJobValidationUnderscorefile :: Text -- ^ The file ID used for validation. You can retrieve the validation results with the [Files API](/docs/api-reference/files/retrieve-contents).
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuningJob where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuningJob")
instance ToJSON FineTuningJob where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuningJob")


-- | For fine-tuning jobs that have &#x60;failed&#x60;, this will contain more information on the cause of the failure.
data FineTuningJobError = FineTuningJobError
  { fineTuningJobErrorCode :: Text -- ^ A machine-readable error code.
  , fineTuningJobErrorMessage :: Text -- ^ A human-readable error message.
  , fineTuningJobErrorParam :: Text -- ^ The parameter that was invalid, usually `training_file` or `validation_file`. This field will be null if the failure was not parameter-specific.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuningJobError where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuningJobError")
instance ToJSON FineTuningJobError where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuningJobError")


-- | Fine-tuning job event object
data FineTuningJobEvent = FineTuningJobEvent
  { fineTuningJobEventId :: Text -- ^ 
  , fineTuningJobEventCreatedUnderscoreat :: Int -- ^ 
  , fineTuningJobEventLevel :: Text -- ^ 
  , fineTuningJobEventMessage :: Text -- ^ 
  , fineTuningJobEventObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuningJobEvent where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuningJobEvent")
instance ToJSON FineTuningJobEvent where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuningJobEvent")


-- | The hyperparameters used for the fine-tuning job. See the [fine-tuning guide](/docs/guides/fine-tuning) for more details.
data FineTuningJobHyperparameters = FineTuningJobHyperparameters
  { fineTuningJobHyperparametersNUnderscoreepochs :: FineTuningJobHyperparametersNEpochs -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuningJobHyperparameters where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuningJobHyperparameters")
instance ToJSON FineTuningJobHyperparameters where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuningJobHyperparameters")


-- | The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset. \&quot;auto\&quot; decides the optimal number of epochs based on the size of the dataset. If setting the number manually, we support any number between 1 and 50 epochs.
data FineTuningJobHyperparametersNEpochs = FineTuningJobHyperparametersNEpochs
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuningJobHyperparametersNEpochs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuningJobHyperparametersNEpochs")
instance ToJSON FineTuningJobHyperparametersNEpochs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuningJobHyperparametersNEpochs")


-- | 
data FunctionObject = FunctionObject
  { functionObjectDescription :: Maybe Text -- ^ A description of what the function does, used by the model to choose when and how to call the function.
  , functionObjectName :: Text -- ^ The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
  , functionObjectParameters :: Maybe (Map.Map String Value) -- ^ The parameters the functions accepts, described as a JSON Schema object. See the [guide](/docs/guides/text-generation/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.   Omitting `parameters` defines a function with an empty parameter list.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FunctionObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "functionObject")
instance ToJSON FunctionObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "functionObject")


-- | Represents the url or the content of an image generated by the OpenAI API.
data Image = Image
  { imageB64Underscorejson :: Maybe Text -- ^ The base64-encoded JSON of the generated image, if `response_format` is `b64_json`.
  , imageUrl :: Maybe Text -- ^ The URL of the generated image, if `response_format` is `url` (default).
  , imageRevisedUnderscoreprompt :: Maybe Text -- ^ The prompt that was used to generate the image, if there was any revision to the prompt.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON Image where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "image")
instance ToJSON Image where
  toJSON = genericToJSON (removeFieldLabelPrefix False "image")


-- | 
data ImagesResponse = ImagesResponse
  { imagesResponseCreated :: Int -- ^ 
  , imagesResponseData :: [Image] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ImagesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "imagesResponse")
instance ToJSON ImagesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "imagesResponse")


-- | 
data ListAssistantFilesResponse = ListAssistantFilesResponse
  { listAssistantFilesResponseObject :: Text -- ^ 
  , listAssistantFilesResponseData :: [AssistantFileObject] -- ^ 
  , listAssistantFilesResponseFirstUnderscoreid :: Text -- ^ 
  , listAssistantFilesResponseLastUnderscoreid :: Text -- ^ 
  , listAssistantFilesResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListAssistantFilesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listAssistantFilesResponse")
instance ToJSON ListAssistantFilesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listAssistantFilesResponse")


-- | 
data ListAssistantsResponse = ListAssistantsResponse
  { listAssistantsResponseObject :: Text -- ^ 
  , listAssistantsResponseData :: [AssistantObject] -- ^ 
  , listAssistantsResponseFirstUnderscoreid :: Text -- ^ 
  , listAssistantsResponseLastUnderscoreid :: Text -- ^ 
  , listAssistantsResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListAssistantsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listAssistantsResponse")
instance ToJSON ListAssistantsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listAssistantsResponse")


-- | 
data ListFilesResponse = ListFilesResponse
  { listFilesResponseData :: [OpenAIFile] -- ^ 
  , listFilesResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListFilesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listFilesResponse")
instance ToJSON ListFilesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listFilesResponse")


-- | 
data ListFineTuningJobEventsResponse = ListFineTuningJobEventsResponse
  { listFineTuningJobEventsResponseData :: [FineTuningJobEvent] -- ^ 
  , listFineTuningJobEventsResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListFineTuningJobEventsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listFineTuningJobEventsResponse")
instance ToJSON ListFineTuningJobEventsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listFineTuningJobEventsResponse")


-- | 
data ListMessageFilesResponse = ListMessageFilesResponse
  { listMessageFilesResponseObject :: Text -- ^ 
  , listMessageFilesResponseData :: [MessageFileObject] -- ^ 
  , listMessageFilesResponseFirstUnderscoreid :: Text -- ^ 
  , listMessageFilesResponseLastUnderscoreid :: Text -- ^ 
  , listMessageFilesResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListMessageFilesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listMessageFilesResponse")
instance ToJSON ListMessageFilesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listMessageFilesResponse")


-- | 
data ListMessagesResponse = ListMessagesResponse
  { listMessagesResponseObject :: Text -- ^ 
  , listMessagesResponseData :: [MessageObject] -- ^ 
  , listMessagesResponseFirstUnderscoreid :: Text -- ^ 
  , listMessagesResponseLastUnderscoreid :: Text -- ^ 
  , listMessagesResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListMessagesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listMessagesResponse")
instance ToJSON ListMessagesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listMessagesResponse")


-- | 
data ListModelsResponse = ListModelsResponse
  { listModelsResponseObject :: Text -- ^ 
  , listModelsResponseData :: [Model] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListModelsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listModelsResponse")
instance ToJSON ListModelsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listModelsResponse")


-- | 
data ListPaginatedFineTuningJobsResponse = ListPaginatedFineTuningJobsResponse
  { listPaginatedFineTuningJobsResponseData :: [FineTuningJob] -- ^ 
  , listPaginatedFineTuningJobsResponseHasUnderscoremore :: Bool -- ^ 
  , listPaginatedFineTuningJobsResponseObject :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListPaginatedFineTuningJobsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listPaginatedFineTuningJobsResponse")
instance ToJSON ListPaginatedFineTuningJobsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listPaginatedFineTuningJobsResponse")


-- | 
data ListRunStepsResponse = ListRunStepsResponse
  { listRunStepsResponseObject :: Text -- ^ 
  , listRunStepsResponseData :: [RunStepObject] -- ^ 
  , listRunStepsResponseFirstUnderscoreid :: Text -- ^ 
  , listRunStepsResponseLastUnderscoreid :: Text -- ^ 
  , listRunStepsResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListRunStepsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listRunStepsResponse")
instance ToJSON ListRunStepsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listRunStepsResponse")


-- | 
data ListRunsResponse = ListRunsResponse
  { listRunsResponseObject :: Text -- ^ 
  , listRunsResponseData :: [RunObject] -- ^ 
  , listRunsResponseFirstUnderscoreid :: Text -- ^ 
  , listRunsResponseLastUnderscoreid :: Text -- ^ 
  , listRunsResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListRunsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listRunsResponse")
instance ToJSON ListRunsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listRunsResponse")


-- | 
data ListThreadsResponse = ListThreadsResponse
  { listThreadsResponseObject :: Text -- ^ 
  , listThreadsResponseData :: [ThreadObject] -- ^ 
  , listThreadsResponseFirstUnderscoreid :: Text -- ^ 
  , listThreadsResponseLastUnderscoreid :: Text -- ^ 
  , listThreadsResponseHasUnderscoremore :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListThreadsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listThreadsResponse")
instance ToJSON ListThreadsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listThreadsResponse")


-- | References an image [File](/docs/api-reference/files) in the content of a message.
data MessageContentImageFileObject = MessageContentImageFileObject
  { messageContentImageFileObjectType :: Text -- ^ Always `image_file`.
  , messageContentImageFileObjectImageUnderscorefile :: MessageContentImageFileObjectImageFile -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentImageFileObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentImageFileObject")
instance ToJSON MessageContentImageFileObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentImageFileObject")


-- | 
data MessageContentImageFileObjectImageFile = MessageContentImageFileObjectImageFile
  { messageContentImageFileObjectImageFileFileUnderscoreid :: Text -- ^ The [File](/docs/api-reference/files) ID of the image in the message content.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentImageFileObjectImageFile where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentImageFileObjectImageFile")
instance ToJSON MessageContentImageFileObjectImageFile where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentImageFileObjectImageFile")


-- | A citation within the message that points to a specific quote from a specific File associated with the assistant or the message. Generated when the assistant uses the \&quot;retrieval\&quot; tool to search files.
data MessageContentTextAnnotationsFileCitationObject = MessageContentTextAnnotationsFileCitationObject
  { messageContentTextAnnotationsFileCitationObjectType :: Text -- ^ Always `file_citation`.
  , messageContentTextAnnotationsFileCitationObjectText :: Text -- ^ The text in the message content that needs to be replaced.
  , messageContentTextAnnotationsFileCitationObjectFileUnderscorecitation :: MessageContentTextAnnotationsFileCitationObjectFileCitation -- ^ 
  , messageContentTextAnnotationsFileCitationObjectStartUnderscoreindex :: Int -- ^ 
  , messageContentTextAnnotationsFileCitationObjectEndUnderscoreindex :: Int -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextAnnotationsFileCitationObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextAnnotationsFileCitationObject")
instance ToJSON MessageContentTextAnnotationsFileCitationObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextAnnotationsFileCitationObject")


-- | 
data MessageContentTextAnnotationsFileCitationObjectFileCitation = MessageContentTextAnnotationsFileCitationObjectFileCitation
  { messageContentTextAnnotationsFileCitationObjectFileCitationFileUnderscoreid :: Text -- ^ The ID of the specific File the citation is from.
  , messageContentTextAnnotationsFileCitationObjectFileCitationQuote :: Text -- ^ The specific quote in the file.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextAnnotationsFileCitationObjectFileCitation where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextAnnotationsFileCitationObjectFileCitation")
instance ToJSON MessageContentTextAnnotationsFileCitationObjectFileCitation where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextAnnotationsFileCitationObjectFileCitation")


-- | A URL for the file that&#39;s generated when the assistant used the &#x60;code_interpreter&#x60; tool to generate a file.
data MessageContentTextAnnotationsFilePathObject = MessageContentTextAnnotationsFilePathObject
  { messageContentTextAnnotationsFilePathObjectType :: Text -- ^ Always `file_path`.
  , messageContentTextAnnotationsFilePathObjectText :: Text -- ^ The text in the message content that needs to be replaced.
  , messageContentTextAnnotationsFilePathObjectFileUnderscorepath :: MessageContentTextAnnotationsFilePathObjectFilePath -- ^ 
  , messageContentTextAnnotationsFilePathObjectStartUnderscoreindex :: Int -- ^ 
  , messageContentTextAnnotationsFilePathObjectEndUnderscoreindex :: Int -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextAnnotationsFilePathObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextAnnotationsFilePathObject")
instance ToJSON MessageContentTextAnnotationsFilePathObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextAnnotationsFilePathObject")


-- | 
data MessageContentTextAnnotationsFilePathObjectFilePath = MessageContentTextAnnotationsFilePathObjectFilePath
  { messageContentTextAnnotationsFilePathObjectFilePathFileUnderscoreid :: Text -- ^ The ID of the file that was generated.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextAnnotationsFilePathObjectFilePath where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextAnnotationsFilePathObjectFilePath")
instance ToJSON MessageContentTextAnnotationsFilePathObjectFilePath where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextAnnotationsFilePathObjectFilePath")


-- | The text content that is part of a message.
data MessageContentTextObject = MessageContentTextObject
  { messageContentTextObjectType :: Text -- ^ Always `text`.
  , messageContentTextObjectText :: MessageContentTextObjectText -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextObject")
instance ToJSON MessageContentTextObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextObject")


-- | 
data MessageContentTextObjectText = MessageContentTextObjectText
  { messageContentTextObjectTextValue :: Text -- ^ The data that makes up the text.
  , messageContentTextObjectTextAnnotations :: [MessageContentTextObjectTextAnnotationsInner] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextObjectText where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextObjectText")
instance ToJSON MessageContentTextObjectText where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextObjectText")


-- | 
data MessageContentTextObjectTextAnnotationsInner = MessageContentTextObjectTextAnnotationsInner
  { messageContentTextObjectTextAnnotationsInnerType :: Text -- ^ Always `file_path`.
  , messageContentTextObjectTextAnnotationsInnerText :: Text -- ^ The text in the message content that needs to be replaced.
  , messageContentTextObjectTextAnnotationsInnerFileUnderscorecitation :: MessageContentTextAnnotationsFileCitationObjectFileCitation -- ^ 
  , messageContentTextObjectTextAnnotationsInnerStartUnderscoreindex :: Int -- ^ 
  , messageContentTextObjectTextAnnotationsInnerEndUnderscoreindex :: Int -- ^ 
  , messageContentTextObjectTextAnnotationsInnerFileUnderscorepath :: MessageContentTextAnnotationsFilePathObjectFilePath -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageContentTextObjectTextAnnotationsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageContentTextObjectTextAnnotationsInner")
instance ToJSON MessageContentTextObjectTextAnnotationsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageContentTextObjectTextAnnotationsInner")


-- | A list of files attached to a &#x60;message&#x60;.
data MessageFileObject = MessageFileObject
  { messageFileObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , messageFileObjectObject :: Text -- ^ The object type, which is always `thread.message.file`.
  , messageFileObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the message file was created.
  , messageFileObjectMessageUnderscoreid :: Text -- ^ The ID of the [message](/docs/api-reference/messages) that the [File](/docs/api-reference/files) is attached to.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageFileObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageFileObject")
instance ToJSON MessageFileObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageFileObject")


-- | Represents a message within a [thread](/docs/api-reference/threads).
data MessageObject = MessageObject
  { messageObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , messageObjectObject :: Text -- ^ The object type, which is always `thread.message`.
  , messageObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the message was created.
  , messageObjectThreadUnderscoreid :: Text -- ^ The [thread](/docs/api-reference/threads) ID that this message belongs to.
  , messageObjectRole :: Text -- ^ The entity that produced the message. One of `user` or `assistant`.
  , messageObjectContent :: [MessageObjectContentInner] -- ^ The content of the message in array of text and/or images.
  , messageObjectAssistantUnderscoreid :: Text -- ^ If applicable, the ID of the [assistant](/docs/api-reference/assistants) that authored this message.
  , messageObjectRunUnderscoreid :: Text -- ^ If applicable, the ID of the [run](/docs/api-reference/runs) associated with the authoring of this message.
  , messageObjectFileUnderscoreids :: [Text] -- ^ A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.
  , messageObjectMetadata :: Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageObject")
instance ToJSON MessageObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageObject")


-- | 
data MessageObjectContentInner = MessageObjectContentInner
  { messageObjectContentInnerType :: Text -- ^ Always `text`.
  , messageObjectContentInnerImageUnderscorefile :: MessageContentImageFileObjectImageFile -- ^ 
  , messageObjectContentInnerText :: MessageContentTextObjectText -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON MessageObjectContentInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "messageObjectContentInner")
instance ToJSON MessageObjectContentInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "messageObjectContentInner")


-- | Describes an OpenAI model offering that can be used with the API.
data Model = Model
  { modelId :: Text -- ^ The model identifier, which can be referenced in the API endpoints.
  , modelCreated :: Int -- ^ The Unix timestamp (in seconds) when the model was created.
  , modelObject :: Text -- ^ The object type, which is always \"model\".
  , modelOwnedUnderscoreby :: Text -- ^ The organization that owns the model.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON Model where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "model")
instance ToJSON Model where
  toJSON = genericToJSON (removeFieldLabelPrefix False "model")


-- | 
data ModifyAssistantRequest = ModifyAssistantRequest
  { modifyAssistantRequestModel :: Maybe CreateAssistantRequestModel -- ^ 
  , modifyAssistantRequestName :: Maybe Text -- ^ The name of the assistant. The maximum length is 256 characters. 
  , modifyAssistantRequestDescription :: Maybe Text -- ^ The description of the assistant. The maximum length is 512 characters. 
  , modifyAssistantRequestInstructions :: Maybe Text -- ^ The system instructions that the assistant uses. The maximum length is 32768 characters. 
  , modifyAssistantRequestTools :: Maybe [AssistantObjectToolsInner] -- ^ A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types `code_interpreter`, `retrieval`, or `function`. 
  , modifyAssistantRequestFileUnderscoreids :: Maybe [Text] -- ^ A list of [File](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. If a file was previously attached to the list but does not show up in the list, it will be deleted from the assistant. 
  , modifyAssistantRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ModifyAssistantRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "modifyAssistantRequest")
instance ToJSON ModifyAssistantRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "modifyAssistantRequest")


-- | 
data ModifyMessageRequest = ModifyMessageRequest
  { modifyMessageRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ModifyMessageRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "modifyMessageRequest")
instance ToJSON ModifyMessageRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "modifyMessageRequest")


-- | 
data ModifyRunRequest = ModifyRunRequest
  { modifyRunRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ModifyRunRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "modifyRunRequest")
instance ToJSON ModifyRunRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "modifyRunRequest")


-- | 
data ModifyThreadRequest = ModifyThreadRequest
  { modifyThreadRequestMetadata :: Maybe Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ModifyThreadRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "modifyThreadRequest")
instance ToJSON ModifyThreadRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "modifyThreadRequest")


-- | The &#x60;File&#x60; object represents a document that has been uploaded to OpenAI.
data OpenAIFile = OpenAIFile
  { openAIFileId :: Text -- ^ The file identifier, which can be referenced in the API endpoints.
  , openAIFileBytes :: Int -- ^ The size of the file, in bytes.
  , openAIFileCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the file was created.
  , openAIFileFilename :: Text -- ^ The name of the file.
  , openAIFileObject :: Text -- ^ The object type, which is always `file`.
  , openAIFilePurpose :: Text -- ^ The intended purpose of the file. Supported values are `fine-tune`, `fine-tune-results`, `assistants`, and `assistants_output`.
  , openAIFileStatus :: Text -- ^ Deprecated. The current status of the file, which can be either `uploaded`, `processed`, or `error`.
  , openAIFileStatusUnderscoredetails :: Maybe Text -- ^ Deprecated. For details on why a fine-tuning training file failed validation, see the `error` field on `fine_tuning.job`.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON OpenAIFile where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "openAIFile")
instance ToJSON OpenAIFile where
  toJSON = genericToJSON (removeFieldLabelPrefix False "openAIFile")


-- | Usage statistics related to the run. This value will be &#x60;null&#x60; if the run is not in a terminal state (i.e. &#x60;in_progress&#x60;, &#x60;queued&#x60;, etc.).
data RunCompletionUsage = RunCompletionUsage
  { runCompletionUsageCompletionUnderscoretokens :: Int -- ^ Number of completion tokens used over the course of the run.
  , runCompletionUsagePromptUnderscoretokens :: Int -- ^ Number of prompt tokens used over the course of the run.
  , runCompletionUsageTotalUnderscoretokens :: Int -- ^ Total number of tokens used (prompt + completion).
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunCompletionUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runCompletionUsage")
instance ToJSON RunCompletionUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runCompletionUsage")


-- | Represents an execution run on a [thread](/docs/api-reference/threads).
data RunObject = RunObject
  { runObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , runObjectObject :: Text -- ^ The object type, which is always `thread.run`.
  , runObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run was created.
  , runObjectThreadUnderscoreid :: Text -- ^ The ID of the [thread](/docs/api-reference/threads) that was executed on as a part of this run.
  , runObjectAssistantUnderscoreid :: Text -- ^ The ID of the [assistant](/docs/api-reference/assistants) used for execution of this run.
  , runObjectStatus :: Text -- ^ The status of the run, which can be either `queued`, `in_progress`, `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`, or `expired`.
  , runObjectRequiredUnderscoreaction :: RunObjectRequiredAction -- ^ 
  , runObjectLastUnderscoreerror :: RunObjectLastError -- ^ 
  , runObjectExpiresUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run will expire.
  , runObjectStartedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run was started.
  , runObjectCancelledUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run was cancelled.
  , runObjectFailedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run failed.
  , runObjectCompletedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run was completed.
  , runObjectModel :: Text -- ^ The model that the [assistant](/docs/api-reference/assistants) used for this run.
  , runObjectInstructions :: Text -- ^ The instructions that the [assistant](/docs/api-reference/assistants) used for this run.
  , runObjectTools :: [AssistantObjectToolsInner] -- ^ The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.
  , runObjectFileUnderscoreids :: [Text] -- ^ The list of [File](/docs/api-reference/files) IDs the [assistant](/docs/api-reference/assistants) used for this run.
  , runObjectMetadata :: Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  , runObjectUsage :: RunCompletionUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runObject")
instance ToJSON RunObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runObject")


-- | The last error associated with this run. Will be &#x60;null&#x60; if there are no errors.
data RunObjectLastError = RunObjectLastError
  { runObjectLastErrorCode :: Text -- ^ One of `server_error`, `rate_limit_exceeded`, or `invalid_prompt`.
  , runObjectLastErrorMessage :: Text -- ^ A human-readable description of the error.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunObjectLastError where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runObjectLastError")
instance ToJSON RunObjectLastError where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runObjectLastError")


-- | Details on the action required to continue the run. Will be &#x60;null&#x60; if no action is required.
data RunObjectRequiredAction = RunObjectRequiredAction
  { runObjectRequiredActionType :: Text -- ^ For now, this is always `submit_tool_outputs`.
  , runObjectRequiredActionSubmitUnderscoretoolUnderscoreoutputs :: RunObjectRequiredActionSubmitToolOutputs -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunObjectRequiredAction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runObjectRequiredAction")
instance ToJSON RunObjectRequiredAction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runObjectRequiredAction")


-- | Details on the tool outputs needed for this run to continue.
data RunObjectRequiredActionSubmitToolOutputs = RunObjectRequiredActionSubmitToolOutputs
  { runObjectRequiredActionSubmitToolOutputsToolUnderscorecalls :: [RunToolCallObject] -- ^ A list of the relevant tool calls.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunObjectRequiredActionSubmitToolOutputs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runObjectRequiredActionSubmitToolOutputs")
instance ToJSON RunObjectRequiredActionSubmitToolOutputs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runObjectRequiredActionSubmitToolOutputs")


-- | Usage statistics related to the run step. This value will be &#x60;null&#x60; while the run step&#39;s status is &#x60;in_progress&#x60;.
data RunStepCompletionUsage = RunStepCompletionUsage
  { runStepCompletionUsageCompletionUnderscoretokens :: Int -- ^ Number of completion tokens used over the course of the run step.
  , runStepCompletionUsagePromptUnderscoretokens :: Int -- ^ Number of prompt tokens used over the course of the run step.
  , runStepCompletionUsageTotalUnderscoretokens :: Int -- ^ Total number of tokens used (prompt + completion).
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepCompletionUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepCompletionUsage")
instance ToJSON RunStepCompletionUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepCompletionUsage")


-- | Details of the message creation by the run step.
data RunStepDetailsMessageCreationObject = RunStepDetailsMessageCreationObject
  { runStepDetailsMessageCreationObjectType :: Text -- ^ Always `message_creation`.
  , runStepDetailsMessageCreationObjectMessageUnderscorecreation :: RunStepDetailsMessageCreationObjectMessageCreation -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsMessageCreationObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsMessageCreationObject")
instance ToJSON RunStepDetailsMessageCreationObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsMessageCreationObject")


-- | 
data RunStepDetailsMessageCreationObjectMessageCreation = RunStepDetailsMessageCreationObjectMessageCreation
  { runStepDetailsMessageCreationObjectMessageCreationMessageUnderscoreid :: Text -- ^ The ID of the message that was created by this run step.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsMessageCreationObjectMessageCreation where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsMessageCreationObjectMessageCreation")
instance ToJSON RunStepDetailsMessageCreationObjectMessageCreation where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsMessageCreationObjectMessageCreation")


-- | Details of the Code Interpreter tool call the run step was involved in.
data RunStepDetailsToolCallsCodeObject = RunStepDetailsToolCallsCodeObject
  { runStepDetailsToolCallsCodeObjectId :: Text -- ^ The ID of the tool call.
  , runStepDetailsToolCallsCodeObjectType :: Text -- ^ The type of tool call. This is always going to be `code_interpreter` for this type of tool call.
  , runStepDetailsToolCallsCodeObjectCodeUnderscoreinterpreter :: RunStepDetailsToolCallsCodeObjectCodeInterpreter -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeObject")
instance ToJSON RunStepDetailsToolCallsCodeObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeObject")


-- | The Code Interpreter tool call definition.
data RunStepDetailsToolCallsCodeObjectCodeInterpreter = RunStepDetailsToolCallsCodeObjectCodeInterpreter
  { runStepDetailsToolCallsCodeObjectCodeInterpreterInput :: Text -- ^ The input to the Code Interpreter tool call.
  , runStepDetailsToolCallsCodeObjectCodeInterpreterOutputs :: [RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner] -- ^ The outputs from the Code Interpreter tool call. Code Interpreter can output one or more items, including text (`logs`) or images (`image`). Each of these are represented by a different object type.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeObjectCodeInterpreter where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeObjectCodeInterpreter")
instance ToJSON RunStepDetailsToolCallsCodeObjectCodeInterpreter where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeObjectCodeInterpreter")


-- | 
data RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner = RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner
  { runStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInnerType :: Text -- ^ Always `image`.
  , runStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInnerLogs :: Text -- ^ The text output from the Code Interpreter tool call.
  , runStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInnerImage :: RunStepDetailsToolCallsCodeOutputImageObjectImage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner")
instance ToJSON RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner")


-- | 
data RunStepDetailsToolCallsCodeOutputImageObject = RunStepDetailsToolCallsCodeOutputImageObject
  { runStepDetailsToolCallsCodeOutputImageObjectType :: Text -- ^ Always `image`.
  , runStepDetailsToolCallsCodeOutputImageObjectImage :: RunStepDetailsToolCallsCodeOutputImageObjectImage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeOutputImageObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeOutputImageObject")
instance ToJSON RunStepDetailsToolCallsCodeOutputImageObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeOutputImageObject")


-- | 
data RunStepDetailsToolCallsCodeOutputImageObjectImage = RunStepDetailsToolCallsCodeOutputImageObjectImage
  { runStepDetailsToolCallsCodeOutputImageObjectImageFileUnderscoreid :: Text -- ^ The [file](/docs/api-reference/files) ID of the image.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeOutputImageObjectImage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeOutputImageObjectImage")
instance ToJSON RunStepDetailsToolCallsCodeOutputImageObjectImage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeOutputImageObjectImage")


-- | Text output from the Code Interpreter tool call as part of a run step.
data RunStepDetailsToolCallsCodeOutputLogsObject = RunStepDetailsToolCallsCodeOutputLogsObject
  { runStepDetailsToolCallsCodeOutputLogsObjectType :: Text -- ^ Always `logs`.
  , runStepDetailsToolCallsCodeOutputLogsObjectLogs :: Text -- ^ The text output from the Code Interpreter tool call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsCodeOutputLogsObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsCodeOutputLogsObject")
instance ToJSON RunStepDetailsToolCallsCodeOutputLogsObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsCodeOutputLogsObject")


-- | 
data RunStepDetailsToolCallsFunctionObject = RunStepDetailsToolCallsFunctionObject
  { runStepDetailsToolCallsFunctionObjectId :: Text -- ^ The ID of the tool call object.
  , runStepDetailsToolCallsFunctionObjectType :: Text -- ^ The type of tool call. This is always going to be `function` for this type of tool call.
  , runStepDetailsToolCallsFunctionObjectFunction :: RunStepDetailsToolCallsFunctionObjectFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsFunctionObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsFunctionObject")
instance ToJSON RunStepDetailsToolCallsFunctionObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsFunctionObject")


-- | The definition of the function that was called.
data RunStepDetailsToolCallsFunctionObjectFunction = RunStepDetailsToolCallsFunctionObjectFunction
  { runStepDetailsToolCallsFunctionObjectFunctionName :: Text -- ^ The name of the function.
  , runStepDetailsToolCallsFunctionObjectFunctionArguments :: Text -- ^ The arguments passed to the function.
  , runStepDetailsToolCallsFunctionObjectFunctionOutput :: Text -- ^ The output of the function. This will be `null` if the outputs have not been [submitted](/docs/api-reference/runs/submitToolOutputs) yet.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsFunctionObjectFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsFunctionObjectFunction")
instance ToJSON RunStepDetailsToolCallsFunctionObjectFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsFunctionObjectFunction")


-- | Details of the tool call.
data RunStepDetailsToolCallsObject = RunStepDetailsToolCallsObject
  { runStepDetailsToolCallsObjectType :: Text -- ^ Always `tool_calls`.
  , runStepDetailsToolCallsObjectToolUnderscorecalls :: [RunStepDetailsToolCallsObjectToolCallsInner] -- ^ An array of tool calls the run step was involved in. These can be associated with one of three types of tools: `code_interpreter`, `retrieval`, or `function`. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsObject")
instance ToJSON RunStepDetailsToolCallsObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsObject")


-- | 
data RunStepDetailsToolCallsObjectToolCallsInner = RunStepDetailsToolCallsObjectToolCallsInner
  { runStepDetailsToolCallsObjectToolCallsInnerId :: Text -- ^ The ID of the tool call object.
  , runStepDetailsToolCallsObjectToolCallsInnerType :: Text -- ^ The type of tool call. This is always going to be `function` for this type of tool call.
  , runStepDetailsToolCallsObjectToolCallsInnerCodeUnderscoreinterpreter :: RunStepDetailsToolCallsCodeObjectCodeInterpreter -- ^ 
  , runStepDetailsToolCallsObjectToolCallsInnerRetrieval :: Value -- ^ For now, this is always going to be an empty object.
  , runStepDetailsToolCallsObjectToolCallsInnerFunction :: RunStepDetailsToolCallsFunctionObjectFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsObjectToolCallsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsObjectToolCallsInner")
instance ToJSON RunStepDetailsToolCallsObjectToolCallsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsObjectToolCallsInner")


-- | 
data RunStepDetailsToolCallsRetrievalObject = RunStepDetailsToolCallsRetrievalObject
  { runStepDetailsToolCallsRetrievalObjectId :: Text -- ^ The ID of the tool call object.
  , runStepDetailsToolCallsRetrievalObjectType :: Text -- ^ The type of tool call. This is always going to be `retrieval` for this type of tool call.
  , runStepDetailsToolCallsRetrievalObjectRetrieval :: Value -- ^ For now, this is always going to be an empty object.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepDetailsToolCallsRetrievalObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepDetailsToolCallsRetrievalObject")
instance ToJSON RunStepDetailsToolCallsRetrievalObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepDetailsToolCallsRetrievalObject")


-- | Represents a step in execution of a run. 
data RunStepObject = RunStepObject
  { runStepObjectId :: Text -- ^ The identifier of the run step, which can be referenced in API endpoints.
  , runStepObjectObject :: Text -- ^ The object type, which is always `thread.run.step`.
  , runStepObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run step was created.
  , runStepObjectAssistantUnderscoreid :: Text -- ^ The ID of the [assistant](/docs/api-reference/assistants) associated with the run step.
  , runStepObjectThreadUnderscoreid :: Text -- ^ The ID of the [thread](/docs/api-reference/threads) that was run.
  , runStepObjectRunUnderscoreid :: Text -- ^ The ID of the [run](/docs/api-reference/runs) that this run step is a part of.
  , runStepObjectType :: Text -- ^ The type of run step, which can be either `message_creation` or `tool_calls`.
  , runStepObjectStatus :: Text -- ^ The status of the run step, which can be either `in_progress`, `cancelled`, `failed`, `completed`, or `expired`.
  , runStepObjectStepUnderscoredetails :: RunStepObjectStepDetails -- ^ 
  , runStepObjectLastUnderscoreerror :: RunStepObjectLastError -- ^ 
  , runStepObjectExpiredUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run step expired. A step is considered expired if the parent run is expired.
  , runStepObjectCancelledUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run step was cancelled.
  , runStepObjectFailedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run step failed.
  , runStepObjectCompletedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the run step completed.
  , runStepObjectMetadata :: Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  , runStepObjectUsage :: RunStepCompletionUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepObject")
instance ToJSON RunStepObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepObject")


-- | The last error associated with this run step. Will be &#x60;null&#x60; if there are no errors.
data RunStepObjectLastError = RunStepObjectLastError
  { runStepObjectLastErrorCode :: Text -- ^ One of `server_error` or `rate_limit_exceeded`.
  , runStepObjectLastErrorMessage :: Text -- ^ A human-readable description of the error.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepObjectLastError where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepObjectLastError")
instance ToJSON RunStepObjectLastError where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepObjectLastError")


-- | The details of the run step.
data RunStepObjectStepDetails = RunStepObjectStepDetails
  { runStepObjectStepDetailsType :: Text -- ^ Always `tool_calls`.
  , runStepObjectStepDetailsMessageUnderscorecreation :: RunStepDetailsMessageCreationObjectMessageCreation -- ^ 
  , runStepObjectStepDetailsToolUnderscorecalls :: [RunStepDetailsToolCallsObjectToolCallsInner] -- ^ An array of tool calls the run step was involved in. These can be associated with one of three types of tools: `code_interpreter`, `retrieval`, or `function`. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunStepObjectStepDetails where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runStepObjectStepDetails")
instance ToJSON RunStepObjectStepDetails where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runStepObjectStepDetails")


-- | Tool call objects
data RunToolCallObject = RunToolCallObject
  { runToolCallObjectId :: Text -- ^ The ID of the tool call. This ID must be referenced when you submit the tool outputs in using the [Submit tool outputs to run](/docs/api-reference/runs/submitToolOutputs) endpoint.
  , runToolCallObjectType :: Text -- ^ The type of tool call the output is required for. For now, this is always `function`.
  , runToolCallObjectFunction :: RunToolCallObjectFunction -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunToolCallObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runToolCallObject")
instance ToJSON RunToolCallObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runToolCallObject")


-- | The function definition.
data RunToolCallObjectFunction = RunToolCallObjectFunction
  { runToolCallObjectFunctionName :: Text -- ^ The name of the function.
  , runToolCallObjectFunctionArguments :: Text -- ^ The arguments that the model expects you to pass to the function.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON RunToolCallObjectFunction where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "runToolCallObjectFunction")
instance ToJSON RunToolCallObjectFunction where
  toJSON = genericToJSON (removeFieldLabelPrefix False "runToolCallObjectFunction")


-- | 
data SubmitToolOutputsRunRequest = SubmitToolOutputsRunRequest
  { submitToolOutputsRunRequestToolUnderscoreoutputs :: [SubmitToolOutputsRunRequestToolOutputsInner] -- ^ A list of tools for which the outputs are being submitted.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON SubmitToolOutputsRunRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "submitToolOutputsRunRequest")
instance ToJSON SubmitToolOutputsRunRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "submitToolOutputsRunRequest")


-- | 
data SubmitToolOutputsRunRequestToolOutputsInner = SubmitToolOutputsRunRequestToolOutputsInner
  { submitToolOutputsRunRequestToolOutputsInnerToolUnderscorecallUnderscoreid :: Maybe Text -- ^ The ID of the tool call in the `required_action` object within the run object the output is being submitted for.
  , submitToolOutputsRunRequestToolOutputsInnerOutput :: Maybe Text -- ^ The output of the tool call to be submitted to continue the run.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON SubmitToolOutputsRunRequestToolOutputsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "submitToolOutputsRunRequestToolOutputsInner")
instance ToJSON SubmitToolOutputsRunRequestToolOutputsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "submitToolOutputsRunRequestToolOutputsInner")


-- | Represents a thread that contains [messages](/docs/api-reference/messages).
data ThreadObject = ThreadObject
  { threadObjectId :: Text -- ^ The identifier, which can be referenced in API endpoints.
  , threadObjectObject :: Text -- ^ The object type, which is always `thread`.
  , threadObjectCreatedUnderscoreat :: Int -- ^ The Unix timestamp (in seconds) for when the thread was created.
  , threadObjectMetadata :: Value -- ^ Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maxium of 512 characters long. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ThreadObject where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "threadObject")
instance ToJSON ThreadObject where
  toJSON = genericToJSON (removeFieldLabelPrefix False "threadObject")


-- | 
data TranscriptionSegment = TranscriptionSegment
  { transcriptionSegmentId :: Int -- ^ Unique identifier of the segment.
  , transcriptionSegmentSeek :: Int -- ^ Seek offset of the segment.
  , transcriptionSegmentStart :: Float -- ^ Start time of the segment in seconds.
  , transcriptionSegmentEnd :: Float -- ^ End time of the segment in seconds.
  , transcriptionSegmentText :: Text -- ^ Text content of the segment.
  , transcriptionSegmentTokens :: [Int] -- ^ Array of token IDs for the text content.
  , transcriptionSegmentTemperature :: Float -- ^ Temperature parameter used for generating the segment.
  , transcriptionSegmentAvgUnderscorelogprob :: Float -- ^ Average logprob of the segment. If the value is lower than -1, consider the logprobs failed.
  , transcriptionSegmentCompressionUnderscoreratio :: Float -- ^ Compression ratio of the segment. If the value is greater than 2.4, consider the compression failed.
  , transcriptionSegmentNoUnderscorespeechUnderscoreprob :: Float -- ^ Probability of no speech in the segment. If the value is higher than 1.0 and the `avg_logprob` is below -1, consider this segment silent.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON TranscriptionSegment where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "transcriptionSegment")
instance ToJSON TranscriptionSegment where
  toJSON = genericToJSON (removeFieldLabelPrefix False "transcriptionSegment")


-- | 
data TranscriptionWord = TranscriptionWord
  { transcriptionWordWord :: Text -- ^ The text content of the word.
  , transcriptionWordStart :: Float -- ^ Start time of the word in seconds.
  , transcriptionWordEnd :: Float -- ^ End time of the word in seconds.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON TranscriptionWord where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "transcriptionWord")
instance ToJSON TranscriptionWord where
  toJSON = genericToJSON (removeFieldLabelPrefix False "transcriptionWord")


uncapitalize :: String -> String
uncapitalize (first:rest) = Char.toLower first : rest
uncapitalize [] = []

-- | Remove a field label prefix during JSON parsing.
--   Also perform any replacements for special characters.
--   The @forParsing@ parameter is to distinguish between the cases in which we're using this
--   to power a @FromJSON@ or a @ToJSON@ instance. In the first case we're parsing, and we want
--   to replace special characters with their quoted equivalents (because we cannot have special
--   chars in identifier names), while we want to do vice versa when sending data instead.
removeFieldLabelPrefix :: Bool -> String -> Options
removeFieldLabelPrefix forParsing prefix =
  defaultOptions
    { omitNothingFields  = True
    , fieldLabelModifier = uncapitalize . fromMaybe (error ("did not find prefix " ++ prefix)) . stripPrefix prefix . replaceSpecialChars
    }
  where
    replaceSpecialChars field = foldl (&) field (map mkCharReplacement specialChars)
    specialChars =
      [ ("$", "'Dollar")
      , ("^", "'Caret")
      , ("|", "'Pipe")
      , ("=", "'Equal")
      , ("*", "'Star")
      , ("-", "'Dash")
      , ("&", "'Ampersand")
      , ("%", "'Percent")
      , ("#", "'Hash")
      , ("@", "'At")
      , ("!", "'Exclamation")
      , ("+", "'Plus")
      , (":", "'Colon")
      , (";", "'Semicolon")
      , (">", "'GreaterThan")
      , ("<", "'LessThan")
      , (".", "'Period")
      , ("_", "'Underscore")
      , ("?", "'Question_Mark")
      , (",", "'Comma")
      , ("'", "'Quote")
      , ("/", "'Slash")
      , ("(", "'Left_Parenthesis")
      , (")", "'Right_Parenthesis")
      , ("{", "'Left_Curly_Bracket")
      , ("}", "'Right_Curly_Bracket")
      , ("[", "'Left_Square_Bracket")
      , ("]", "'Right_Square_Bracket")
      , ("~", "'Tilde")
      , ("`", "'Backtick")
      , ("<=", "'Less_Than_Or_Equal_To")
      , (">=", "'Greater_Than_Or_Equal_To")
      , ("!=", "'Not_Equal")
      , ("<>", "'Not_Equal")
      , ("~=", "'Tilde_Equal")
      , ("\\", "'Back_Slash")
      , ("\"", "'Double_Quote")
      ]
    mkCharReplacement (replaceStr, searchStr) = T.unpack . replacer (T.pack searchStr) (T.pack replaceStr) . T.pack
    replacer =
      if forParsing
        then flip T.replace
        else T.replace
