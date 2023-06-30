{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveDataTypeable         #-}
{-# LANGUAGE DeriveGeneric              #-}
{-# OPTIONS_GHC -fno-warn-unused-binds -fno-warn-unused-imports #-}

module OpenAI.Types (
  ChatCompletionFunctions (..),
  ChatCompletionRequestMessage (..),
  ChatCompletionRequestMessageFunctionCall (..),
  ChatCompletionResponseMessage (..),
  ChatCompletionResponseMessageFunctionCall (..),
  ChatCompletionStreamResponseDelta (..),
  CreateChatCompletionRequest (..),
  CreateChatCompletionRequestFunctionCall (..),
  CreateChatCompletionRequestFunctionCallOneOf (..),
  CreateChatCompletionRequestModel (..),
  CreateChatCompletionRequestStop (..),
  CreateChatCompletionResponse (..),
  CreateChatCompletionResponseChoicesInner (..),
  CreateChatCompletionStreamResponse (..),
  CreateChatCompletionStreamResponseChoicesInner (..),
  CreateCompletionRequest (..),
  CreateCompletionRequestModel (..),
  CreateCompletionRequestPrompt (..),
  CreateCompletionRequestStop (..),
  CreateCompletionResponse (..),
  CreateCompletionResponseChoicesInner (..),
  CreateCompletionResponseChoicesInnerLogprobs (..),
  CreateCompletionResponseUsage (..),
  CreateEditRequest (..),
  CreateEditRequestModel (..),
  CreateEditResponse (..),
  CreateEditResponseChoicesInner (..),
  CreateEmbeddingRequest (..),
  CreateEmbeddingRequestInput (..),
  CreateEmbeddingRequestModel (..),
  CreateEmbeddingResponse (..),
  CreateEmbeddingResponseDataInner (..),
  CreateEmbeddingResponseUsage (..),
  CreateFineTuneRequest (..),
  CreateFineTuneRequestModel (..),
  CreateImageRequest (..),
  CreateModerationRequest (..),
  CreateModerationRequestInput (..),
  CreateModerationRequestModel (..),
  CreateModerationResponse (..),
  CreateModerationResponseResultsInner (..),
  CreateModerationResponseResultsInnerCategories (..),
  CreateModerationResponseResultsInnerCategoryScores (..),
  CreateTranscriptionResponse (..),
  CreateTranslationResponse (..),
  DeleteFileResponse (..),
  DeleteModelResponse (..),
  Error (..),
  ErrorResponse (..),
  FineTune (..),
  FineTuneEvent (..),
  ImagesResponse (..),
  ImagesResponseDataInner (..),
  ListFilesResponse (..),
  ListFineTuneEventsResponse (..),
  ListFineTunesResponse (..),
  ListModelsResponse (..),
  Model (..),
  OpenAIFile (..),
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


-- | 
data ChatCompletionFunctions = ChatCompletionFunctions
  { chatCompletionFunctionsName :: Text -- ^ The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
  , chatCompletionFunctionsDescription :: Maybe Text -- ^ A description of what the function does, used by the model to choose when and how to call the function.
  , chatCompletionFunctionsParameters :: (Map.Map String Value) -- ^ The parameters the functions accepts, described as a JSON Schema object. See the [guide](/docs/guides/gpt/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.  To describe a function that accepts no parameters, provide the value `{\"type\": \"object\", \"properties\": {}}`.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionFunctions where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionFunctions")
instance ToJSON ChatCompletionFunctions where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionFunctions")


-- | 
data ChatCompletionRequestMessage = ChatCompletionRequestMessage
  { chatCompletionRequestMessageRole :: Text -- ^ The role of the messages author. One of `system`, `user`, `assistant`, or `function`.
  , chatCompletionRequestMessageContent :: Text -- ^ The contents of the message. `content` is required for all messages, and may be null for assistant messages with function calls.
  , chatCompletionRequestMessageName :: Maybe Text -- ^ The name of the author of this message. `name` is required if role is `function`, and it should be the name of the function whose response is in the `content`. May contain a-z, A-Z, 0-9, and underscores, with a maximum length of 64 characters.
  , chatCompletionRequestMessageFunctionUnderscorecall :: Maybe ChatCompletionRequestMessageFunctionCall -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessage")
instance ToJSON ChatCompletionRequestMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessage")


-- | The name and arguments of a function that should be called, as generated by the model.
data ChatCompletionRequestMessageFunctionCall = ChatCompletionRequestMessageFunctionCall
  { chatCompletionRequestMessageFunctionCallName :: Text -- ^ The name of the function to call.
  , chatCompletionRequestMessageFunctionCallArguments :: Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionRequestMessageFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionRequestMessageFunctionCall")
instance ToJSON ChatCompletionRequestMessageFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionRequestMessageFunctionCall")


-- | 
data ChatCompletionResponseMessage = ChatCompletionResponseMessage
  { chatCompletionResponseMessageRole :: Text -- ^ The role of the author of this message.
  , chatCompletionResponseMessageContent :: Maybe Text -- ^ The contents of the message.
  , chatCompletionResponseMessageFunctionUnderscorecall :: Maybe ChatCompletionResponseMessageFunctionCall -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionResponseMessage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionResponseMessage")
instance ToJSON ChatCompletionResponseMessage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionResponseMessage")


-- | The name and arguments of a function that should be called, as generated by the model.
data ChatCompletionResponseMessageFunctionCall = ChatCompletionResponseMessageFunctionCall
  { chatCompletionResponseMessageFunctionCallName :: Maybe Text -- ^ The name of the function to call.
  , chatCompletionResponseMessageFunctionCallArguments :: Maybe Text -- ^ The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionResponseMessageFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionResponseMessageFunctionCall")
instance ToJSON ChatCompletionResponseMessageFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionResponseMessageFunctionCall")


-- | 
data ChatCompletionStreamResponseDelta = ChatCompletionStreamResponseDelta
  { chatCompletionStreamResponseDeltaRole :: Maybe Text -- ^ The role of the author of this message.
  , chatCompletionStreamResponseDeltaContent :: Maybe Text -- ^ The contents of the chunk message.
  , chatCompletionStreamResponseDeltaFunctionUnderscorecall :: Maybe ChatCompletionResponseMessageFunctionCall -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ChatCompletionStreamResponseDelta where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "chatCompletionStreamResponseDelta")
instance ToJSON ChatCompletionStreamResponseDelta where
  toJSON = genericToJSON (removeFieldLabelPrefix False "chatCompletionStreamResponseDelta")


-- | 
data CreateChatCompletionRequest = CreateChatCompletionRequest
  { createChatCompletionRequestModel :: CreateChatCompletionRequestModel -- ^ 
  , createChatCompletionRequestMessages :: [ChatCompletionRequestMessage] -- ^ A list of messages comprising the conversation so far. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb).
  , createChatCompletionRequestFunctions :: Maybe [ChatCompletionFunctions] -- ^ A list of functions the model may generate JSON inputs for.
  , createChatCompletionRequestFunctionUnderscorecall :: Maybe CreateChatCompletionRequestFunctionCall -- ^ 
  , createChatCompletionRequestTemperature :: Maybe Double -- ^ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. 
  , createChatCompletionRequestTopUnderscorep :: Maybe Double -- ^ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. 
  , createChatCompletionRequestN :: Maybe Int -- ^ How many chat completion choices to generate for each input message.
  , createChatCompletionRequestStream :: Maybe Bool -- ^ If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb). 
  , createChatCompletionRequestStop :: Maybe CreateChatCompletionRequestStop -- ^ 
  , createChatCompletionRequestMaxUnderscoretokens :: Maybe Int -- ^ The maximum number of [tokens](/tokenizer) to generate in the chat completion.  The total length of input tokens and generated tokens is limited by the model's context length. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) for counting tokens. 
  , createChatCompletionRequestPresenceUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details) 
  , createChatCompletionRequestFrequencyUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details) 
  , createChatCompletionRequestLogitUnderscorebias :: Maybe Value -- ^ Modify the likelihood of specified tokens appearing in the completion.  Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token. 
  , createChatCompletionRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequest")
instance ToJSON CreateChatCompletionRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequest")


-- | Controls how the model responds to function calls. \&quot;none\&quot; means the model does not call a function, and responds to the end-user. \&quot;auto\&quot; means the model can pick between an end-user or calling a function.  Specifying a particular function via &#x60;{\&quot;name\&quot;:\\ \&quot;my_function\&quot;}&#x60; forces the model to call that function. \&quot;none\&quot; is the default when no functions are present. \&quot;auto\&quot; is the default if functions are present.
data CreateChatCompletionRequestFunctionCall = CreateChatCompletionRequestFunctionCall
  { createChatCompletionRequestFunctionCallName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestFunctionCall where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestFunctionCall")
instance ToJSON CreateChatCompletionRequestFunctionCall where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestFunctionCall")


-- | 
data CreateChatCompletionRequestFunctionCallOneOf = CreateChatCompletionRequestFunctionCallOneOf
  { createChatCompletionRequestFunctionCallOneOfName :: Text -- ^ The name of the function to call.
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestFunctionCallOneOf where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestFunctionCallOneOf")
instance ToJSON CreateChatCompletionRequestFunctionCallOneOf where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestFunctionCallOneOf")


-- | ID of the model to use. See the [model endpoint compatibility](/docs/models/model-endpoint-compatibility) table for details on which models work with the Chat API.
data CreateChatCompletionRequestModel = CreateChatCompletionRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestModel")
instance ToJSON CreateChatCompletionRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestModel")


-- | Up to 4 sequences where the API will stop generating further tokens. 
data CreateChatCompletionRequestStop = CreateChatCompletionRequestStop
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionRequestStop where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionRequestStop")
instance ToJSON CreateChatCompletionRequestStop where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionRequestStop")


-- | 
data CreateChatCompletionResponse = CreateChatCompletionResponse
  { createChatCompletionResponseId :: Text -- ^ 
  , createChatCompletionResponseObject :: Text -- ^ 
  , createChatCompletionResponseCreated :: Int -- ^ 
  , createChatCompletionResponseModel :: Text -- ^ 
  , createChatCompletionResponseChoices :: [CreateChatCompletionResponseChoicesInner] -- ^ 
  , createChatCompletionResponseUsage :: Maybe CreateCompletionResponseUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionResponse")
instance ToJSON CreateChatCompletionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionResponse")


-- | 
data CreateChatCompletionResponseChoicesInner = CreateChatCompletionResponseChoicesInner
  { createChatCompletionResponseChoicesInnerIndex :: Maybe Int -- ^ 
  , createChatCompletionResponseChoicesInnerMessage :: Maybe ChatCompletionResponseMessage -- ^ 
  , createChatCompletionResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionResponseChoicesInner")
instance ToJSON CreateChatCompletionResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionResponseChoicesInner")


-- | 
data CreateChatCompletionStreamResponse = CreateChatCompletionStreamResponse
  { createChatCompletionStreamResponseId :: Text -- ^ 
  , createChatCompletionStreamResponseObject :: Text -- ^ 
  , createChatCompletionStreamResponseCreated :: Int -- ^ 
  , createChatCompletionStreamResponseModel :: Text -- ^ 
  , createChatCompletionStreamResponseChoices :: [CreateChatCompletionStreamResponseChoicesInner] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionStreamResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionStreamResponse")
instance ToJSON CreateChatCompletionStreamResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionStreamResponse")


-- | 
data CreateChatCompletionStreamResponseChoicesInner = CreateChatCompletionStreamResponseChoicesInner
  { createChatCompletionStreamResponseChoicesInnerIndex :: Maybe Int -- ^ 
  , createChatCompletionStreamResponseChoicesInnerDelta :: Maybe ChatCompletionStreamResponseDelta -- ^ 
  , createChatCompletionStreamResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateChatCompletionStreamResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createChatCompletionStreamResponseChoicesInner")
instance ToJSON CreateChatCompletionStreamResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createChatCompletionStreamResponseChoicesInner")


-- | 
data CreateCompletionRequest = CreateCompletionRequest
  { createCompletionRequestModel :: CreateCompletionRequestModel -- ^ 
  , createCompletionRequestPrompt :: CreateCompletionRequestPrompt -- ^ 
  , createCompletionRequestSuffix :: Maybe Text -- ^ The suffix that comes after a completion of inserted text.
  , createCompletionRequestMaxUnderscoretokens :: Maybe Int -- ^ The maximum number of [tokens](/tokenizer) to generate in the completion.  The token count of your prompt plus `max_tokens` cannot exceed the model's context length. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) for counting tokens. 
  , createCompletionRequestTemperature :: Maybe Double -- ^ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. 
  , createCompletionRequestTopUnderscorep :: Maybe Double -- ^ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. 
  , createCompletionRequestN :: Maybe Int -- ^ How many completions to generate for each prompt.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. 
  , createCompletionRequestStream :: Maybe Bool -- ^ Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb). 
  , createCompletionRequestLogprobs :: Maybe Int -- ^ Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.  The maximum value for `logprobs` is 5. 
  , createCompletionRequestEcho :: Maybe Bool -- ^ Echo back the prompt in addition to the completion 
  , createCompletionRequestStop :: Maybe CreateCompletionRequestStop -- ^ 
  , createCompletionRequestPresenceUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details) 
  , createCompletionRequestFrequencyUnderscorepenalty :: Maybe Double -- ^ Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/api-reference/parameter-details) 
  , createCompletionRequestBestUnderscoreof :: Maybe Int -- ^ Generates `best_of` completions server-side and returns the \"best\" (the one with the highest log probability per token). Results cannot be streamed.  When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. 
  , createCompletionRequestLogitUnderscorebias :: Maybe Value -- ^ Modify the likelihood of specified tokens appearing in the completion.  Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass `{\"50256\": -100}` to prevent the <|endoftext|> token from being generated. 
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


-- | 
data CreateCompletionResponse = CreateCompletionResponse
  { createCompletionResponseId :: Text -- ^ 
  , createCompletionResponseObject :: Text -- ^ 
  , createCompletionResponseCreated :: Int -- ^ 
  , createCompletionResponseModel :: Text -- ^ 
  , createCompletionResponseChoices :: [CreateCompletionResponseChoicesInner] -- ^ 
  , createCompletionResponseUsage :: Maybe CreateCompletionResponseUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponse")
instance ToJSON CreateCompletionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponse")


-- | 
data CreateCompletionResponseChoicesInner = CreateCompletionResponseChoicesInner
  { createCompletionResponseChoicesInnerText :: Text -- ^ 
  , createCompletionResponseChoicesInnerIndex :: Int -- ^ 
  , createCompletionResponseChoicesInnerLogprobs :: CreateCompletionResponseChoicesInnerLogprobs -- ^ 
  , createCompletionResponseChoicesInnerFinishUnderscorereason :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponseChoicesInner")
instance ToJSON CreateCompletionResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponseChoicesInner")


-- | 
data CreateCompletionResponseChoicesInnerLogprobs = CreateCompletionResponseChoicesInnerLogprobs
  { createCompletionResponseChoicesInnerLogprobsTokens :: Maybe [Text] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTokenUnderscorelogprobs :: Maybe [Double] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTopUnderscorelogprobs :: Maybe [Value] -- ^ 
  , createCompletionResponseChoicesInnerLogprobsTextUnderscoreoffset :: Maybe [Int] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponseChoicesInnerLogprobs where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponseChoicesInnerLogprobs")
instance ToJSON CreateCompletionResponseChoicesInnerLogprobs where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponseChoicesInnerLogprobs")


-- | 
data CreateCompletionResponseUsage = CreateCompletionResponseUsage
  { createCompletionResponseUsagePromptUnderscoretokens :: Int -- ^ 
  , createCompletionResponseUsageCompletionUnderscoretokens :: Int -- ^ 
  , createCompletionResponseUsageTotalUnderscoretokens :: Int -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateCompletionResponseUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createCompletionResponseUsage")
instance ToJSON CreateCompletionResponseUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createCompletionResponseUsage")


-- | 
data CreateEditRequest = CreateEditRequest
  { createEditRequestModel :: CreateEditRequestModel -- ^ 
  , createEditRequestInput :: Maybe Text -- ^ The input text to use as a starting point for the edit.
  , createEditRequestInstruction :: Text -- ^ The instruction that tells the model how to edit the prompt.
  , createEditRequestN :: Maybe Int -- ^ How many edits to generate for the input and instruction.
  , createEditRequestTemperature :: Maybe Double -- ^ What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. 
  , createEditRequestTopUnderscorep :: Maybe Double -- ^ An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEditRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEditRequest")
instance ToJSON CreateEditRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEditRequest")


-- | ID of the model to use. You can use the &#x60;text-davinci-edit-001&#x60; or &#x60;code-davinci-edit-001&#x60; model with this endpoint.
data CreateEditRequestModel = CreateEditRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEditRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEditRequestModel")
instance ToJSON CreateEditRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEditRequestModel")


-- | 
data CreateEditResponse = CreateEditResponse
  { createEditResponseObject :: Text -- ^ 
  , createEditResponseCreated :: Int -- ^ 
  , createEditResponseChoices :: [CreateEditResponseChoicesInner] -- ^ 
  , createEditResponseUsage :: CreateCompletionResponseUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEditResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEditResponse")
instance ToJSON CreateEditResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEditResponse")


-- | 
data CreateEditResponseChoicesInner = CreateEditResponseChoicesInner
  { createEditResponseChoicesInnerText :: Maybe Text -- ^ 
  , createEditResponseChoicesInnerIndex :: Maybe Int -- ^ 
  , createEditResponseChoicesInnerLogprobs :: Maybe CreateCompletionResponseChoicesInnerLogprobs -- ^ 
  , createEditResponseChoicesInnerFinishUnderscorereason :: Maybe Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEditResponseChoicesInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEditResponseChoicesInner")
instance ToJSON CreateEditResponseChoicesInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEditResponseChoicesInner")


-- | 
data CreateEmbeddingRequest = CreateEmbeddingRequest
  { createEmbeddingRequestModel :: CreateEmbeddingRequestModel -- ^ 
  , createEmbeddingRequestInput :: CreateEmbeddingRequestInput -- ^ 
  , createEmbeddingRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingRequest")
instance ToJSON CreateEmbeddingRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingRequest")


-- | Input text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. Each input must not exceed the max input tokens for the model (8191 tokens for &#x60;text-embedding-ada-002&#x60;). [Example Python code](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) for counting tokens. 
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
  { createEmbeddingResponseObject :: Text -- ^ 
  , createEmbeddingResponseModel :: Text -- ^ 
  , createEmbeddingResponseData :: [CreateEmbeddingResponseDataInner] -- ^ 
  , createEmbeddingResponseUsage :: CreateEmbeddingResponseUsage -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingResponse")
instance ToJSON CreateEmbeddingResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingResponse")


-- | 
data CreateEmbeddingResponseDataInner = CreateEmbeddingResponseDataInner
  { createEmbeddingResponseDataInnerIndex :: Int -- ^ 
  , createEmbeddingResponseDataInnerObject :: Text -- ^ 
  , createEmbeddingResponseDataInnerEmbedding :: [Double] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingResponseDataInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingResponseDataInner")
instance ToJSON CreateEmbeddingResponseDataInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingResponseDataInner")


-- | 
data CreateEmbeddingResponseUsage = CreateEmbeddingResponseUsage
  { createEmbeddingResponseUsagePromptUnderscoretokens :: Int -- ^ 
  , createEmbeddingResponseUsageTotalUnderscoretokens :: Int -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateEmbeddingResponseUsage where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createEmbeddingResponseUsage")
instance ToJSON CreateEmbeddingResponseUsage where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createEmbeddingResponseUsage")


-- | 
data CreateFineTuneRequest = CreateFineTuneRequest
  { createFineTuneRequestTrainingUnderscorefile :: Text -- ^ The ID of an uploaded file that contains training data.  See [upload file](/docs/api-reference/files/upload) for how to upload a file.  Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys \"prompt\" and \"completion\". Additionally, you must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/fine-tuning/creating-training-data) for more details. 
  , createFineTuneRequestValidationUnderscorefile :: Maybe Text -- ^ The ID of an uploaded file that contains validation data.  If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the [fine-tuning results file](/docs/guides/fine-tuning/analyzing-your-fine-tuned-model). Your train and validation data should be mutually exclusive.  Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys \"prompt\" and \"completion\". Additionally, you must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/fine-tuning/creating-training-data) for more details. 
  , createFineTuneRequestModel :: Maybe CreateFineTuneRequestModel -- ^ 
  , createFineTuneRequestNUnderscoreepochs :: Maybe Int -- ^ The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset. 
  , createFineTuneRequestBatchUnderscoresize :: Maybe Int -- ^ The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.  By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general, we've found that larger batch sizes tend to work better for larger datasets. 
  , createFineTuneRequestLearningUnderscorerateUnderscoremultiplier :: Maybe Double -- ^ The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.  By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final `batch_size` (larger learning rates tend to perform better with larger batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results. 
  , createFineTuneRequestPromptUnderscorelossUnderscoreweight :: Maybe Double -- ^ The weight to use for loss on the prompt tokens. This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0), and can add a stabilizing effect to training when completions are short.  If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt. 
  , createFineTuneRequestComputeUnderscoreclassificationUnderscoremetrics :: Maybe Bool -- ^ If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch. These metrics can be viewed in the [results file](/docs/guides/fine-tuning/analyzing-your-fine-tuned-model).  In order to compute classification metrics, you must provide a `validation_file`. Additionally, you must specify `classification_n_classes` for multiclass classification or `classification_positive_class` for binary classification. 
  , createFineTuneRequestClassificationUnderscorenUnderscoreclasses :: Maybe Int -- ^ The number of classes in a classification task.  This parameter is required for multiclass classification. 
  , createFineTuneRequestClassificationUnderscorepositiveUnderscoreclass :: Maybe Text -- ^ The positive class in binary classification.  This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification. 
  , createFineTuneRequestClassificationUnderscorebetas :: Maybe [Double] -- ^ If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score is a generalization of F-1 score. This is only used for binary classification.  With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger beta score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision and less on recall. 
  , createFineTuneRequestSuffix :: Maybe Text -- ^ A string of up to 40 characters that will be added to your fine-tuned model name.  For example, a `suffix` of \"custom-model-name\" would produce a model name like `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`. 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuneRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuneRequest")
instance ToJSON CreateFineTuneRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuneRequest")


-- | The name of the base model to fine-tune. You can select one of \&quot;ada\&quot;, \&quot;babbage\&quot;, \&quot;curie\&quot;, \&quot;davinci\&quot;, or a fine-tuned model created after 2022-04-21. To learn more about these models, see the [Models](https://platform.openai.com/docs/models) documentation. 
data CreateFineTuneRequestModel = CreateFineTuneRequestModel
  { 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateFineTuneRequestModel where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createFineTuneRequestModel")
instance ToJSON CreateFineTuneRequestModel where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createFineTuneRequestModel")


-- | 
data CreateImageRequest = CreateImageRequest
  { createImageRequestPrompt :: Text -- ^ A text description of the desired image(s). The maximum length is 1000 characters.
  , createImageRequestN :: Maybe Int -- ^ The number of images to generate. Must be between 1 and 10.
  , createImageRequestSize :: Maybe Text -- ^ The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024`.
  , createImageRequestResponseUnderscoreformat :: Maybe Text -- ^ The format in which the generated images are returned. Must be one of `url` or `b64_json`.
  , createImageRequestUser :: Maybe Text -- ^ A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateImageRequest where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createImageRequest")
instance ToJSON CreateImageRequest where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createImageRequest")


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


-- | 
data CreateModerationResponse = CreateModerationResponse
  { createModerationResponseId :: Text -- ^ 
  , createModerationResponseModel :: Text -- ^ 
  , createModerationResponseResults :: [CreateModerationResponseResultsInner] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponse")
instance ToJSON CreateModerationResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponse")


-- | 
data CreateModerationResponseResultsInner = CreateModerationResponseResultsInner
  { createModerationResponseResultsInnerFlagged :: Bool -- ^ 
  , createModerationResponseResultsInnerCategories :: CreateModerationResponseResultsInnerCategories -- ^ 
  , createModerationResponseResultsInnerCategoryUnderscorescores :: CreateModerationResponseResultsInnerCategoryScores -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInner")
instance ToJSON CreateModerationResponseResultsInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInner")


-- | 
data CreateModerationResponseResultsInnerCategories = CreateModerationResponseResultsInnerCategories
  { createModerationResponseResultsInnerCategoriesHate :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesHateSlashthreatening :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesSelfDashharm :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesSexual :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesSexualSlashminors :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesViolence :: Bool -- ^ 
  , createModerationResponseResultsInnerCategoriesViolenceSlashgraphic :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInnerCategories where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInnerCategories")
instance ToJSON CreateModerationResponseResultsInnerCategories where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInnerCategories")


-- | 
data CreateModerationResponseResultsInnerCategoryScores = CreateModerationResponseResultsInnerCategoryScores
  { createModerationResponseResultsInnerCategoryScoresHate :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresHateSlashthreatening :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresSelfDashharm :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresSexual :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresSexualSlashminors :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresViolence :: Double -- ^ 
  , createModerationResponseResultsInnerCategoryScoresViolenceSlashgraphic :: Double -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateModerationResponseResultsInnerCategoryScores where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createModerationResponseResultsInnerCategoryScores")
instance ToJSON CreateModerationResponseResultsInnerCategoryScores where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createModerationResponseResultsInnerCategoryScores")


-- | 
data CreateTranscriptionResponse = CreateTranscriptionResponse
  { createTranscriptionResponseText :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranscriptionResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranscriptionResponse")
instance ToJSON CreateTranscriptionResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranscriptionResponse")


-- | 
data CreateTranslationResponse = CreateTranslationResponse
  { createTranslationResponseText :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON CreateTranslationResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "createTranslationResponse")
instance ToJSON CreateTranslationResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "createTranslationResponse")


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
data DeleteModelResponse = DeleteModelResponse
  { deleteModelResponseId :: Text -- ^ 
  , deleteModelResponseObject :: Text -- ^ 
  , deleteModelResponseDeleted :: Bool -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON DeleteModelResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "deleteModelResponse")
instance ToJSON DeleteModelResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "deleteModelResponse")


-- | 
data Error = Error
  { errorType :: Text -- ^ 
  , errorMessage :: Text -- ^ 
  , errorParam :: Text -- ^ 
  , errorCode :: Text -- ^ 
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


-- | 
data FineTune = FineTune
  { fineTuneId :: Text -- ^ 
  , fineTuneObject :: Text -- ^ 
  , fineTuneCreatedUnderscoreat :: Int -- ^ 
  , fineTuneUpdatedUnderscoreat :: Int -- ^ 
  , fineTuneModel :: Text -- ^ 
  , fineTuneFineUnderscoretunedUnderscoremodel :: Text -- ^ 
  , fineTuneOrganizationUnderscoreid :: Text -- ^ 
  , fineTuneStatus :: Text -- ^ 
  , fineTuneHyperparams :: Value -- ^ 
  , fineTuneTrainingUnderscorefiles :: [OpenAIFile] -- ^ 
  , fineTuneValidationUnderscorefiles :: [OpenAIFile] -- ^ 
  , fineTuneResultUnderscorefiles :: [OpenAIFile] -- ^ 
  , fineTuneEvents :: Maybe [FineTuneEvent] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTune where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTune")
instance ToJSON FineTune where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTune")


-- | 
data FineTuneEvent = FineTuneEvent
  { fineTuneEventObject :: Text -- ^ 
  , fineTuneEventCreatedUnderscoreat :: Int -- ^ 
  , fineTuneEventLevel :: Text -- ^ 
  , fineTuneEventMessage :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON FineTuneEvent where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "fineTuneEvent")
instance ToJSON FineTuneEvent where
  toJSON = genericToJSON (removeFieldLabelPrefix False "fineTuneEvent")


-- | 
data ImagesResponse = ImagesResponse
  { imagesResponseCreated :: Int -- ^ 
  , imagesResponseData :: [ImagesResponseDataInner] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ImagesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "imagesResponse")
instance ToJSON ImagesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "imagesResponse")


-- | 
data ImagesResponseDataInner = ImagesResponseDataInner
  { imagesResponseDataInnerUrl :: Maybe Text -- ^ 
  , imagesResponseDataInnerB64Underscorejson :: Maybe Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ImagesResponseDataInner where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "imagesResponseDataInner")
instance ToJSON ImagesResponseDataInner where
  toJSON = genericToJSON (removeFieldLabelPrefix False "imagesResponseDataInner")


-- | 
data ListFilesResponse = ListFilesResponse
  { listFilesResponseObject :: Text -- ^ 
  , listFilesResponseData :: [OpenAIFile] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListFilesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listFilesResponse")
instance ToJSON ListFilesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listFilesResponse")


-- | 
data ListFineTuneEventsResponse = ListFineTuneEventsResponse
  { listFineTuneEventsResponseObject :: Text -- ^ 
  , listFineTuneEventsResponseData :: [FineTuneEvent] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListFineTuneEventsResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listFineTuneEventsResponse")
instance ToJSON ListFineTuneEventsResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listFineTuneEventsResponse")


-- | 
data ListFineTunesResponse = ListFineTunesResponse
  { listFineTunesResponseObject :: Text -- ^ 
  , listFineTunesResponseData :: [FineTune] -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON ListFineTunesResponse where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "listFineTunesResponse")
instance ToJSON ListFineTunesResponse where
  toJSON = genericToJSON (removeFieldLabelPrefix False "listFineTunesResponse")


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
data Model = Model
  { modelId :: Text -- ^ 
  , modelObject :: Text -- ^ 
  , modelCreated :: Int -- ^ 
  , modelOwnedUnderscoreby :: Text -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON Model where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "model")
instance ToJSON Model where
  toJSON = genericToJSON (removeFieldLabelPrefix False "model")


-- | 
data OpenAIFile = OpenAIFile
  { openAIFileId :: Text -- ^ 
  , openAIFileObject :: Text -- ^ 
  , openAIFileBytes :: Int -- ^ 
  , openAIFileCreatedUnderscoreat :: Int -- ^ 
  , openAIFileFilename :: Text -- ^ 
  , openAIFilePurpose :: Text -- ^ 
  , openAIFileStatus :: Maybe Text -- ^ 
  , openAIFileStatusUnderscoredetails :: Maybe Value -- ^ 
  } deriving (Show, Eq, Generic, Data)

instance FromJSON OpenAIFile where
  parseJSON = genericParseJSON (removeFieldLabelPrefix True "openAIFile")
instance ToJSON OpenAIFile where
  toJSON = genericToJSON (removeFieldLabelPrefix False "openAIFile")


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
