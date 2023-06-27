{-# LANGUAGE NoImplicitPrelude #-}
{-# OPTIONS_GHC -fno-warn-unused-matches #-}
module Handler.OpenAI where

import           Import


-- | Immediately cancel a fine-tune job. 
--
-- operationId: cancelFineTune
postFineTunesByTextCancelR :: Text -- ^ The ID of the fine-tune job to cancel 
                           -> Handler Value
postFineTunesByTextCancelR fineTuneId = notImplemented

-- | Creates a model response for the given chat conversation.
--
-- operationId: createChatCompletion
postChatCompletionsR :: Handler Value
postChatCompletionsR = notImplemented

-- | Creates a completion for the provided prompt and parameters.
--
-- operationId: createCompletion
postCompletionsR :: Handler Value
postCompletionsR = notImplemented

-- | Creates a new edit for the provided input, instruction, and parameters.
--
-- operationId: createEdit
postEditsR :: Handler Value
postEditsR = notImplemented

-- | Creates an embedding vector representing the input text.
--
-- operationId: createEmbedding
postEmbeddingsR :: Handler Value
postEmbeddingsR = notImplemented

-- | Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB. Please contact us if you need to increase the storage limit. 
--
-- operationId: createFile
postFilesR :: Handler Value
postFilesR = notImplemented

-- | Creates a job that fine-tunes a specified model from a given dataset.  Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.  [Learn more about Fine-tuning](/docs/guides/fine-tuning) 
--
-- operationId: createFineTune
postFineTunesR :: Handler Value
postFineTunesR = notImplemented

-- | Creates an image given a prompt.
--
-- operationId: createImage
postImagesGenerationsR :: Handler Value
postImagesGenerationsR = notImplemented

-- | Creates an edited or extended image given an original image and a prompt.
--
-- operationId: createImageEdit
postImagesEditsR :: Handler Value
postImagesEditsR = notImplemented

-- | Creates a variation of a given image.
--
-- operationId: createImageVariation
postImagesVariationsR :: Handler Value
postImagesVariationsR = notImplemented

-- | Classifies if text violates OpenAI&#39;s Content Policy
--
-- operationId: createModeration
postModerationsR :: Handler Value
postModerationsR = notImplemented

-- | Transcribes audio into the input language.
--
-- operationId: createTranscription
postAudioTranscriptionsR :: Handler Value
postAudioTranscriptionsR = notImplemented

-- | Translates audio into English.
--
-- operationId: createTranslation
postAudioTranslationsR :: Handler Value
postAudioTranslationsR = notImplemented

-- | Delete a file.
--
-- operationId: deleteFile
deleteFilesByTextR :: Text -- ^ The ID of the file to use for this request
                   -> Handler Value
deleteFilesByTextR fileId = notImplemented

-- | Delete a fine-tuned model. You must have the Owner role in your organization.
--
-- operationId: deleteModel
deleteModelsByTextR :: Text -- ^ The model to delete
                    -> Handler Value
deleteModelsByTextR model = notImplemented

-- | Returns the contents of the specified file
--
-- operationId: downloadFile
getFilesByTextContentR :: Text -- ^ The ID of the file to use for this request
                       -> Handler Value
getFilesByTextContentR fileId = notImplemented

-- | Returns a list of files that belong to the user&#39;s organization.
--
-- operationId: listFiles
getFilesR :: Handler Value
getFilesR = notImplemented

-- | Get fine-grained status updates for a fine-tune job. 
--
-- operationId: listFineTuneEvents
getFineTunesByTextEventsR :: Text -- ^ The ID of the fine-tune job to get events for. 
                          -> Handler Value
getFineTunesByTextEventsR fineTuneId = notImplemented

-- | List your organization&#39;s fine-tuning jobs 
--
-- operationId: listFineTunes
getFineTunesR :: Handler Value
getFineTunesR = notImplemented

-- | Lists the currently available models, and provides basic information about each one such as the owner and availability.
--
-- operationId: listModels
getModelsR :: Handler Value
getModelsR = notImplemented

-- | Returns information about a specific file.
--
-- operationId: retrieveFile
getFilesByTextR :: Text -- ^ The ID of the file to use for this request
                -> Handler Value
getFilesByTextR fileId = notImplemented

-- | Gets info about the fine-tune job.  [Learn more about Fine-tuning](/docs/guides/fine-tuning) 
--
-- operationId: retrieveFineTune
getFineTunesByTextR :: Text -- ^ The ID of the fine-tune job 
                    -> Handler Value
getFineTunesByTextR fineTuneId = notImplemented

-- | Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
--
-- operationId: retrieveModel
getModelsByTextR :: Text -- ^ The ID of the model to use for this request
                 -> Handler Value
getModelsByTextR model = notImplemented
