{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE OverloadedStrings #-}
module Handler.OpenAISpec (spec) where

import           TestImport


spec :: Spec
spec = withApp $ do

    describe "postFineTunesByTextCancelR" $
        it "returns 501 Not Implemented" $ do
            post $ FineTunesByTextCancelR "ft-AF1WoRqd3aJAHsqc9NY7iL8F"
            statusIs 501

    describe "postChatCompletionsR" $
        it "returns 501 Not Implemented" $ do
            post ChatCompletionsR
            statusIs 501

    describe "postCompletionsR" $
        it "returns 501 Not Implemented" $ do
            post CompletionsR
            statusIs 501

    describe "postEditsR" $
        it "returns 501 Not Implemented" $ do
            post EditsR
            statusIs 501

    describe "postEmbeddingsR" $
        it "returns 501 Not Implemented" $ do
            post EmbeddingsR
            statusIs 501

    describe "postFilesR" $
        it "returns 501 Not Implemented" $ do
            post FilesR
            statusIs 501

    describe "postFineTunesR" $
        it "returns 501 Not Implemented" $ do
            post FineTunesR
            statusIs 501

    describe "postImagesGenerationsR" $
        it "returns 501 Not Implemented" $ do
            post ImagesGenerationsR
            statusIs 501

    describe "postImagesEditsR" $
        it "returns 501 Not Implemented" $ do
            post ImagesEditsR
            statusIs 501

    describe "postImagesVariationsR" $
        it "returns 501 Not Implemented" $ do
            post ImagesVariationsR
            statusIs 501

    describe "postModerationsR" $
        it "returns 501 Not Implemented" $ do
            post ModerationsR
            statusIs 501

    describe "postAudioTranscriptionsR" $
        it "returns 501 Not Implemented" $ do
            post AudioTranscriptionsR
            statusIs 501

    describe "postAudioTranslationsR" $
        it "returns 501 Not Implemented" $ do
            post AudioTranslationsR
            statusIs 501

    describe "deleteFilesByTextR" $
        it "returns 501 Not Implemented" $ do
            performMethod "DELETE" $ FilesByTextR "fileId_example"
            statusIs 501

    describe "deleteModelsByTextR" $
        it "returns 501 Not Implemented" $ do
            performMethod "DELETE" $ ModelsByTextR "curie:ft-acmeco-2021-03-03-21-44-20"
            statusIs 501

    describe "getFilesByTextContentR" $
        it "returns 501 Not Implemented" $ do
            get $ FilesByTextContentR "fileId_example"
            statusIs 501

    describe "getFilesR" $
        it "returns 501 Not Implemented" $ do
            get FilesR
            statusIs 501

    describe "getFineTunesByTextEventsR" $
        it "returns 501 Not Implemented" $ do
            get $ FineTunesByTextEventsR "ft-AF1WoRqd3aJAHsqc9NY7iL8F"
            statusIs 501

    describe "getFineTunesR" $
        it "returns 501 Not Implemented" $ do
            get FineTunesR
            statusIs 501

    describe "getModelsR" $
        it "returns 501 Not Implemented" $ do
            get ModelsR
            statusIs 501

    describe "getFilesByTextR" $
        it "returns 501 Not Implemented" $ do
            get $ FilesByTextR "fileId_example"
            statusIs 501

    describe "getFineTunesByTextR" $
        it "returns 501 Not Implemented" $ do
            get $ FineTunesByTextR "ft-AF1WoRqd3aJAHsqc9NY7iL8F"
            statusIs 501

    describe "getModelsByTextR" $
        it "returns 501 Not Implemented" $ do
            get $ ModelsByTextR "text-davinci-001"
            statusIs 501
