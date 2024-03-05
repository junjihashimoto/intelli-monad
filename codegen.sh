#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/9e4e0807d2142d17f463b26a8b796b3fe20a3011.tar.gz
#!nix-shell -i bash
#!nix-shell -p openapi-generator-cli -p hpack


openapi-generator-cli generate -i openai-openapi/openapi.yaml -g haskell-http-client -o openai-client-gen
mv openai-client-gen/openai.cabal openai-client-gen/openai-client-gen.cabal
openapi-generator-cli generate -i openai-openapi/openapi.yaml -g haskell -o openai-servant-gen
mv openai-servant-gen/openai.cabal openai-servant-gen/openai-servant-gen.cabal

sed -i \
    -e 's/Author Name Here/Junji Hashimoto/g' \
    -e 's/author.name@email.com/junji.hashimoto@gmail.com/g' \
    -e 's/YEAR - AUTHOR/2023 - Junji Hashimoto/g' \
    -e 's/UnspecifiedLicense/MIT/g' \
    -e 's/name:\(.*openai\)$/name:\1-client/g' \
    openai-client-gen/openai-client-gen.cabal

sed -i \
    -e 's/Author Name Here/Junji Hashimoto/g' \
    -e 's/author.name@email.com/junji.hashimoto@gmail.com/g' \
    -e 's/YEAR - AUTHOR/2023 - Junji Hashimoto/g' \
    -e 's/UnspecifiedLicense/MIT/g' \
    -e 's/name:\(.*openai\)$/name:\1-servant-gen/g' \
    openai-servant-gen/openai-servant-gen.cabal

sed -i -e 's/CreateTranscriptionRequestModel/String/g' openai-servant-gen/lib/OpenAI/API.hs
sed -i -e 's/Paths_openai//g' openai-client-gen/openai-client-gen.cabal
sed -i -e 's/CreateTranscriptionRequestModel/String/g' openai-client-gen/lib/OpenAI/Model.hs
sed -i -e 's/AnyType/A.Value/g' openai-client-gen/lib/OpenAI/Model.hs openai-client-gen/lib/OpenAI/ModelLens.hs

curl https://www.stackage.org/lts-22.12/cabal.config |\
sed -e 's/with-compiler: .*$//g'  > cabal.project.freeze
