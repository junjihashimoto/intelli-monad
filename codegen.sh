#!/usr/bin/env nix-shell
#!nix-shell -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/9e4e0807d2142d17f463b26a8b796b3fe20a3011.tar.gz
#!nix-shell -i bash
#!nix-shell -p openapi-generator-cli
openapi-generator-cli generate -i openai-openapi/openapi.yaml -g haskell-http-client -o openai-client
openapi-generator-cli generate -i openai-openapi/openapi.yaml -g haskell -o openai-servant
openapi-generator-cli generate -i openai-openapi/openapi.yaml -g haskell-yesod -o openai-yesod
