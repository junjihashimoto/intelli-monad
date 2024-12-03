{
  nixConfig = {
    bash-prompt = "\[intelli-monad$(__git_ps1 \" (%s)\")\]$ ";
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };
  inputs.flake-compat = {
    url = "github:edolstra/flake-compat";
    flake = false;
  };

  outputs = { self, nixpkgs, utils, flake-compat  }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {inherit system;};
      in
      {
        devShell = with pkgs; mkShell {
          buildInputs = [
            git
            cabal-install
            zlib
            haskell.compiler.ghc983
          ];
          shellHook = ''
            source ${git}/share/bash-completion/completions/git-prompt.sh
          '';
        };
      });
}
