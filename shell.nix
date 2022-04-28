let

#   pkgs = builtins.fetchTarball {
#     url = "https://github.com/NixOS/nixpkgs/archive/nixpkgs-unstable.tar.gz";
#     sha256 = "sha256:0x5j9q1vi00c6kavnjlrwl3yy1xs60c34pkygm49dld2sgws7n0a";
# };

pkgs= builtins.fetchGit {
  # Descriptive name to make the store path easier to identify
  name = "nixos-unstable-2018-09-12";
  url = "https://github.com/nixos/nixpkgs/";
  ref = "refs/heads/master";
  rev = "fe237597d151a33b6aab54f1f5a0af6353c74d04";
};

myHaskellPackageOverlay = self: super: {
  myHaskellPackages = super.haskell.packages.ghc922.override {
    overrides = hself: hsuper: rec {

      hmatrix-sundials = super.haskell.lib.dontCheck (
        hself.callCabal2nix "hmatrix-sundials" (builtins.fetchGit {
          url = "https://github.com/novadiscovery/hmatrix-sundials";
          rev = "76bfee5b5a8377dc3f7161514761946a60d4834a";
        }) { sundials_arkode          = self.sundials;
             sundials_cvode           = self.sundials;
             klu                      = self.suitesparse;
             suitesparseconfig        = self.suitesparse;
             sundials_sunlinsolklu    = self.sundials;
             sundials_sunmatrixsparse = self.sundials;
           });

      inline-r = super.haskell.lib.dontCheck (
        hself.callCabal2nixWithOptions "inline-r" (builtins.fetchGit {
          url = "https://github.com/tweag/HaskellR";
          rev = "c3ba1023480e26ade420896bcb629ceaad59f308";
        }) "--subpath inline-r" { });

      singletons = self.haskell.lib.addBuildDepends(
        hself.callHackageDirect {
          pkg = "singletons";
          ver = "3.0.1";
          sha256 = "sha256-ixHWZae6AxjRUldMgpYolXBGsOUT5ZVIw9HZkxrhHQ0=";
        } { }) [ ];

      singletons-th = self.haskell.lib.addBuildDepends(
        hself.callHackageDirect {
          pkg = "singletons-th";
          ver = "3.1";
          sha256 = "sha256-34nyestxt8KNTSlmr1Y8nElNXa/wZ1+fuLEUVjZX8dk=";
        } { }) [ ];

      retry = self.haskell.lib.dontCheck hsuper.retry;

      Chart = self.haskell.lib.doJailbreak hsuper.Chart;

      size-based = self.haskell.lib.doJailbreak hsuper.size-based;

      random-fu = super.haskell.lib.dontCheck (
        hself.callCabal2nixWithOptions "random-fu" (builtins.fetchGit {
          url = "https://github.com/haskell-numerics/random-fu";
          rev = "18a6ba6b29c7ca3b3ff34ea6ca0eca910da72726";
        }) "--subpath random-fu" { });

      rvar = super.haskell.lib.dontCheck (
        hself.callCabal2nixWithOptions "rvar" (builtins.fetchGit {
          url = "https://github.com/haskell-numerics/random-fu";
          rev = "18a6ba6b29c7ca3b3ff34ea6ca0eca910da72726";
        }) "--subpath rvar" { });

    };
  };
};

in

{ nixpkgs ? import pkgs {
  config.allowBroken = true;
  config.allowUnsupportedSystem = true;
  overlays = [ myHaskellPackageOverlay ]; }
}:

let

  pkgs = nixpkgs;

  R-with-my-packages = pkgs.rWrapper.override{
    packages = with pkgs.rPackages; [
      ggplot2
      ggridges
      outbreaks
      tidyverse
      (buildRPackage {
        name = "smcsamplers";
        src = pkgs.fetchFromGitHub {
          owner = "pierrejacob";
          repo = "smcsamplers";
          rev = "097192f7d5df520d9b026d442dfec493a3051374";
          sha256 = "00facn1ylcbai4sbcidpp991899csz2ppmmkv0khvqxfncddr0f2";
        };
        propagatedBuildInputs = [ coda MASS mvtnorm loo shape rstan tidyverse doParallel igraph ggraph doRNG reshape2 ];
    })
    ]; };

  haskellDeps = ps: with ps; [
    (pkgs.haskell.lib.dontCheck ad)
    hmatrix-sundials
    cassava
    bytestring
    katip
    monad-extras
    vector
    QuickCheck
    random-fu
    Cabal
    (pkgs.haskell.lib.dontCheck inline-r)
    mtl
    formatting
    katip
    Chart
    Chart-diagrams
    diagrams-svg
  ];

  # FIXME: One day I will be able to use julia with nix
  environment.systemPackages = with pkgs; [ julia_17-bin ];

in

pkgs.stdenv.mkDerivation {
  name = "xxx";

  buildInputs = [
    pkgs.libintlOrEmpty
    R-with-my-packages
    pkgs.pandoc
    pkgs.texlive.combined.scheme-full
    (pkgs.myHaskellPackages.ghcWithPackages haskellDeps)
  ];
  shellHook = ''
    R_LIBS_USER=$(Rscript -e ".libPaths()" | cut -c 6- | sed 's/[ \t]*$//' | sed 's/"//g' | sed -z 's/\n/:/g;s/:'''$/\n/' | sed 's/ //g')
    export R_LIBS_USER
    '';
}
