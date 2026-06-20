{
  pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = gpu == "CUDA";
  },
  gpu ? "NONE",
  arch ? "NATIVE",
  hdf5 ? false,
  mpi ? false,
  extra ? "",
}:

let
  gpuUpper = pkgs.lib.toUpper gpu;
  archUpper = pkgs.lib.toUpper arch;
  name = "nt2";
  adios2Pkg = (pkgs.callPackage ./adios2.nix { inherit pkgs mpi hdf5; });
  kokkosPkg = (
    pkgs.callPackage ./kokkos.nix {
      inherit pkgs;
      stdenv = pkgs.stdenv;
      arch = archUpper;
      gpu = gpuUpper;
    }
  );
  extraPkgs = map (name: pkgs.${name}) (
    pkgs.lib.filter (s: s != "") (pkgs.lib.splitString "," extra)
  );
  envVars = {
    compiler = {
      NONE = {
        CXX = "g++";
        CC = "gcc";
      };
      HIP = {
        CXX = "clang++";
        CC = "clang";
      };
      CUDA = { };
    };
  };
in
pkgs.mkShell {
  name =
    "${name}"
    + (if gpu != "NONE" then "-${pkgs.lib.toLower gpu}" else "")
    + (if mpi then "-mpi" else "");
  nativeBuildInputs =
    with pkgs;
    [
      zlib
      cmake

      adios2Pkg
      kokkosPkg

      python314

      cmake-format
      cmake-lint
      neocmakelsp
      black
      pyright
      taplo
      vscode-langservers-extracted
    ]
    ++ extraPkgs;

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ];

  shellHook = ''
    BLUE='\033[0;34m'
    NC='\033[0m'

    echo "following environment variables are set:"
  ''
  + pkgs.lib.concatStringsSep "" (
    pkgs.lib.mapAttrsToList (
      category: vars:
      pkgs.lib.concatStringsSep "" (
        pkgs.lib.mapAttrsToList (name: value: ''
          export ${name}=${value}
          echo -e "  ''\${BLUE}${name}''\${NC}=${value}"
        '') vars.${gpuUpper}
      )
    ) envVars
  )
  + ''
    echo ""
    echo -e "${name} nix-shell activated"
  '';

}
