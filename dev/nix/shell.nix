{
  pkgs ? import <nixpkgs> { },
  mpi ? false,
  hdf5 ? false,
  gpu ? "none",
  arch ? "native",
}:

let
  name = "entity-dev";
  adios2Pkg = (pkgs.callPackage ./adios2.nix { inherit pkgs mpi hdf5; });
  kokkosPkg = (pkgs.callPackage ./kokkos.nix { inherit pkgs arch gpu; });
in
pkgs.mkShell {
  name = "${name}-env";
  nativeBuildInputs = with pkgs; [
    zlib
    cmake

    clang-tools

    adios2Pkg
    kokkosPkg

    python312
    python312Packages.jupyter

    cmake-format
    cmake-lint
    neocmakelsp
    black
    pyright
    taplo
    vscode-langservers-extracted
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ([
    pkgs.stdenv.cc.cc
    pkgs.zlib
  ]);

  shellHook = ''
    BLUE='\033[0;34m'
    NC='\033[0m'

    echo ""
    echo -e "${name} nix-shell activated"
  '';
}
