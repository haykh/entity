{
  pkgs ? import <nixpkgs> { },
  stdenv,
  arch,
  gpu,
}:

let
  name = "kokkos";
  pversion = "5.0.1";
  compilerPkgs = {
    "HIP" = with pkgs.rocmPackages; [
      clang
      rocm-core
      clr
      rocthrust
      rocprim
      rocminfo
      rocm-smi
      pkgs.clang-tools
    ];
    "CUDA" = with pkgs.cudaPackages; [
      pkgs.clang-tools
      cudatoolkit
      cuda_cudart
      pkgs.gcc13
    ];
    "NONE" = [
      pkgs.clang-tools
      pkgs.gcc13
    ];
  };
  getArch =
    _:
    if gpu != "NONE" && arch == "NATIVE" then
      throw "Please specify an architecture when the GPU support is enabled. Available architectures: https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html#gpu-architectures"
    else
      arch;
  cmakeExtraFlags = {
    "HIP" = [
      "-D Kokkos_ENABLE_HIP=ON"
      "-D Kokkos_ARCH_${getArch { }}=ON"
      "-D GPU_TARGETS=${builtins.replaceStrings [ "amd_" ] [ "" ] (pkgs.lib.toLower (getArch { }))}"
      "-D CMAKE_CXX_COMPILER=clang++"
    ];
    "CUDA" = [
      "-D Kokkos_ENABLE_CUDA=ON"
      "-D Kokkos_ARCH_${getArch { }}=ON"
      "-D CMAKE_CXX_COMPILER=$WRAPPER_PATH"
    ];
    "NONE" = [ ];
  };
in
pkgs.stdenv.mkDerivation rec {
  pname = "${name}";
  version = "${pversion}";
  src = pkgs.fetchgit {
    url = "https://github.com/kokkos/kokkos/";
    rev = "${pversion}";
    sha256 = "sha256-ChpwGBwE7sNovjdAM/iCeOqqwGufKxAh5vQ3qK6aFBU=";
  };

  nativeBuildInputs = with pkgs; [
    cmake
  ];

  propagatedBuildInputs = compilerPkgs.${gpu};

  patchPhase =
    if gpu == "CUDA" then
      ''
        export WRAPPER_PATH="$(mktemp -d)/nvcc_wrapper"
        cp ${src}/bin/nvcc_wrapper $WRAPPER_PATH
        substituteInPlace $WRAPPER_PATH --replace-fail "#!/usr/bin/env bash" "#!${stdenv.shell}"
        chmod +x "$WRAPPER_PATH"
      ''
    else
      "";

  configurePhase = ''
    cmake -B build -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_STANDARD=20 \
      -D CMAKE_CXX_EXTENSIONS=OFF \
      -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE \
      ${pkgs.lib.concatStringsSep " " cmakeExtraFlags.${gpu}} \
      -D CMAKE_INSTALL_PREFIX=$out
  '';

  buildPhase = ''
    cmake --build build -j
  '';

  installPhase = ''
    cmake --install build
  '';
}
