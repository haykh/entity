{
  pkgs ? import <nixpkgs> { },
  arch ? "native",
  gpu ? "none",
}:

let
  gpuUpper = pkgs.lib.toUpper gpu;
  name = "kokkos";
  version = "4.5.01";
  compilerPkgs = {
    "HIP" = with pkgs.rocmPackages; [
      rocm-core
      clr
      rocthrust
      rocprim
      rocminfo
      rocm-smi
    ];
    "NONE" = [
      pkgs.gcc13
    ];
  };
  cmakeFlags = {
    "HIP" = [
      "-D CMAKE_C_COMPILER=hipcc"
      "-D CMAKE_CXX_COMPILER=hipcc"
    ];
    "NONE" = [ ];
  };
  getArch =
    _:
    if gpu != "none" && arch == "native" then
      throw "Please specify an architecture when the GPU support is enabled. Available architectures: https://kokkos.org/kokkos-core-wiki/keywords.html#architectures"
    else
      pkgs.lib.toUpper arch;

in
pkgs.stdenv.mkDerivation {
  pname = "${name}";
  version = "${version}";
  src = pkgs.fetchgit {
    url = "https://github.com/kokkos/kokkos/";
    rev = "v${version}";
    sha256 = "sha256-cI2p+6J+8BRV5fXTDxxHTfh6P5PeeLUiF73o5zVysHQ=";
  };

  nativeBuildInputs = with pkgs; [
    cmake
  ];

  propagatedBuildInputs = compilerPkgs.${gpuUpper};

  cmakeFlags = [
    "-D CMAKE_CXX_STANDARD=17"
    "-D CMAKE_CXX_EXTENSIONS=OFF"
    "-D CMAKE_POSITION_INDEPENDENT_CODE=TRUE"
    "-D Kokkos_ARCH_${getArch { }}=ON"
    (if gpu != "none" then "-D Kokkos_ENABLE_${gpuUpper}=ON" else "")
    "-D CMAKE_BUILD_TYPE=Release"
  ] ++ cmakeFlags.${gpuUpper};

  enableParallelBuilding = true;
}
