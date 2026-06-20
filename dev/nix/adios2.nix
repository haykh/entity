{
  pkgs ? import <nixpkgs> { },
  hdf5,
  mpi,
}:

let
  name = "adios2";
  version = "2.11.0";
  cmakeFlags = {
    CMAKE_CXX_STANDARD = "20";
    CMAKE_CXX_EXTENSIONS = "OFF";
    CMAKE_POSITION_INDEPENDENT_CODE = "TRUE";
    BUILD_SHARED_LIBS = "ON";
    ADIOS2_USE_HDF5 = if hdf5 then "ON" else "OFF";
    ADIOS2_USE_Python = "OFF";
    ADIOS2_USE_Fortran = "OFF";
    ADIOS2_USE_ZeroMQ = "OFF";
    BUILD_TESTING = "OFF";
    ADIOS2_BUILD_EXAMPLES = "OFF";
    ADIOS2_USE_MPI = if mpi then "ON" else "OFF";
    ADIOS2_HAVE_HDF5_VOL = if (mpi && hdf5) then "ON" else "OFF";
    CMAKE_BUILD_TYPE = "Release";
  };
  stdenv = pkgs.gcc13Stdenv;
in
stdenv.mkDerivation {
  pname = "${name}${if hdf5 then "-hdf5" else ""}${if mpi then "-mpi" else ""}";
  version = "${version}";
  src = pkgs.fetchgit {
    url = "https://github.com/ornladios/ADIOS2/";
    rev = "v${version}";
    sha256 = "sha256-yHPI///17poiCEb7Luu5qfqxTWm9Nh+o9r57mZT26U0=";
  };

  nativeBuildInputs = with pkgs; [
    cmake
    perl
  ];

  propagatedBuildInputs = [
    pkgs.libgcc
  ]
  ++ (
    if hdf5 then
      (
        if mpi then
          [
            pkgs.hdf5-mpi
          ]
        else
          [ pkgs.hdf5-cpp ]
      )
    else
      (
        if mpi then
          [
            pkgs.mpi
          ]
        else
          [ ]
      )
  );

  configurePhase = ''
    cmake -B build $src ${
      pkgs.lib.attrsets.foldlAttrs (
        acc: key: value:
        acc + " -D ${key}=${value}"
      ) "" cmakeFlags
    }
  '';

  buildPhase = ''
    cmake --build build -j
  '';

  installPhase = ''
    sed -i '/if(CMAKE_INSTALL_COMPONENT/,/^[[:space:]]&endif()$/d' build/cmake/install/post/cmake_install.cmake
    cmake --install build --prefix $out
    chmod +x build/cmake/install/post/generate-adios2-config.sh
    sh build/cmake/install/post/generate-adios2-config.sh $out
  '';

  enableParallelBuilding = true;
}
