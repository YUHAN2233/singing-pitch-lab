#!/bin/bash
set -euo pipefail
# FFmpeg n9.0.2, signed upstream tag; no third-party codec libraries.
commit=946fcce07b6dcd0331c8cc609192aeff5e1924f8
mkdir -p third_party_sources native-build native
curl --fail --location --retry 3 \
  "https://github.com/FFmpeg/FFmpeg/archive/$commit.tar.gz" \
  -o third_party_sources/ffmpeg-9.0.2.tar.gz
tar -xzf third_party_sources/ffmpeg-9.0.2.tar.gz -C native-build
cd "native-build/FFmpeg-$commit"
./configure --disable-autodetect --disable-network --disable-gpl --disable-nonfree \
  --disable-version3 --disable-doc --disable-debug --disable-ffplay --disable-ffprobe \
  --disable-avdevice --disable-x86asm --enable-small
make -j"$(sysctl -n hw.ncpu)" > build.log 2>&1 || { tail -100 build.log; exit 1; }
cp ffmpeg ../../native/ffmpeg
./ffmpeg -version > ../../FFMPEG_BUILD.txt
./ffmpeg -L >> ../../FFMPEG_BUILD.txt
cp COPYING.LGPLv2.1 ../../third_party_sources/FFMPEG-LGPL-2.1.txt
cp config.h config_components.h ../../third_party_sources/
cp ffbuild/config.log ../../third_party_sources/
cd ../..
cp packaging/macos/build_ffmpeg.sh third_party_sources/
shasum -a 256 third_party_sources/ffmpeg-9.0.2.tar.gz >> FFMPEG_BUILD.txt
