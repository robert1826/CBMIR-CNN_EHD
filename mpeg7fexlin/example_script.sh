#!/bin/sh

# robert : lib32-libjpeg6-turbo (archlinux) was needed as dep
# as i took the /usr/lib32/libjpeg.so.62 to put with solibs

if [ -n "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/solibs
else
  LD_LIBRARY_PATH=$(pwd)/solibs
fi
export LD_LIBRARY_PATH

./MPEG7Fex EHD imageList.txt ehd_out.txt
