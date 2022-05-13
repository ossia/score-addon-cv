#!/bin/bash
rm -rf release
mkdir -p release

cp -rf CompVis *.{hpp,cpp,txt,json} LICENSE release/

mv release score-addon-compvis
7z a score-addon-compvis.zip score-addon-compvis
