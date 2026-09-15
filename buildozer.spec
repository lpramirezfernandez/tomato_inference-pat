[app]
title = Tomato Inference
package.name = tomatoinference
package.domain = org.example

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json

version = 0.1.0

requirements = python3,kivy,numpy,opencv

orientation = portrait
fullscreen = 0

[buildozer]
log_level = 2
warn_on_root = 1
android.permissions = INTERNET,CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 31
android.minapi = 21
android.ndk = 25b
android.accept_sdk_license = True
