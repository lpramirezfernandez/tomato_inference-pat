[app]
title = Tomato Inference
package.name = tomatoinference
package.domain = org.tomatoinference

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,env

version = 0.1

# NOTE: opencv-python-headless is used instead of opencv-python because
# the full desktop build with Qt/GTK bindings does not compile reliably
# under python-for-android. If the opencv recipe still fails on your
# machine, the fallback is to migrate render_dashboard()'s drawing calls
# from cv2 to Pillow (PIL), which packages far more reliably on Android.
requirements = python3,kivy==2.3.0,opencv-python-headless,numpy,python-dotenv,requests,inference-sdk,plyer,pillow

orientation = portrait
fullscreen = 0

# Camera + internet (Roboflow API calls) + storage (file picker / saving results)
android.permissions = INTERNET,CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a

# Make sure the .env file with ROBOFLOW_API_KEY ships inside the APK.
# For a production app, prefer setting the key via a build-time secret
# instead of bundling .env directly.
android.add_assets = .env

[buildozer]
log_level = 2
warn_on_root = 1
