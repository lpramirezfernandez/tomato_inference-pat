[app]

title = Tomato Inference
package.name = tomatoinference
package.domain = org.tomatoinference

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,env

version = 0.1

requirements = python3,kivy==2.3.0,opencv-python-headless,numpy,python-dotenv,requests,inference-sdk,plyer,pillow

orientation = portrait
fullscreen = 0

# Android permissions
android.permissions = INTERNET,CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# Android versions
android.api = 33
android.minapi = 24

# Build only for modern 64-bit Android devices
android.archs = arm64-v8a

# Include .env in APK
android.add_assets = .env


[buildozer]

log_level = 2
warn_on_root = 1
